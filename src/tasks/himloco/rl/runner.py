import os
import inspect
import math
from typing import Any

import torch
import wandb
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)

from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner

class _OnnxPolicyWrapper(torch.nn.Module):
  """Expose HIMActorCritic.act_inference as a standard ONNX forward."""

  def __init__(self, actor_critic):
    super().__init__()
    self.actor_critic = actor_critic

  def forward(self, obs):
    return self.actor_critic.act_inference(obs)


def _onnx_export_kwargs_single_file() -> dict:
  """
  构建 ONNX 导出的关键字参数，目的是跨不同的 PyTorch 版本确保 ONNX 模型导出为单一文件。
  """
  try:
    params = inspect.signature(torch.onnx.export).parameters # 获取当前 PyTorch 版本中 torch.onnx.export 函数的所有参数
  except (TypeError, ValueError):
    return {} # 如果获取失败，返回空字典

  # 根据不同 PyTorch 版本支持的参数名称，禁用外部数据存储，强制把权重保存在同一个 .onnx 文件中
  if "external_data" in params:
    return {"external_data": False}
  if "use_external_data_format" in params:
    return {"use_external_data_format": False}
  return {} # 如果都不包含，说明默认就是单文件，返回空字典


def _inline_external_onnx_data(onnx_path: str) -> None:
  """如果 ONNX 导出时依然生成了外部的张量数据文件（如模型过大时），此函数将其强制合并回单一的 ONNX 文件中。"""
  data_path = f"{onnx_path}.data"        # 拼接出可能存在的外部数据文件路径
  if not os.path.exists(data_path):      # 如果不存在这个外部数据文件
    return                               # 说明已经是单文件了，直接返回

  try:
    import onnx                          # 延迟导入 onnx 库

    model = onnx.load(onnx_path, load_external_data=True) # 读取 ONNX 模型，并连同外部数据一起加载到内存
    onnx.save_model(model, onnx_path, save_as_external_data=False) # 重新保存模型，并明确指定【不】保存为外部数据格式（即合并到主文件中）
    if os.path.exists(data_path):        # 如果旧的外部数据文件还存在
      os.remove(data_path)               # 将其删除，清理垃圾
    print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}") # 打印成功合并的日志
  except Exception as exc:
    print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}") # 打印合并失败的警告信息


# ==============================
# Him 环境特征转换 Wrapper
# ==============================
class HimVecEnvWrapper(RslRlVecEnvWrapper):
  """Adapt mjlab TensorDict observations to the original HIM tensor env API."""
    
  def __init__(self, env, clip_actions: float | None = None):
    if isinstance(env, RslRlVecEnvWrapper):
      self.env = env.env
      self.clip_actions = env.clip_actions if clip_actions is None else clip_actions
      self.num_envs = env.num_envs
      self.device = env.device
      self.max_episode_length = env.max_episode_length
      self.num_actions = env.num_actions
    else:
      super().__init__(env, clip_actions=clip_actions)

    self.obs_buf: torch.Tensor | None = None
    self.privileged_obs_buf: torch.Tensor | None = None
    self.rew_buf: torch.Tensor | None = None
    self.reset_buf: torch.Tensor | None = None
    self.extras: dict[str, Any] = {}

  @property
  def num_obs(self) -> int:
    return self._group_obs_dim("actor")

  @property
  def num_one_step_obs(self) -> int:
    history_length = self._actor_history_length()
    actor_obs_dim = self.num_obs
    if actor_obs_dim % history_length != 0:
      raise ValueError(
        "HIM actor observation dimension must be divisible by the actor "
        f"history_length: num_obs={actor_obs_dim}, "
        f"history_length={history_length}."
      )
    return actor_obs_dim // history_length

  @property
  def num_privileged_obs(self) -> int | None:
    if "critic" not in self.unwrapped.observation_manager.group_obs_dim:
      return None
    return self._group_obs_dim("critic")

  def reset(self, env_ids=None):
    if env_ids is None:
      obs_td, extras = super().reset()
    else:
      obs_dict, extras = self.env.reset(env_ids=env_ids)
      obs_td = TensorDict(obs_dict, batch_size=[self.num_envs])
    obs, privileged_obs = self._split_observations(obs_td)
    self._update_buffers(obs, privileged_obs, None, None, extras)
    return obs, extras

  def get_observations(self) -> torch.Tensor:
    obs_td = super().get_observations()
    obs, privileged_obs = self._split_observations(obs_td)
    self._update_buffers(obs, privileged_obs, None, None, self.extras)
    return obs

  def get_privileged_observations(self) -> torch.Tensor | None:
    if self.privileged_obs_buf is None:
      self.get_observations()
    return self.privileged_obs_buf

  def step(self, actions: torch.Tensor):
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    previous_critic_obs = (
      self.privileged_obs_buf.clone()
      if self.privileged_obs_buf is not None
      else None
    )
    obs_td, rewards, dones, infos = super().step(actions)
    obs, privileged_obs = self._split_observations(obs_td)

    rewards = rewards.view(-1)
    dones = dones.view(-1).to(dtype=torch.long)
    termination_ids = torch.nonzero(dones > 0, as_tuple=False).flatten()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    termination_privileged_obs = self._termination_critic_obs(
      infos, termination_ids, previous_critic_obs, critic_obs
    )

    self._update_buffers(obs, privileged_obs, rewards, dones, infos)
    return (
      obs,
      privileged_obs,
      rewards,
      dones,
      infos,
      termination_ids,
      termination_privileged_obs,
    )

  def _split_observations(
    self, obs_td: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    actor_obs = self._get_group_tensor(obs_td, "actor", required=True)
    critic_obs = self._get_group_tensor(obs_td, "critic", required=False)
    return actor_obs, critic_obs

  def _get_group_tensor(
    self, obs_td: TensorDict, group_name: str, required: bool
  ) -> torch.Tensor | None:
    try:
      value = obs_td[group_name]
    except KeyError:
      value = None

    if value is None:
      if required:
        raise KeyError(f"Observation group '{group_name}' is required for HIM.")
      return None
    if not isinstance(value, torch.Tensor):
      raise TypeError(
        f"HIM requires concatenated tensor observations for group "
        f"'{group_name}', got {type(value)!r}."
      )
    if value.dim() > 2:
      value = value.flatten(start_dim=1)
    if group_name == "actor":
      value = self._actor_obs_for_him(value)
    return value

  def _group_obs_dim(self, group_name: str) -> int:
    group_dim = self.unwrapped.observation_manager.group_obs_dim[group_name]
    if not isinstance(group_dim, tuple):
      raise ValueError(
        f"HIM requires concatenated observations for group '{group_name}'."
      )
    return int(math.prod(group_dim))

  def _actor_history_length(self) -> int:
    actor_cfg = self.cfg.observations.get("actor")
    history_length = getattr(actor_cfg, "history_length", None)
    if history_length is None or history_length <= 0:
      return 1
    return int(history_length)

  def _actor_obs_for_him(self, obs: torch.Tensor) -> torch.Tensor:
    """Convert mjlab term-major history into HIM step-major history.

    mjlab's CircularBuffer exposes history oldest-to-newest, and
    ObservationManager flattens each term as term-major
    ``[term0_oldest...term0_newest, term1_oldest...]``. HIMActorCritic reads
    ``obs[:, :num_one_step_obs]`` as the current single-step observation, so the
    wrapper flips the history axis after rebuilding per-step observations.
    """
    history_length = self._actor_history_length()
    if history_length <= 1:
      return obs

    base_dims = self._actor_term_base_dims(history_length)
    expected_dim = sum(base_dims) * history_length
    if obs.shape[1] != expected_dim:
      raise ValueError(
        f"Unexpected actor observation dimension for HIM history conversion: "
        f"got {obs.shape[1]}, expected {expected_dim}."
      )
    chunks = torch.split(
      obs, [base_dim * history_length for base_dim in base_dims], dim=1
    )
    term_histories = [
      chunk.reshape(obs.shape[0], history_length, base_dim)
      for chunk, base_dim in zip(chunks, base_dims, strict=True)
    ]
    step_major = torch.cat(term_histories, dim=2).flip(dims=[1])
    return step_major.reshape(obs.shape[0], -1)

  def _actor_term_base_dims(self, history_length: int) -> list[int]:
    term_dims = self.unwrapped.observation_manager.group_obs_term_dim["actor"]
    base_dims: list[int] = []
    for dims in term_dims:
      total_dim = int(math.prod(dims))
      if total_dim % history_length != 0:
        raise ValueError(
          "HIM actor term observation dimension must be divisible by the "
          f"actor history_length: term_dim={total_dim}, "
          f"history_length={history_length}."
        )
      base_dims.append(total_dim // history_length)
    return base_dims

  def _termination_critic_obs(
    self,
    infos: dict[str, Any],
    termination_ids: torch.Tensor,
    previous_critic_obs: torch.Tensor | None,
    next_critic_obs: torch.Tensor,
  ) -> torch.Tensor:
    if len(termination_ids) == 0:
      return next_critic_obs[:0]

    terminal_obs = self._extract_terminal_group_obs(infos, "critic")
    if terminal_obs is not None:
      if terminal_obs.shape[0] == self.num_envs:
        return terminal_obs[termination_ids]
      if terminal_obs.shape[0] == len(termination_ids):
        return terminal_obs
      raise ValueError(
        "Terminal critic observation batch does not match full env batch or "
        f"done subset: got {terminal_obs.shape[0]}, num_envs={self.num_envs}, "
        f"num_done={len(termination_ids)}."
      )

    if previous_critic_obs is not None:
      infos["him_terminal_critic_obs_source"] = "previous_critic_obs_fallback"
      return previous_critic_obs[termination_ids]

    infos["him_terminal_critic_obs_source"] = "zero_fallback_no_previous_obs"
    return torch.zeros_like(next_critic_obs[termination_ids])

  def _extract_terminal_group_obs(
    self, infos: dict[str, Any], group_name: str
  ) -> torch.Tensor | None:
    for key in (
      "terminal_observation",
      "terminal_observations",
      "final_observation",
      "final_observations",
    ):
      if key not in infos:
        continue
      terminal = infos[key]
      if isinstance(terminal, TensorDict) or hasattr(terminal, "keys"):
        if group_name in terminal.keys():
          value = terminal[group_name]
        elif "observations" in terminal.keys() and group_name in terminal["observations"]:
          value = terminal["observations"][group_name]
        else:
          continue
      else:
        value = terminal
      if isinstance(value, torch.Tensor):
        if value.dim() > 2:
          value = value.flatten(start_dim=1)
        return value
    return None

  def _update_buffers(
    self,
    obs: torch.Tensor,
    privileged_obs: torch.Tensor | None,
    rewards: torch.Tensor | None,
    dones: torch.Tensor | None,
    infos: dict[str, Any] | None,
  ) -> None:
    self.obs_buf = obs
    self.privileged_obs_buf = privileged_obs
    if rewards is not None:
      self.rew_buf = rewards
    if dones is not None:
      self.reset_buf = dones
    if infos is not None:
      self.extras = infos


# ==============================
# 完整重写的 HIM 专用 Runner
# ==============================
class HIMLocoOnPolicyRunner(HIMOnPolicyRunner):
  env: HimVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir, device='cpu', **kwargs):
    # 1. 确保环境是被 HimVecEnvWrapper 包裹的
    if not isinstance(env, HimVecEnvWrapper):
        env = HimVecEnvWrapper(env)
    
    # 2. 桥接并映射参数（注意此时 train_cfg 是一个字典）
    train_cfg_dict = {
        "runner": {
            "policy_class_name": "HIMActorCritic", 
            "algorithm_class_name": train_cfg["algorithm"].get("class_name", "HIMPPO"),
            "num_steps_per_env": train_cfg["num_steps_per_env"],
            "save_interval": train_cfg["save_interval"],
         },
        "algorithm": train_cfg["algorithm"].copy(),
        "policy": {
            "init_noise_std": train_cfg["actor"]["distribution_cfg"].get("init_std", 1.0),
            "actor_hidden_dims": train_cfg["actor"]["hidden_dims"],
            "critic_hidden_dims": train_cfg["critic"]["hidden_dims"],
            "estimator_hidden_dims": train_cfg["estimator"]["hidden_dims"], 
            "activation": train_cfg["actor"]["activation"],
        }
    }
        
    # 防止 rsl_rl 基类算法解析报错
    import inspect
    from rsl_rl.algorithms import HIMPPO
    
    supported_args = inspect.signature(HIMPPO.__init__).parameters.keys()
    train_cfg_dict["algorithm"] = {
        k: v for k, v in train_cfg_dict["algorithm"].items() 
        if k in supported_args or k == "kwargs"
    }
        
    # 保存属性以供导出使用
    self.logger_type = train_cfg.get("logger", "tensorboard")
    self.empirical_normalization = train_cfg["actor"].get("obs_normalization", True)
        
    # 调用底层 HIMOnPolicyRunner 初始化，直接使用 train.py 传过来的 log_dir
    super().__init__(env=env, train_cfg=train_cfg_dict, log_dir=log_dir, device=device)

  def _export_policy_to_onnx(self, path: str, filename: str):
      """[新增] 将策略网络导出为 ONNX 格式"""
      import torch
      
      model = _OnnxPolicyWrapper(self.alg.actor_critic).to(self.device)
      model.eval()
      
      # 创建符合观测维度的假输入数据
      dummy_input = torch.zeros(1, self.env.num_obs, device=self.device)
      
      export_path = os.path.join(path, filename)
      
      torch.onnx.export(
          model,
          dummy_input,
          export_path,
          input_names=["obs"],
          output_names=["action"],
          opset_version=11, 
          **_onnx_export_kwargs_single_file()
      )
      print(f"[INFO] Successfully exported ONNX policy to: {export_path}")
  
  
  def save(self, path: str, infos=None):
    """
    重写保存逻辑，保证每次保存 PyTorch 检查点的同时，也能生成附加有完整元数据的 ONNX 文件。
    """
    super().save(path, infos)
    
    # 修复路径生成：使用 dirname 避免直接用 split("model") 带来的隐患
    policy_path = os.path.dirname(path) 
    filename = "policy.onnx"
    
    self._export_policy_to_onnx(policy_path, filename)
    
    logger_type = getattr(self, "logger_type", "local")
    run_name: str = (
      wandb.run.name if logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    
    onnx_path = os.path.join(policy_path, filename)
    
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    
    _inline_external_onnx_data(onnx_path)
    
    if logger_type in ["wandb"]:
      wandb.save(onnx_path, base_path=policy_path)
