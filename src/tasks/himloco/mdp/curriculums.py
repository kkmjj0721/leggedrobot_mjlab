from __future__ import annotations # 允许延迟评估类型提示，使得可以在类定义完成前引用类名本身

from typing import TYPE_CHECKING, TypedDict, cast # 导入类型提示相关的工具

import torch # 导入 PyTorch 库，用于张量计算（强化学习环境通常是向量化的，处理成百上千个环境）

from mjlab.entity import Entity # 导入自定义的实体类（代表环境中的物理对象，如机器人）
from mjlab.managers.scene_entity_config import SceneEntityCfg # 导入场景实体配置类

from .velocity_command import UniformVelocityCommandCfg # 导入均匀速度指令配置类

# TYPE_CHECKING 在运行时为 False，仅在静态类型检查（如 mypy）时为 True。
# 这样做可以避免循环导入问题。
if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# 定义一个全局的默认场景实体配置，默认名字为 "robot"
_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


# 定义一个结构化字典类型，用于表示速度课程的“阶段”
class VelocityStage(TypedDict):
  step: int                               # 触发该阶段的训练步数阈值
  lin_vel_x: tuple[float, float] | None   # 线速度 X（前进/后退）的范围 (min, max)，可以为空
  lin_vel_y: tuple[float, float] | None   # 线速度 Y（横向移动）的范围，可以为空
  ang_vel_z: tuple[float, float] | None   # 角速度 Z（旋转）的范围，可以为空


# 定义一个结构化字典类型，用于表示奖励权重课程的“阶段”
class RewardWeightStage(TypedDict):
  step: int     # 触发该阶段的训练步数阈值
  weight: float # 该阶段对应的新权重值


# ---------------------------------------------------------------------------
# 地形等级更新函数
# ---------------------------------------------------------------------------
def terrain_levels_vel(
  env: ManagerBasedRlEnv,          # RL环境实例
  env_ids: torch.Tensor,           # 需要处理的并行环境ID张量（例如 [0, 5, 12]）
  command_name: str,               # 速度指令的名称
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG, # 要评估的资产（默认是robot）
) -> torch.Tensor:
  # 从场景中获取机器人实体对象
  asset: Entity = env.scene[asset_cfg.name]

  # 获取地形和地形生成器
  terrain = env.scene.terrain
  assert terrain is not None # 确保地形存在
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None # 确保地形生成器存在

  # 从指令管理器中获取当前机器人的速度指令
  command = env.command_manager.get_command(command_name)
  assert command is not None

  # 计算机器人行走的实际距离。
  # 使用当前全局坐标的 XY 减去 起始坐标的 XY，然后求 L2 范数（直线距离）
  # asset.data.root_link_pos_w 是机器人的世界坐标，env.scene.env_origins 是起始坐标
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # 判断是否升级：如果走过的距离大于当前地形区块尺寸（size[0]）的一半，则表现良好，准备升级。
  move_up = distance > terrain_generator.size[0] / 2

  # 判断是否降级：
  # 理论最大移动距离 = 目标速度向量的模 * 回合最大时间
  # 如果实际走过的距离 < 理论距离的 50%，说明没达到预期。
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  # 互斥操作：如果已经满足了升级条件，强制将降级条件设为 False
  move_down *= ~move_up

  # 调用地形对象的更新函数，根据 move_up 和 move_down 将指定的机器人们移动到新地形起点
  terrain.update_env_origins(env_ids, move_up, move_down)

  # 返回所有地形等级的平均值，通常用于日志记录（Logging），观察整体训练难度是否在上升
  return torch.mean(terrain.terrain_levels.float())


# ---------------------------------------------------------------------------
# 速度指令更新函数
# ---------------------------------------------------------------------------
def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage], # 传入之前定义的阶段列表
) -> dict[str, torch.Tensor]:
  del env_ids  # 此变量在此函数中未使用，显式删除避免 lint 警告。由于是修改全局配置，不需要区分环境ID。
  
  # 获取负责处理该指令的组件
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  # 将其配置强制转换为 UniformVelocityCommandCfg 类型，以便修改
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  
  # 遍历所有定义的训练阶段
  for stage in velocity_stages:
    # 如果当前全局训练步数超过了该阶段设置的步数
    if env.common_step_counter > stage["step"]:
      # 如果字典中存在该维度且不为空，则更新指令配置中的范围
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
        
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # ...
  }


# ---------------------------------------------------------------------------
# 奖励权重更新函数
# ---------------------------------------------------------------------------
def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,                       # 要修改的奖励项的名称
  weight_stages: list[RewardWeightStage], # 权重阶段列表
) -> torch.Tensor:
  """根据训练步数阶段更新奖励项的权重。"""
  del env_ids  # 未使用。权重配置是全局共享的，不区分具体环境ID。
  
  # 获取对应奖励项的配置对象
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  
  # 遍历权重阶段
  for stage in weight_stages:
    # 如果当前步数大于设定阈值
    if env.common_step_counter > stage["step"]:
      # 将奖励权重覆盖为当前阶段的权重
      reward_term_cfg.weight = stage["weight"]
      
  # 将最终生效的权重打包成张量返回（通常用于日志记录）
  return torch.tensor([reward_term_cfg.weight])