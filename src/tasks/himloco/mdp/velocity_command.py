from __future__ import annotations # 允许类型提示在定义前使用（延迟计算）

from collections.abc import Callable # 导入可调用对象的类型提示
from dataclasses import dataclass, field # 导入数据类，用于优雅地定义配置结构
from typing import TYPE_CHECKING # 用于静态类型检查，避免循环导入

import numpy as np # 导入 NumPy，主要用于可视化时的 CPU 计算
import torch # 导入 PyTorch，用于 GPU 上的张量批处理计算

from mjlab.entity import Entity # 导入物理实体基类（代表机器人）
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg # 导入指令管理器的基类和配置基类
from mjlab.utils.lab_api.math import ( # 导入数学工具函数
  matrix_from_quat, # 四元数转旋转矩阵
  quat_apply,       # 用四元数旋转向量
  wrap_to_pi,       # 将角度限制在 [-pi, pi] 之间
)

# 仅在静态检查时导入，运行时不导入，避免循环依赖
if TYPE_CHECKING:
  import viser # Viser 是一个用于 3D 可视化和 Web GUI 的库
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


# ==============================================================================
# 均匀速度指令类（核心业务逻辑）
# ==============================================================================
class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg # 指定配置类的类型

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env) # 初始化父类

    # 参数校验：如果开启了朝向控制，但没设置朝向范围，则报错
    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    # 参数校验：如果设置了朝向范围，但没开启朝向控制，则报错
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    # 从环境中获取被控制的机器人实体
    self.robot: Entity = env.scene[cfg.entity_name]

    # 初始化各类张量缓存（分配在 GPU 上，大小通常为 [num_envs, ...]）
    # 速度指令缓存：包含线速度 x, 线速度 y, 角速度 z (yaw)
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    # 目标朝向角度缓存
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    # 当前朝向与目标朝向的误差缓存
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    
    # 布尔掩码：标记哪些环境当前正在使用“朝向控制”模式
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # 布尔掩码：标记哪些环境当前被要求“原地站立”（速度为0）
    self.is_standing_env = torch.zeros_like(self.is_heading_env)

    # 性能指标记录：用于 TensorBoard 或日志，记录机器人跟踪指令的误差
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # GUI（图形界面）状态变量，在 create_gui 中被赋值
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None # 遥控器开关
    self._joystick_sliders: list[viser.GuiSliderHandle] = []      # 控制滑块列表
    self._joystick_get_env_idx: Callable[[], int] | None = None   # 获取当前被选中环境索引的回调函数

  @property
  def command(self) -> torch.Tensor:
    """对外暴露的接口，返回当前的速度指令。"""
    return self.vel_command_b

  def _update_metrics(self) -> None:
    """计算并累加跟踪误差指标。"""
    # 计算当前指令的最大持续步数
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    
    # 累加 XY 线速度误差（目标指令 - 机器人机体坐标系下的实际线速度）/ 最大步数（用于做归一化平均）
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    # 累加 Yaw 角速度误差
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """为指定环境（达到重采样时间间隔的机器人）重新生成随机指令。"""
    # 创建一个空张量用于生成随机数
    r = torch.empty(len(env_ids), device=self.device)
    
    # 在配置的范围内，均匀随机采样 X、Y 线速度和 Z 角速度
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    
    # 死区设置：如果合成速度的模长太小（<0.1），就直接将指令归零，避免微弱的指令干扰
    self.vel_command_b[env_ids, :] *= (torch.norm(self.vel_command_b[env_ids, :], dim=1) > 0.1).unsqueeze(1)
    
    # 处理朝向指令逻辑
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      # 随机生成目标朝向
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      # 掷骰子决定这些环境是否应用朝向控制（根据配置的概率 rel_heading_envs）
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
      
    # 掷骰子决定哪些环境要进入“站立不动”模式
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # 初始速度注入机制：
    # 有时候为了让机器人更快学会跑步，会在给出指令的瞬间，直接在物理引擎里把机器人的实际速度设为指令速度（一种作弊式的课程学习手段）。
    init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_vel_env_ids = env_ids[init_vel_mask] # 选出需要注入初始速度的环境
    if len(init_vel_env_ids) > 0:
      # 获取机器人当前在世界坐标系的位姿和局部坐标系的速度
      root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
      
      # 强制将机体局部坐标系的线速度修改为当前的目标指令速度
      lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
      # 将修改后的局部线速度转换回世界坐标系
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      
      # 修改角速度
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
      root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
      
      # 拼接成完整的根节点状态张量 [位置, 姿态, 世界线速度, 局部角速度]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      # 强行写入仿真引擎底层
      self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)

  def _update_command(self) -> None:
    """在每个物理步执行，主要用于更新朝向控制生成的角速度。"""
    if self.cfg.heading_command:
      # 计算当前实际朝向与目标朝向的误差，并规范化到 [-pi, pi]
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      # 获取开启了朝向控制的环境索引
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      # 使用比例控制 (P-control) 将朝向误差转化为角速度指令，并用配置的最大最小角速度进行截断 (clip)
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
      
    # 对于被判定为“站立”的环境，强制将所有速度指令设为 0
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  # ==============================================================================
  # GUI / 遥控器逻辑
  # ==============================================================================
  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """在 Viser Web 界面中创建速度滑块（类似虚拟摇杆）。"""
    from viser import Icon

    ranges = self.cfg.ranges
    # 定义需要创建滑块的 3 个轴向和它们的默认最大值
    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    # 在 GUI 中创建一个折叠面板
    with server.gui.add_folder(name.capitalize()):
      # 总开关复选框
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        # 添加一个用于动态调整该轴“最大范围”的滑块
        max_input = server.gui.add_slider(
          f"Max {label}", initial_value=max_val, step=0.1, min=0.1, max=10.0
        )
        # 添加实际控制指令数值的滑块
        slider = server.gui.add_slider(
          label, min=-max_val, max=max_val, step=0.05, initial_value=0.0
        )

        # 回调函数：当范围滑块改变时，动态更新控制滑块的上限和下限
        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      # 紧急停止/归零按钮
      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # 存储 GUI 句柄，供 compute() 方法读取
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    """覆盖基类的 compute 方法。每一帧都会调用。"""
    super().compute(dt) # 运行原有的逻辑（如重采样和更新指令）
    
    # 如果 GUI 被激活且启用了遥控模式，则覆盖随机生成的指令
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx() # 获取用户正在观看的那个环境 ID
      # 将该环境的指令强制修改为 GUI 滑块上的数值
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  # ==============================================================================
  # 3D 箭头可视化
  # ==============================================================================
  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """在 3D 视图中画出指令速度（深色箭头）和实际速度（亮色箭头）。"""
    # 获取需要进行可视化的环境索引
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    # 将张量数据从 GPU 拉取回 CPU，转换为 NumPy 数组以便画图
    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale # 箭头的长度缩放系数
    z_offset = self.cfg.viz.z_offset # 箭头在机器人身体上方的悬浮高度

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # 跳过那些还没有被正确初始化的机器人（位置完全是原点 0,0,0）
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # 内部辅助函数：将机体坐标系下的局部向量转换为世界坐标系下的点
      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec # 矩阵乘法进行旋转，加上位移

      # 绘制指令线速度箭头（深蓝色）
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # 绘制指令角速度箭头（深绿色）
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # 绘制实际线速度箭头（青色/亮蓝色）
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # 绘制实际角速度箭头（亮绿色）
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )

# ==============================================================================
# 指令类的配置数据结构
# ==============================================================================
@dataclass(kw_only=True) # 使用 kw_only 强制初始化时必须带上参数名 (如 entity_name="robot")
class UniformVelocityCommandCfg(CommandTermCfg):
  entity_name: str # 绑定的机器人在场景中的名称
  heading_command: bool = False # 是否开启基于角度的朝向控制
  heading_control_stiffness: float = 1.0 # 朝向控制 P 控制器的刚度（系数）
  rel_standing_envs: float = 0.0 # 有多少比例的环境保持原地站立
  rel_heading_envs: float = 1.0  # 在开启朝向控制时，有多少比例的环境实际应用朝向控制
  init_velocity_prob: float = 0.0 # 在生成新指令时，直接强行赋予机器人对应初速度的概率

  @dataclass
  class Ranges:
    # 随机生成指令时的数值上下界
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    # 箭头可视化的配置
    z_offset: float = 0.2 # 箭头距离身体的 Z 轴高度
    scale: float = 0.5    # 箭头的整体缩放

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    """工厂方法：被环境管理器调用，实例化真实的业务类。"""
    return UniformVelocityCommand(self, env)

  def __post_init__(self):
    """Dataclass 的钩子函数，在初始化后自动执行，做基础的参数验证。"""
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )