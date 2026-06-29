from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
    resolve_matching_names_values,
)


if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# 定义一个默认的场景实体配置，默认指向名为 "robot" 的资产
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")



# ==========================================
# 速度跟踪奖励 (Tracking Rewards)
# ==========================================

def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.
    跟踪指令设定的机座线速度奖励。假定指令的 Z 轴（垂直）速度为零。
    """
    asset: Entity = env.scene[asset_cfg.name] # 从场景中获取机器人实体
    command = env.command_manager.get_command(command_name) # 获取当前的期望速度指令
    assert command is not None, f"Command '{command_name}' not found." # 确保指令存在
    actual = asset.data.root_link_lin_vel_b # 获取机器人根节点（机身）在机身坐标系下的实际线速度
    # 计算 X 和 Y 方向上的速度误差平方和（指令速度 - 实际速度）
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    # 计算 Z 方向的速度误差平方（由于期望 Z 速度为 0，所以直接取实际 Z 速度的平方）
    z_error = torch.square(actual[:, 2])
    # 总速度误差，Z 轴误差被放大了 2 倍（强烈惩罚机器人在 Z 轴上的上下跳动）
    lin_vel_error = xy_error + (2 * z_error)
    # 使用高斯核函数（RBF）将误差转化为 [0, 1] 之间的奖励信号
    return torch.exp(-lin_vel_error / std**2)



def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others.
    跟踪角速度的奖励。假定指令的 X、Y（滚转、俯仰）角速度为零，只控制 Z（偏航）角速度。
    """
    asset: Entity = env.scene[asset_cfg.name] # 获取机器人实体
    command = env.command_manager.get_command(command_name) # 获取期望角速度指令
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b # 获取机身在自身坐标系下的实际角速度
    # 计算 Z 轴（偏航转向）的角速度误差平方
    z_error = torch.square(command[:, 2] - actual[:, 2])
    # 计算 X 和 Y 轴上的角速度平方和（期望值为 0，所以实际值的平方即为误差）
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    # 总角速度误差，XY 轴（身体晃动）的误差权重较低，仅为 0.05
    ang_vel_error = z_error + (0.05 * xy_error)
    # 转化为 [0, 1] 区间的奖励
    return torch.exp(-ang_vel_error / std**2)



# ==========================================
# 姿态与运动惩罚项 (Posture & Motion Penalties)
# ==========================================

def body_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward flat base orientation (robot being upright).
    惩罚身体倾斜，鼓励机器人机身保持水平（直立）。
    """
    asset: Entity = env.scene[asset_cfg.name] # 获取机器人实体

    if asset_cfg.body_ids:
        # 如果指定了具体的身体部件 ID，计算该部件的投影重力
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # 获取部件在世界坐标系下的四元数 [Batch, N, 4]
        body_quat_w = body_quat_w.squeeze(1)  # 降维去挤压掉多余维度变为 [Batch, 4]
        gravity_w = asset.data.gravity_vec_w  # 获取世界坐标系下的重力向量 [3]
        # 根据部件的旋转姿态，将重力向量反向旋转，投影到部件自身坐标系下 [Batch, 3]
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  
        # 如果机身完全水平，重力应该完全在 -Z 轴上，X 和 Y 投影应该为 0。计算 X 和 Y 投影的平方和作为惩罚
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        # 如果未指定部件，直接使用根节点（整体机身）的投影重力，计算 XY 轴上的偏差平方和
        xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return xy_squared # 返回惩罚值（值越小说明姿态越正）



def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,               # 用于检测自身碰撞的传感器名称
    force_threshold: float = 10.0,  # 判定为碰撞的接触力阈值
) -> torch.Tensor:
    """Penalize self-collisions.
    自碰撞惩罚。如果机器人自己的腿打结或者部件互相碰撞，给予惩罚。
    """
    sensor: ContactSensor = env.scene[sensor_name] # 获取接触传感器
    data = sensor.data # 获取传感器数据
    if data.force_history is not None:
        # 如果存在力学历史记录: [Batch, Entity_N, History_Length, 3 (xyz)]
        force_mag = torch.norm(data.force_history, dim=-1)  # 计算接触力的模长 [Batch, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # 只要在这段时间内有任何部位受力超过阈值，计为碰撞（hit） [Batch, H]
        return hit.sum(dim=-1).float()  # 累加时间步内的碰撞次数作为惩罚值 [Batch]
    assert data.found is not None # 确保基本的接触发现标识存在
    return data.found.squeeze(-1) # 如果没有历史记录，直接返回瞬时的接触发现结果作为惩罚



def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive body angular velocities.
    惩罚过大的机身角速度（防止机器人走路时身体过度摇晃）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    # 获取指定部件在世界坐标系下的角速度
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
    ang_vel = ang_vel.squeeze(1) # 降维
    ang_vel_xy = ang_vel[:, :2]  # 只提取 X 和 Y（滚转和俯仰）方向的角速度。不惩罚 Z（偏航转向），因为转向是正常的。
    return torch.sum(torch.square(ang_vel_xy), dim=1) # 返回 XY 角速度的平方和作为惩罚



def angular_momentum_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize whole-body angular momentum to encourage natural arm swing.
    惩罚全身的角动量，以鼓励自然的摆臂行为（双足机器人的摆臂可以抵消腿部产生的角动量）。
    """
    angmom_sensor: BuiltinSensor = env.scene[sensor_name] # 获取角动量传感器
    angmom = angmom_sensor.data # 获取角动量数据 (xyz)
    angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1) # 计算角动量大小的平方
    angmom_magnitude = torch.sqrt(angmom_magnitude_sq) # 计算角动量大小
    # 将角动量的平均值记录到环境日志中，用于可视化和监控
    env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
    return angmom_magnitude_sq # 返回角动量的平方作为惩罚



# ==========================================
# 步态与足端控制 (Feet & Gait Mechanics)
# ==========================================

def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 0.4,             # 目标腾空时间（秒）
    command_name: str | None = None,
    command_threshold: float = 0.1,     # 指令阈值，速度大于此值才激活该奖励
) -> torch.Tensor:
    """Reward feet air time.
    奖励足端腾空时间。鼓励机器人把脚抬起来走，而不是在地上拖着走。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    air_time = sensor_data.current_air_time         # 获取当前处于空中的时间
    contact_time = sensor_data.current_contact_time # 获取当前接触地面的时间
    in_contact = contact_time > 0.0                 # 布尔值：当前是否接触地面
    # 如果触地，计算当前阶段时间为接触时间，否则为空中时间
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    # 判断是否为单支撑相（比如两条腿中只有一条接触地面，均值 = 0.5）
    single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
    # 获取最小的阶段时间作为基准
    mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # 计算实际阶段时间与目标时间的误差
    error = torch.abs(mode_time - threshold)
    # 如果误差小于阈值，给予正奖励；否则为 0
    reward = torch.clamp(threshold - error, min=0.0)
    
    # 只有当机器人收到移动指令时，才计算并赋予此奖励
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1) # 线速度大小
            angular_norm = torch.abs(command[:, 2])         # 角速度大小
            total_command = linear_norm + angular_norm      # 总指令强度
            scale = (total_command > command_threshold).float() # 判断是否在移动
            reward *= scale # 乘以缩放系数（不动时奖励强制为0）
    return reward



def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,           # 目标抬脚离地高度
    command_name: str | None = None,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity.
    脚部抬起高度惩罚项。如果脚部运动时高度偏离目标高度，则给予惩罚（用脚部速度加权）。
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # 获取脚部在世界坐标系下的 Z 轴高度 [Batch, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # 获取脚部的 XY 面水平速度 [Batch, N, 2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # 计算水平速度大小 [Batch, N]
    delta = torch.abs(foot_z - target_height)  # 计算实际高度与目标高度的绝对偏差 [Batch, N]
    # 计算惩罚值：高度偏差乘以速度。这意味着如果脚运动得快，它就更应该处于目标高度。
    cost = torch.sum(delta * vel_norm, dim=1)  # [Batch]
    
    # 仅在机器人处于移动指令时激活惩罚
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost



def feet_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact).
    滑步惩罚。惩罚脚在接触地面时，仍然有水平（XY）方向的滑动速度。
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float() # 是否在移动
    
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # 筛选当前踩在地面上的脚 [Batch, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # 脚的水平速度 [Batch, N, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # 计算水平速度大小 [Batch, N]
    vel_xy_norm_sq = torch.square(vel_xy_norm)  # 速度平方 [Batch, N]
    
    # 只有触地的脚并且机器人在移动时才计算滑步惩罚
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
    
    # 记录日志
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
        num_in_contact, min=1
    )
    env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
    return cost


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls.
    软着陆惩罚。惩罚脚落地瞬间产生的巨大冲击力，鼓励机器人“轻拿轻放”，保护硬件。
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # 获取脚部受到的接触力 [Batch, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)  # 计算接触力的大小 [Batch, N]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # 找到刚刚落地的脚 [Batch, N]
    
    # 只提取落地那一瞬间的力
    landing_impact = force_magnitude * first_contact.float()  # [Batch, N]
    cost = torch.sum(landing_impact, dim=1)  # 所有脚的冲击力累加作为惩罚 [Batch]
    
    # 记录日志
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
    
    # 只在指令激活时生效
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


class variable_posture:
    """Penalize deviation from default pose with speed-dependent tolerance.
    基于速度变化容忍度的姿态偏差惩罚。
    机器人站立时，我们希望它严格保持标准姿势；但奔跑时，由于动作幅度大，必须放宽对“偏离标准姿势”的惩罚。
    使用每个关节的标准差(std)来控制各关节的容忍度。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos # 机器人的默认（标准）关节角度
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        # 解析并加载站立(standing)模式下各个关节的容忍度 std
        _, _, std_standing = resolve_matching_names_values(
            data=cfg.params["std_standing"],
            list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(
            std_standing, device=env.device, dtype=torch.float32
        )

        # 加载行走(walking)模式下各个关节的容忍度 std
        _, _, std_walking = resolve_matching_names_values(
            data=cfg.params["std_walking"],
            list_of_strings=joint_names,
        )
        self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

        # 加载奔跑(running)模式下各个关节的容忍度 std
        _, _, std_running = resolve_matching_names_values(
            data=cfg.params["std_running"],
            list_of_strings=joint_names,
        )
        self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.5,     # 判定为步行的速度下限
        running_threshold: float = 1.5,     # 判定为奔跑的速度下限
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # 占位符，未使用。直接使用 __init__ 预先计算好的值。

        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None

        # 计算总速度指令
        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        # 判断当前所处的运动状态（掩码）
        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
            (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        # 根据运动状态，插值（融合）出当前的宽容度 std
        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        # 当前姿态与默认姿态的角度误差平方
        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        # 使用高斯核函数转化为奖励信号： 误差越大/std越小，则 exp 趋近于 0（不给奖励）；误差越小，exp 趋向 1。
        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """原地站立惩罚。
    当机器人接到“保持静止”的指令时，任何关节偏离其默认姿态的行为都会受到惩罚。
    """
    asset: Entity = env.scene[asset_cfg.name]
    # 计算当前所有关节与其默认姿势的角度偏差
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # 误差平方和
    reward = torch.sum(torch.square(diff_angle), dim=1)
    
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            # 只有在速度指令小于阈值（也就是在原点站立时）才会施加该惩罚项
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward



def hip_pos(
        env: ManagerBasedRlEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        command_threshold: float = 0.01
) -> torch.Tensor:
    """
    惩罚在没有侧向(vy)和转向(wz)指令时，hip关节偏离默认中点位置的行为。
    防止机器人在直线行走或站立时出现外八字或劈叉现象。
    """
    robot = env.scene[asset_cfg.name]
    joint_indices = asset_cfg.joint_ids
    
    # 如果没有匹配到对应的关节，返回 0 惩罚
    if joint_indices is None or len(joint_indices) == 0:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        
    # 1. 获取当前的速度指令。
    # 通常 twist 指令的形状是 [num_envs, 3]，对应 [vx(前进), vy(侧向), wz(偏航转向)]
    commands = env.command_manager.get_command(command_name)
    vy_cmd = commands[:, 1]
    wz_cmd = commands[:, 2]
    
    # 2. 判断是否“没有侧向指令”且“没有转向指令”
    # 在强化学习中，直接 == 0.0 可能因为噪声或浮点数问题判断失败，使用一个小阈值更稳妥。
    no_lateral = torch.abs(vy_cmd) <= command_threshold
    no_yaw = torch.abs(wz_cmd) <= command_threshold
    
    # 将两个条件求逻辑“与”，并转为浮点型张量 (True -> 1.0, False -> 0.0)
    flag = torch.logical_and(no_lateral, no_yaw).float()
    
    # 3. 计算 hip 关节的偏移量
    current_pos = robot.data.joint_pos[:, joint_indices]
    default_pos = robot.data.default_joint_pos[:, joint_indices]
    
    # 计算均方误差平方和：SUM( (pos - default)^2 )
    deviation = torch.sum(torch.square(current_pos - default_pos), dim=1)
    
    # 4. 用 flag 掩码乘以偏移量。只有满足无指令条件的环境才会产生大于 0 的返回值。
    return flag * deviation
