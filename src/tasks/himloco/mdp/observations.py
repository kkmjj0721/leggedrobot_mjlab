from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# 定义一个全局默认常量，指向名为 "robot" 的物理实体资产配置
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """

  """
  # 从环境的场景管理器中，根据配置的名称（如 "robot"）获取实体对象
  asset: Entity = env.scene[asset_cfg.name]
  # 返回世界坐标系下的 Z 轴位置（高度）。
  # site_pos_w 维度假定为 [num_envs, num_sites, 3]，索引 2 代表 Z 轴 (X=0, Y=1, Z=2)
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # 返回形状: (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """
  
  """
  # 从环境中根据传入的名称获取指定的接触传感器对象
  sensor: ContactSensor = env.scene[sensor_name]
  # 获取传感器内部存储的数据对象
  sensor_data = sensor.data
  # 提取脚部当前的滞空时间（距离上次接触地面的时长）
  current_air_time = sensor_data.current_air_time
  # 安全性检查：断言该时间数据不为空，防止后续运算引发异常
  assert current_air_time is not None
  # 返回滞空时间张量
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """
  
  """
  # 获取指定的接触传感器对象
  sensor: ContactSensor = env.scene[sensor_name]
  # 获取传感器内部存储的数据对象
  sensor_data = sensor.data
  # 安全性检查：断言接触检测数据已被初始化
  assert sensor_data.found is not None
  # 如果 found > 0 代表发生了接触，返回布尔张量，并转化为浮点型 (1.0 代表接触，0.0 代表悬空)
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """
  
  """
  # 获取指定的接触传感器对象
  sensor: ContactSensor = env.scene[sensor_name]
  # 获取传感器内部存储的数据对象
  sensor_data = sensor.data
  # 安全性检查：断言受力数据不为空
  assert sensor_data.force is not None
  # 将受力张量从第二个维度开始展平。例如将形状 [batch_size, num_sensors, 3] 展平为 [batch_size, num_sensors * 3]
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  # 对受力数值进行平滑处理：取符号后乘以 log(1 + 绝对值)，这可以有效压缩极端的受力峰值，稳定神经网络训练
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  """
  
  """
  # 计算归一化后的全局相位 (0 到 1 之间的小数)。
  # env.episode_length_buf * env.step_dt 得到当前环境经过的真实时间
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  # 初始化一个形状为 (并行环境数量, 2) 的全 0 张量，用于存放相位的 sin 和 cos 分量
  phase = torch.zeros(env.num_envs, 2, device=env.device)
  # 计算正弦波分量：sin(相位 * 2π)
  phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  # 计算余弦波分量：cos(相位 * 2π)
  phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  # 从指令管理器中获取机器人的运动指令，并计算其 L2 范数（即速度大小）。如果 < 0.1 则视为机器人在要求保持静止站立
  stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
  # 使用条件覆盖：如果当前是静止站立状态，则相位信息全部置为 0；如果处于运动状态，则使用计算出的 sin/cos 相位
  phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
  # 返回最终处理过的相位张量
  return phase
