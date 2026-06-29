from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# 非法接触检测函数
# ---------------------------------------------------------------------------
def illegal_contact(
  env: ManagerBasedRlEnv, # 当前的强化学习环境管理器
  sensor_name: str,       # 需要检查的传感器名称（在配置中定义，例如 "body_contact_sensor"）
  force_threshold: float = 10.0, # 接触力的容忍阈值，默认 10.0 牛顿
) -> torch.Tensor:
  
  # 从环境的场景（scene）中获取指定的接触传感器对象
  sensor: ContactSensor = env.scene[sensor_name]
  
  # 获取该传感器当前的观测数据
  data = sensor.data
  
  # -------------------------------------------------------------------------
  # 分支 1：如果传感器记录了接触力的历史数据
  # -------------------------------------------------------------------------
  if data.force_history is not None:
    # 此时 data.force_history 的维度是 [B, N, H, 3]
    # B: 环境数量, N: 监控的部位数量, H: 历史步数, 3: XYZ方向上的力
    
    # torch.norm 计算最后一个维度（dim=-1，即 XYZ 向量）的 L2 范数（也就是力的模长/绝对大小）
    # 计算后的 force_mag 维度变成了 [B, N, H]
    force_mag = torch.norm(data.force_history, dim=-1)  
    
    # force_mag > force_threshold 会生成一个 [B, N, H] 的布尔张量 (True/False)
    # 第一个 .any(dim=-1) 检查在历史 H 步中，是否至少有一次受力超标，维度变为 [B, N]
    # 第二个 .any(dim=-1) 检查在 N 个监控部位中，是否至少有一个部位受力超标，维度变为 [B]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  
    
  # -------------------------------------------------------------------------
  # 分支 2：如果传感器没有配置记录力的大小，或者不支持输出具体的力（回退逻辑）
  # -------------------------------------------------------------------------
  # 确保 found 属性存在。found 通常是一个布尔张量 [B, N]，仅表示“是否检测到了碰撞”，不包含力度大小
  assert data.found is not None
  
  # 只要这 N 个监控部位中（dim=-1），有任何一个部位发生了碰撞（值为 True），
  # 就返回 True。最终返回一个维度为 [B] 的布尔张量。
  return torch.any(data.found, dim=-1)