from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg


##
# MJCF and assets. (MJCF 模型与资产加载)
##

# 定义 Go2 的 MuJoCo XML (MJCF) 描述文件的绝对路径
GO2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """解析 XML 文件并生成 MuJoCo 的 MjSpec (模型规范) 对象。"""
  spec = mujoco.MjSpec.from_file(str(GO2_XML)) # 让 MuJoCo 从 XML 文件生成规范对象。
  return spec # 返回完整的模型规范。


##
# Actuator config. (执行器/电机配置)
##

GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
  delay_min_lag=2,
  delay_max_lag=4,
  delay_hold_prob=0.3,         # 30% chance to keep current lag
  delay_update_period=10,
)

GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
  delay_min_lag=2,
  delay_max_lag=4,
  delay_hold_prob=0.3,         # 30% chance to keep current lag
  delay_update_period=10,
)

GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
  delay_min_lag=2,
  delay_max_lag=4,
  delay_hold_prob=0.3,         # 30% chance to keep current lag
  delay_update_period=10,
)


##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),                       # base初始坐标
  joint_pos={                                 # 关节初始位置
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},                      # 关节初始速度
)


##
# Collision config. (碰撞配置)
##

# 正则表达式：匹配四个脚尖的碰撞体 (FR_foot_collision, FL_foot_collision, RR_foot_collision, RL_foot_collision)
_foot_regex = "^[FR][LR]_foot_collision$"


# 配置1：仅脚掌碰撞模式 (禁用自身碰撞，仅允许脚与地面等外部物体碰撞)
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,), # 仅针对脚部碰撞体生效
  contype=0, # 碰撞类型掩码设为0（用于MuJoCo的碰撞过滤机制）
  conaffinity=1, # 碰撞亲和力掩码设为1
  condim=3, # 接触维度为3（包含法向力和两个方向的摩擦力）
  priority=1, # 碰撞优先级
  friction=(0.6,), # 摩擦系数设为 0.6
  solimp=(0.9, 0.95, 0.023), # 求解器阻抗参数 (决定接触的软硬度/弹性)
)


# 配置2：全碰撞模式 (开启所有部件的碰撞，但赋予脚掌特殊的物理属性)
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",), # 匹配所有带 _collision 后缀的几何体
  condim={_foot_regex: 3, ".*_collision": 1}, # 脚的接触维度为3(计算摩擦)，其他身体部件接触维度为1(仅计算法向碰撞，不计算摩擦省算力)
  priority={_foot_regex: 1}, # 脚掌碰撞具有高优先级
  friction={_foot_regex: (0.6,)}, # 为脚掌分配 0.6 的摩擦系数
  solimp={_foot_regex: (0.9, 0.95, 0.023)}, # 脚掌接触的阻抗模型参数
  contype=1, # 全局碰撞类型设为1
  conaffinity=0, # 全局碰撞亲和力设为0（如果两个物体的 type AND affinity 为 0，通常不会相互碰撞，这里通常用来防止机器人自我穿模）
)


##
# Final config. (最终整体配置)
##

# 将之前定义的三个电机组别打包成关节运动学配置
GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9, # 软限位系数：限制在关节物理行程的 90% 内运动，防止真实机器人撞击机械限位。
)


def get_go2_robot_cfg() -> EntityCfg:
  """获取一个新的 Go2 机器人配置实例。

  每次调用返回一个全新的 EntityCfg 对象，防止在多处实例化机器人时共享同一个配置
  导致参数被意外修改（变异问题）。
  """
  return EntityCfg(
    init_state=INIT_STATE, # 注入初始状态
    collisions=(FULL_COLLISION,), # 注入碰撞规则（此处默认使用了全碰撞）
    spec_fn=get_spec, # 注入获取模型规范的函数
    articulation=GO2_ARTICULATION, # 注入关节电机配置
  )


# 主程序入口：如果直接运行此脚本，则启动可视化界面
if __name__ == "__main__":
  import mujoco.viewer as viewer # 导入 MuJoCo 官方提供的交互式查看器

  from mjlab.entity.entity import Entity # 导入 mjlab 的实体基类

  # 1. 使用刚才写好的配置工厂函数生成实体实例
  robot = Entity(get_go2_robot_cfg())

  # 2. 将机器人的规范 (spec) 编译为 MuJoCo 底层的 Model，并启动 Viewer 窗口查看
  viewer.launch(robot.spec.compile())