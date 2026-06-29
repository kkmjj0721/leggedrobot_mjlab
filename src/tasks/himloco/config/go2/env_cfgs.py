from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.himloco.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
    play: bool = False, # play 模式通常指用于评估/推理展示的模式（而非训练模式）
) -> ManagerBasedRlEnvCfg:
    """创建 Unitree Go2 复杂地形速度控制配置。"""

    cfg = make_velocity_env_cfg() # 实例化基础的速度环境配置

    # --- 物理引擎核心参数调整 ---
    cfg.sim.mujoco.ccd_iterations = 500 # 连续碰撞检测(CCD)的迭代次数，设为 500 提高碰撞精度
    cfg.sim.contact_sensor_maxmatch = 500 # 接触传感器的最大匹配数，防止复杂地形下接触点丢失

    # --- 场景与实体配置 ---
    cfg.scene.entities = {"robot": get_go2_robot_cfg()} # 将 Go2 机器人载入场景

    # --- 射线投射(雷达)传感器配置 ---
    # 将地形扫描传感器的坐标系绑定到 Go2 的机身基座 (base_link) 上
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "base_link"

    # --- 足端与刚体名称定义 ---
    foot_names = ("FL", "FR", "RL", "RR") # 四条腿的缩写
    site_names = ("FL", "FR", "RL", "RR") # 用于足端相关计算的 site（附着点）名称
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names) # 足端碰撞体(geom)的名称生成

    # --- 接触传感器配置 ---
    # 1. 正常足端触地传感器
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        # 主要检测体：机器人的足端碰撞体
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        # 次要检测体：地形
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), # 记录的数据字段：是否接触、接触力大小
        reduce="netforce", # 数据归约方式：计算净力
        num_slots=1,
        track_air_time=True, # 追踪腾空时间（用于计算步态和奖励）
    )

    # 2. 非足端触地传感器（用于检测摔倒）
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
        mode="geom",
        entity="robot",
        pattern=r".*_collision\d*$", # 正则匹配：抓取所有碰撞体...
        exclude=tuple(geom_names),    # 排除掉足端的碰撞体（即检测身体、大腿等是否触地）
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"), # 接触对象：地形
        fields=("found", "force"),
        reduce="none", # 不做归约，保留所有接触点信息
        num_slots=1,
        history_length=4, # 记录过去 4 步的历史数据
    )

    # 将上述两个接触传感器添加到场景中
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        nonfoot_ground_cfg,
    )

    # --- 地形课程学习配置 ---
    # 如果存在地形生成器，开启课程学习（Curriculum），让机器人从简单地形逐渐过渡到复杂地形
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    # --- 动作空间配置 ---
    joint_pos_action = cfg.actions["joint_pos"] # 获取关节位置动作配置
    assert isinstance(joint_pos_action, JointPositionActionCfg) # 确保其类型正确

    # --- 渲染器(Viewer)配置 ---
    cfg.viewer.body_name = "base_link" # 视角锁定在机身
    cfg.viewer.distance = 1.5          # 相机距离
    cfg.viewer.elevation = -10.0       # 相机俯仰角

    # --- 奖励函数(Rewards)参数配置 ---
    # 姿态奖励：定义不同运动状态下各关节位置的标准差（容忍度）
    cfg.rewards["pose"].params["std_standing"] = { # 站立状态
        r".*(FL|FR|RL|RR)_hip_joint.*": 0.05,   # 髋关节允许极小偏差
        r".*(FL|FR|RL|RR)_thigh_joint.*": 0.1,  # 大腿关节允许较小偏差
        r".*(FL|FR|RL|RR)_calf_joint.*": 0.15,  # 小腿关节允许适中偏差
    }
    cfg.rewards["pose"].params["std_walking"] = {  # 行走状态（偏差容忍度增加）
        r".*(FL|FR|RL|RR)_hip_joint.*": 0.15,
        r".*(FL|FR|RL|RR)_thigh_joint.*": 0.35,
        r".*(FL|FR|RL|RR)_calf_joint.*": 0.5,
    }
    cfg.rewards["pose"].params["std_running"] = {  # 奔跑状态（与行走设置一致）
        r".*(FL|FR|RL|RR)_hip_joint.*": 0.15,
        r".*(FL|FR|RL|RR)_thigh_joint.*": 0.35,
        r".*(FL|FR|RL|RR)_calf_joint.*": 0.5,
    }    

    # 其他惩罚/奖励指标的依附刚体设定
    cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",) # 躯干姿态惩罚
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)        # 躯干角速度惩罚
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names          # 抬腿高度奖励
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names               # 足端滑移惩罚

    # --- 终止条件(Terminations)配置 ---
    # 非法接触：如果非足端部位接触地面且受力大于 10.0，则判定为摔倒，结束回合
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
    )

    # 当用于测试模型而不是训练时，覆盖部分参数
    if play:
        cfg.episode_length_s = int(1e9) # 将回合长度设为极大值（无限长）

        cfg.observations["actor"].enable_corruption = False # 关闭观测噪声（不增加随机扰动）
        cfg.events.pop("push_robot", None) # 移除随机推挤机器人的事件
        cfg.curriculum = {} # 禁用课程学习
        # 强制在重置时随机化地形
        cfg.events["randomize_terrain"] = EventTermCfg(
        func=envs_mdp.randomize_terrain,
        mode="reset",
        params={},
        )

        # 缩小地形生成面积并移除地形课程机制，适合展示
        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg



def unitree_go2_flat_env_cfg(
    play: bool = False
) -> ManagerBasedRlEnvCfg:
    """创建 Unitree Go2 平坦地形速度控制配置。"""
    # 继承上面定义的复杂地形配置
    cfg = unitree_go2_rough_env_cfg(play=play)

    # --- 物理引擎降级与优化 ---
    # 因为是平地，不需要那么复杂的物理计算，以此加快仿真速度
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50 # 减少连续碰撞检测迭代次数
    cfg.sim.contact_sensor_maxmatch = 64 # 减少接触传感器匹配数
    cfg.sim.nconmax = None

    # --- 切换地形类型 ---
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane" # 改为绝对平面的地形
    cfg.scene.terrain.terrain_generator = None # 移除地形生成器

    # --- 移除不必要的传感器与观测值 ---
    # 因为是平地，不需要地形雷达扫描数据
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )

    # 从 Actor 和 Critic 网络的观测值中删除地形高度扫描数据
    del cfg.observations["critic"].terms["height_scan"]

    # 移除地形课程学习等级配置
    cfg.curriculum.pop("terrain_levels", None)

    # --- 推理模式速度限制 ---
    if play:
        twist_cmd = cfg.commands["twist"] # 获取速度指令配置
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        # 限定展示时的速度生成范围：
        twist_cmd.ranges.lin_vel_x = (-1.0, 1.0) # 前进后退速度 (-0.5 到 1.0 m/s)
        twist_cmd.ranges.lin_vel_y = (-1.0, 1.0) # 横移速度 (-0.5 到 0.5 m/s)
        twist_cmd.ranges.ang_vel_z = (-3.14, 3.14) # 转向角速度 (-0.5 到 0.5 rad/s)

    return cfg