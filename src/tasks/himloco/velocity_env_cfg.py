import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
import src.tasks.himloco.mdp as mdp


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base velocity tracking task configuration."""
    # 定义工厂函数，返回一个完整的基于管理器的 RL 环境配置对象。

    ##
    # Sensors (传感器配置)
    ##

    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        # 定义一个射线投射传感器（模拟激光雷达或深度相机扫描地形）。
        frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
        # 绑定的坐标系留空，由具体机器人去设定（通常绑定在机器人基座 base_link）。
        ray_alignment="yaw",
        # 射线对齐方式：跟随机器人的偏航角(yaw)旋转，但不随俯仰和横滚变化（保持水平面相对固定）。
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        # 扫描图案：1.6米长，1.0米宽的网格，分辨率0.1米（即 16x10 = 160 根射线）。
        max_distance=5.0,
        # 射线最大探测距离为 5.0 米。
        exclude_parent_body=True,
        # 扫描时忽略机器人自身的身体，防止射线打到自己。
        debug_vis=True,
        # 开启调试可视化。
        viz=RayCastSensorCfg.VizCfg(show_normals=True),
        # 可视化配置：显示击中点处的法线。
    )

    ##
    # Observations (观测空间配置：神经网络的输入)
    ##
    # Actor 是策略网络（Policy），控制机器人运动，只能看到有噪声的、可实际获取的传感器数据。
    actor_terms = {
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
            # 当前要求机器人达到的目标速度指令（vx, vy, yaw_rate）。
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            # 获取基座的角速度（来自 IMU），并注入均匀分布的噪声（[-0.2, 0.2]），模拟真实世界传感器误差。
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            # 投影重力向量（隐式包含机器人的 roll 和 pitch 姿态），注入噪声。
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            # 相对关节位置（编码器数据），注入噪声。
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            # 关节速度，通常现实中的速度信号非常嘈杂，因此注入了较大的噪声 ([-1.5, 1.5])。
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
            # 上一帧网络输出的动作（用于让网络自己学习平滑输出，避免动作突变）。
    }

    # Critic 是价值网络（Value Network），仅在训练时使用，用于评估状态好坏。
    # 它可以获取“特权信息”（Privileged Information），即真实世界中无法直接获取的精确物理状态。
    critic_terms = {
        **actor_terms,
        # Critic 包含所有 Actor 的观测数据

        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            # 机器人的真实线速度（现实中仅靠 IMU 很难精准计算，通常需要复杂状态估计，但仿真中直接提取）。
        ),

        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            scale=1 / terrain_scan.max_distance,
            # Critic 获取的地形扫描数据（没有额外添加观测噪声，获取真实状态）。
        ),

        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
            # 真实足端高度特权信息。
        ),

        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
            # 足端滞空时间（判断步态使用）。
        ),

        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
            # 足端是否接触地面的布尔/离散信息。
        ),

        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
            # 足端精确接触力大小。
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True, # 拼接成一维向量输入网络
            enable_corruption=True, # 开启噪声污染
            history_length=6,       # 不保存历史（帧堆叠数量设为 1，即仅当前帧）
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False, # Critic 看到的是纯净世界，不开启噪声污染（注意上面个别项自带了 Unoise，但这里会被全局覆盖或只应用特定策略）
            history_length=1,
        ),
    }

    ##
    # Metrics (指标记录：仅用于监控训练过程，不参与网络反向传播)
    ##
    metrics = {
        "mean_action_acc": MetricsTermCfg(
            func=mdp.mean_action_acc,
            # 记录平均动作加速度（用于观察策略输出的平滑度）。
        ),
    }

    ##
    # Actions (动作空间配置：神经网络的输出)
    ##
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",), # 正则表达式匹配所有关节执行器
            scale=0.25,             # 动作缩放比例（网络输出范围有限，放大 0.25 倍转为物理命令）
            use_default_offset=True,# 以机器人的默认站立姿势作为 0 偏置基准
        )
    }

    ##
    # Commands (指令空间：给网络下达的任务目标)
    ##
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0), # 每隔 3 到 8 秒随机重新生成一个新指令
            rel_standing_envs=0.05,           # 5% 的环境被指令为原地站立（不给速度）
            heading_command=True,             # 开启航向角控制
            heading_control_stiffness=0.5,    # 航向控制的刚度
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 2.0),        # 前后线速度范围：-1m/s(后退) 到 2m/s(前进)
                lin_vel_y=(-1.0, 1.0),        # 左右侧滑速度范围
                ang_vel_z=(-1.0, 1.0),        # 原地转向角速度范围
                heading=(-math.pi, math.pi),  # 目标朝向角范围 (-180° 到 180°)
            ),
        )
    }

    ##
    # Events (事件配置：主要用于重置环境与域随机化 Domain Randomization)
    ##
    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset", # 在环境 Reset 时触发
            params={
                "pose_range": { # 在一定范围内随机初始化机器人的出生位姿和朝向
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {}, # 初始速度为0
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0), # 关节角度严格重置为默认值
                "velocity_range": (-0.0, 0.0), # 关节初始速度为0
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",               # 按固定的时间间隔触发
            interval_range_s=(5.0, 6.0),   # 每 5 到 6 秒踹机器人一脚
            params={
                "velocity_range": {        # 通过瞬间改变机器人的线速度和角速度来模拟“被踢/被推”
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup", # 在仿真刚启动时触发（初始化属性）
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "ranges": (0.3, 1.6), # 将地面的摩擦系数随机设定在 0.3 到 1.6 之间，提高策略的跨地形泛化性
                "shared_random": True, # 所有脚部几何体共享相同的随机摩擦力
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015), # 随机给关节编码器加上固定偏移量，模拟传感器安装误差
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
                "operation": "add",
                "ranges": { # 随机偏移机器人的质心 (CoM) 位置，提高抗背负重物能力
                    0: (-0.05, 0.05), # x轴方向质心偏移
                    1: (-0.05, 0.05), # y轴方向质心偏移
                    2: (-0.05, 0.05), # z轴方向质心偏移
                },
            },
        ),
    }

    ##
    # Rewards (奖励函数：强化学习的灵魂，引导机器人学习期望的动作)
    ##
    rewards = {
        # 1. 任务达成奖励 (正值)
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=1.0, # 追踪目标线速度的奖励
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=1.0, # 追踪目标角速度的奖励
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),

        # 2. 姿态与能量惩罚 (负值，引导自然、省力、安全的运动)
        "body_orientation_l2": RewardTermCfg(
            func=mdp.body_orientation_l2,
            weight=-1.0, # 惩罚躯干不水平（保持背部平直）
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=1.0, # 维持特定姿态的奖励
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "command_name": "twist",
                "std_standing": {}, "std_walking": {}, "std_running": {}, # 根据不同速度阈值配置不同容忍度
                "walking_threshold": 0.1,
                "running_threshold": 1.5,
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.05, # 惩罚躯干额外的晃动（俯仰和横滚角速度）
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
        ),
        "angular_momentum": RewardTermCfg(
            func=mdp.angular_momentum_penalty,
            weight=-0.025, # 惩罚整体角动量过大（防止挥舞四肢）
            params={"sensor_name": "robot/root_angmom"},
        ),
        "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
            # 致命惩罚：如果因为摔倒导致回合提前终止，给予巨额扣分。
        "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
            # 惩罚关节加速度过大，保护真实机器人的电机。
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
            # 惩罚关节角度接近物理极限。
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
            # 惩罚相邻帧的输出动作差异过大，促使策略网络输出平滑曲线。

        # 3. 步态与足端规范 (规范脚部动作)
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=-1.0, # 惩罚足端挥舞高度没有达到目标高度 (0.10米)
            params={
                "target_height": 0.10, "command_name": "twist", "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.25, # 惩罚触地时的滑动（脚踩地上时脚底线速度不为0）
            params={
                "sensor_name": "feet_ground_contact", "command_name": "twist",
                "command_threshold": 0.1, "asset_cfg": SceneEntityCfg("robot", site_names=()),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-3, # 惩罚落地时冲击力过大（要求轻柔落地）
            params={
                "sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.1,
            },
        ),
        "stand_still": RewardTermCfg(
            func=mdp.stand_still,
            weight=-1.0, # 当速度指令极小(<0.1)时，惩罚关节微动，要求像雕塑一样站稳。
            params={
                "command_name": "twist", "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "hip_pos": RewardTermCfg(
            func=mdp.hip_pos,
            weight=-0.5,
            params={
                "command_name": "twist", # 匹配你 command manager 里的命令名称
                "command_threshold": 0.05, # 如果指令绝对值小于 0.05 m/s 或 rad/s，则视为 0
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_.*")),
            },
        ),
    }

    ##
    # Terminations (终止条件：判断什么时候结束当前回合)
    ##
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        # 超时终止：当前回合达到了设定的最大时长。不作为致命惩罚。
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
            # 摔倒终止：机器人的横滚角(Roll)或俯仰角(Pitch)超过了 70 度，判定为翻车。
        ),
    }

    ##
    # Curriculum (课程学习：动态调整训练难度)
    ##
    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
            # 地形课程：如果机器人走得好，将它移到更崎岖的地形；走得不好则降级到平坦地形。
        ),
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    # 速度课程：阶段 0，命令速度在一个较小且容易的范围内
                    {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
                    # 速度课程：当训练步数达到 5000*24 时，解锁更大的速度上限，让它跑得更快
                    {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
                ],
            },
        ),
    }

    ##
    # Assemble and return (打包组装并返回)
    ##
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            # scene (场景): 定义物理世界的整体面貌和包含的实体
            terrain=TerrainEntityCfg(
                # terrain (地形): 配置地面环境
                terrain_type="generator",
                # 地形类型设为“生成器”模式（代码会在运行时动态生成复杂地形，而不是加载一个静态的 3D 模型文件）
                terrain_generator=replace(ROUGH_TERRAINS_CFG),
                # 复制并使用预设的“崎岖地形”生成规则（ROUGH_TERRAINS_CFG 通常包含了怎么生成阶梯、斜坡、坑洼的算法参数）
                max_init_terrain_level=5, 
                # 初始化最大地形等级为 5：当机器人在每个回合刚出生时，最多被放置在难度级别为 5 的地形上。
            ),
            sensors=(terrain_scan,),
            # 将之前定义好的 terrain_scan (激光雷达/射线投射扫描器) 加入场景的传感器列表中。
            num_envs=2048,    
            # 默认的并行环境数量设为 1。
            # 【注】在实际进行 GPU 大规模 RL 训练时，外部的启动脚本通常会覆盖这个参数（例如改写为 4096），从而生成四千条狗同时在不同的地形格子里训练。
            extent=2.0,
            # 定义单个环境边界的范围（边长）为 2.0 米。在并行训练时，这决定了每个环境“格子”之间的间距，防止机器人跑串门。
        ),
        
        # --- 将之前定义好的各个管理器配置直接挂载进来 ---
        observations=observations, # 挂载观测空间（神经网络的输入，如关节位置、速度、地形扫描）
        actions=actions,           # 挂载动作空间（神经网络的输出，如目标关节角度）
        commands=commands,         # 挂载指令系统（给机器人下发的期望移动速度）
        events=events,             # 挂载事件系统（包含环境重置逻辑、以及用于增加鲁棒性的域随机化：如随机踹一脚、随机摩擦力）
        rewards=rewards,           # 挂载奖励函数系统（引导机器人学会走路的加分和扣分规则）
        terminations=terminations, # 挂载终止条件（什么情况下判定回合失败，如倾角过大摔倒）
        curriculum=curriculum,     # 挂载课程学习机制（动态调整地形难度和速度要求）
        metrics=metrics,           # 挂载评估指标（仅用于数据记录和 TensorBoard 曲线绘制，不参与梯度回传）
        
        viewer=ViewerConfig( 
            # viewer (可视化器): 配置图形界面的相机渲染视角
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            # 相机的追踪原点类型：绑定到环境中的某个资产（Asset）的身体上。
            entity_name="robot",
            # 指定追踪的实体名称为 "robot"（即我们的机械狗）。
            body_name="base_link",  
            # 具体绑定的身体部件名称，这里留空。通常会在具体的机器人配置中覆盖为 "base_link" (躯干质心)。
            distance=3.0,
            # 相机距离机器人中心的直线距离设为 3.0 米。
            elevation=-5.0,
            # 相机的俯仰角设为 -5 度（即微微向下俯视机器人）。
            azimuth=90.0,
            # 相机的方位角设为 90 度（通常代表从正侧面或特定侧面观察）。
        ),
        
        sim=SimulationCfg(   
            # sim (仿真底层): 物理引擎的底层计算配置
            nconmax=35,      
            # 最大接触约束数 (Max Contacts)。限制底层引擎处理接触点的数量上限，设为 35 可以在保证四足机器人接触计算准确的前提下节省显存/内存。
            njmax=1500,      
            # 最大关节/标量约束数 (Max Joints)。预留给各种物理约束的内存池大小。
            mujoco=MujocoCfg(
                # MuJoCo 物理引擎专有的高级参数设定
                timestep=0.005,  
                # 物理仿真的最小时间步长为 0.005秒。这意味着物理引擎以 200Hz 的高频运行计算（1 / 0.005 = 200），保证碰撞和受力精准。
                iterations=10,   
                # 接触点约束求解器的最大迭代次数。10 次是一个在计算速度和仿真精度之间折中的经验值。
                ls_iterations=20,
                # 非线性优化中线搜索 (Line Search) 的迭代次数。用于处理复杂接触时的收敛。
            ),
        ),

        decimation=4,        # 控制降采样率：策略网络的频率 = 物理频率(200Hz) / decimation(4) = 50Hz。网络每秒输出 50 次动作。
        episode_length_s=20.0, # 每一个 Episode (回合) 的最大物理时长：20秒。
    )
