from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


class RslRlHimRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Extended runner config with Him-specific parameters."""
    runner_class_name: str = "HIMOnPolicyRunner"
    policy_class_name: str = "HIMActorCritic"
    


def go2_him_ppo_runner_cfg() -> RslRlHimRunnerCfg:
    """Create RL runner configuration for Unitree GO2 Him locomotion task."""
    return RslRlHimRunnerCfg(
        # ================= 1. Actor-Critic 网络配置 =================
        # 策略网络（Actor）：负责输出关节的动作指令
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        # 价值网络（Critic）：评估当前状态的价值
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        # 估计器网络：用于估计机身线速度以及足端状态
        estimator=RslRlModelCfg(
            hidden_dims=(128, 64, 16),
            activation="elu",
            obs_normalization=True,
        ),

        # ================= 2. PPO 算法超参数 =================
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=16,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            class_name="HIMPPO"
        ),

        # ================= 3. 训练流程与日志配置 =================
        experiment_name="go2_him_locomotion", # 实验名称
        logger="tensorboard",                # 使用 TensorBoard 记录训练曲线
        save_interval=1000,                   
        num_steps_per_env=24,                
        max_iterations=10000,               # 最大迭代次数大幅增加至 10万次

    )
      