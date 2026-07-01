from mjlab.tasks.registry import register_mjlab_task
from src.tasks.himloco.rl import HIMLocoOnPolicyRunner


from .env_cfgs import (
  unitree_go2_rough_env_cfg,
)

from .rl_cfg import go2_him_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-Go2-Rough",                                                  # 任务名
  env_cfg=unitree_go2_rough_env_cfg(),                                          # 任务环境配置
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),                            # 评估环境配置
  rl_cfg=go2_him_ppo_runner_cfg(),                                              # 算法配置
  runner_cls=HIMLocoOnPolicyRunner,                                             # 算法运行入口
)




