import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
# train_hyper.py

import os
import sys
import logging
import time
from functools import partial

# 导入 stable-baselines3 的核心组件
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# MODIFIED: 导入 SAC 而不是 PPO
from stable_baselines3 import SAC
import torch.nn as nn

# 导入我们自己的模块
from src.env.hyper_config import HyperConfig
from src.env.hyper_env import GraspHyperEnv

def setup_logger() -> logging.Logger:
    """配置一个简单的日志记录器用于主脚本"""
    logger = logging.getLogger("TrainHyper")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # 确保日志不会重复添加
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s', 
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


from typing import Callable
def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """
    线性学习率衰减
    :param progress_remaining: 1.0 (开始) -> 0.0 (结束)
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining 从 1 降到 0
        return final_value + (initial_value - final_value) * progress_remaining
    return func

class Tee:
    """
    Simple tee for duplicating stdout/stderr to a file and original stream.
    """
    def __init__(self, file_obj, stream):
        self.file = file_obj
        self.stream = stream

    def write(self, data):
        # 写入原始终端 (Console)
        try:
            self.stream.write(data)
            # 关键：对于进度条的 \r (回车)，必须强制 flush 才能看到动态效果
            self.stream.flush() 
        except Exception:
            pass
        
        # 写入日志文件 (File)
        # 注意：为了防止进度条的 \r 把日志文件写乱，可以加个判断
        # 如果 data 纯粹是回车符或包含大量控制符，可以选择不写文件
        try:
            self.file.write(data)
            self.file.flush() # 确保日志实时落盘
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass

    # --- [关键修复] ---
    def isatty(self):
        """
        伪装成终端。
        tqdm 会调用此方法来判断是否显示进度条。
        我们直接返回原始流 (sys.__stdout__) 的状态。
        """
        try:
            return self.stream.isatty()
        except Exception:
            return False

def main():
    """主训练函数"""
    # 1. 加载配置并创建输出目录
    config = HyperConfig()

    os.makedirs(config.SAVE_PATH, exist_ok=True)
    os.makedirs(config.TENSORBOARD_LOG, exist_ok=True)

    # 打开日志文件并把 stdout/stderr 同时写到文件和终端
    log_file = None
    log_fname = None
    try:
        log_fname = os.path.join(
            config.SAVE_PATH, f"train_hyper_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        # line-buffered (buffering=1) 并以 utf-8 写入
        log_file = open(log_fname, "a", buffering=1, encoding='utf-8')
        sys.stdout = Tee(log_file, sys.__stdout__)
        sys.stderr = Tee(log_file, sys.__stderr__)
    except Exception as e:
        try:
            sys.__stderr__.write(f"[train_hyper] 无法打开日志文件: {e}\n")
        except Exception:
            pass

    main_logger = setup_logger()
    if log_fname:
        main_logger.info(f"Configuration loaded and directories created. Log saved to {log_fname}")
    else:
        main_logger.info("Configuration loaded and directories created.")

    # 2. MODIFIED: 创建并行化的训练环境 (采用更灵活的方式)
    main_logger.info(f"正在创建 {config.N_ENVS} 个并行环境...")
    
    # 定义一个辅助函数，用于根据rank创建单个环境
    def make_env(rank: int, is_eval: bool = False):
        def _init():
            env_logger = setup_logger() 
            # 在这里把 is_eval 传给 GraspHyperEnv
            env = GraspHyperEnv(
                config=config, 
                logger=env_logger,
                env_rank=rank,
                num_envs=config.N_ENVS,
                is_eval=is_eval  # <--- 关键传参
            )
            return env
        return _init

    # [修改点 2]：训练环境 (is_eval 默认为 False，不用改)
    if config.N_ENVS > 1:
        # 注意：这里传给 make_env 的只是 rank
        env = SubprocVecEnv([make_env(i) for i in range(config.N_ENVS)])
    else:
        env = DummyVecEnv([make_env(0)])

    main_logger.info("并行环境创建成功。")
    # =========================================================
    # [新增] 自动奖励归一化 (Reward Normalization)
    # norm_obs=True: 也顺便对 Observation 做归一化 (虽然你在 env 里做过了，但再做一次也没坏处)
    # norm_reward=True: 关键！根据回报的历史统计数据动态缩放 Reward
    # clip_reward=10.0: 防止出现极端巨大的奖励
    # =========================================================
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=125.0)
    
    # 注意：使用 VecNormalize 后，评估时(EvalCallback)也需要同步这些统计信息，
    # 或者让 Eval 环境也使用 VecNormalize (但通常评估看原始 Reward 更直观)。
    # 训练时模型看到的是归一化的 Reward，这能极大稳定训练。

    main_logger.info("已启用 VecNormalize 对 Reward 进行动态平滑。")

    # 3. 设置评估回调 (EvalCallback)
    # 创建一个单独的、非并行的环境用于评估，确保评估结果的稳定和一致性
    eval_env = DummyVecEnv([make_env(0, is_eval=True)])  # <--- 显式开启评估模式
    
    # 2. [关键] 使用 VecNormalize 包裹评估环境
    # norm_obs=True:   必须开启！因为模型需要看到归一化后的状态 (与训练时一致)
    # norm_reward=False: 建议关闭！因为评估时我们想看真实的 1000万 分，而不是归一化后的 2.5 分
    # training=False:  必须关闭！评估过程不应该更新均值/方差，否则会污染训练统计
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.)
    
    # ---------------------------------------------------------

    # 3. [进阶技巧] 同步统计数据 (非常重要！)
    # 因为 eval_env 是新创建的，它的均值/方差初始是默认的(0/1)。
    # 但模型是在训练环境的统计分布上训练的。如果不把训练环境学到的统计数据(obs_rms)
    # 复制给评估环境，模型在评估时看到的“归一化状态”就是错的。
    # ---------------------------------------------------------
    # 这是一个脏办法，但最有效：让 eval_env 直接引用 training_env 的 obs_rms
    eval_env.obs_rms = env.obs_rms    # EvalCallback 会在训练期间定期评估智能体，并只保存表现最好的模型

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.SAVE_PATH,
        log_path=config.SAVE_PATH,
        eval_freq=config.EVAL_FREQ, # 之前商量的 2000
        n_eval_episodes=3,          # <--- 关键：设为 3，确保每次把 A, B, C 都跑一遍
        deterministic=True,
        render=False
    )
    main_logger.info("Evaluation callback configured.")

    # 4. 实例化PPO模型
    # 从配置文件中读取所有PPO超参数
    # model = PPO(
    #     policy=config.PPO_POLICY,
    #     env=env,
    #     learning_rate=config.PPO_LEARNING_RATE,
    #     n_steps=config.PPO_N_STEPS,
    #     batch_size=config.PPO_BATCH_SIZE,
    #     n_epochs=config.PPO_N_EPOCHS,
    #     gamma=config.PPO_GAMMA,
    #     gae_lambda=config.PPO_GAE_LAMBDA,
    #     clip_range=config.PPO_CLIP_RANGE,
    #     ent_coef=config.PPO_ENT_COEF,
    #     vf_coef=config.PPO_VF_COEF,
    #     verbose=1,  # 设置为1可以在控制台看到训练进度
    #     tensorboard_log=config.TENSORBOARD_LOG
    # )
    # main_logger.info("PPO model instantiated.")
    # main_logger.info(f"Model policy:\n{model.policy}")

    # 4. MODIFIED: 实例化 SAC 模型
    # 从配置文件中读取所有 SAC 超参数
    model = None
    
    # 1. 检查是否配置了加载路径
    if config.LOAD_MODEL_PATH and os.path.exists(config.LOAD_MODEL_PATH):
        main_logger.info(f"🔄 正在从检查点加载模型: {config.LOAD_MODEL_PATH}")
        
        # 加载模型 (注意：这里必须传入 env，否则还得手动 set_env)
        # custom_objects 是为了兼容可能的版本差异，通常不需要
        model = SAC.load(
            config.LOAD_MODEL_PATH, 
            env=env,
            # 如果你在微调，可能想改学习率，可以在这里覆盖
            # learning_rate=config.SAC_LEARNING_RATE 
            print_system_info=True
        )
        
        # 2. 尝试加载 Replay Buffer (关键步骤)
        if config.LOAD_REPLAY_BUFFER:
            # 自动推导 buffer 路径: "abc.zip" -> "abc_replay_buffer.pkl"
            buffer_path = config.LOAD_MODEL_PATH.replace(".zip", "") + "_replay_buffer.pkl"
            
            if os.path.exists(buffer_path):
                main_logger.info(f"📥 正在加载 Replay Buffer: {buffer_path}")
                model.load_replay_buffer(buffer_path)
                main_logger.info(f"   - Buffer 大小: {model.replay_buffer.size()}")
            else:
                main_logger.warning(f"⚠️ 未找到 Buffer 文件: {buffer_path}，将从空 Buffer 开始 (需要重新热身)。")
        
        main_logger.info("✅ 模型加载完成，准备继续训练...")
        # [新增/关键] 强制更新 target_entropy！
        # 因为 load() 会恢复旧模型保存的参数，如果不加这一行，你在 config 里改的参数是不生效的。
        if hasattr(config, 'SAC_TARGET_ENTROPY'):
            model.target_entropy = float(config.SAC_TARGET_ENTROPY)
            main_logger.info(f"🔧 [参数热修补] 强制将 Target Entropy 更新为: {model.target_entropy}")
        
        # [原有] 强制重置 learning_starts
        model.learning_starts = 0
    
    else:
        # 3. 如果没配置路径，或者文件不存在，则从头创建 (原逻辑)
        if config.LOAD_MODEL_PATH:
            main_logger.warning(f"⚠️ 指定的加载路径不存在: {config.LOAD_MODEL_PATH}，将从头开始训练。")
        else:
            main_logger.info("🆕 未指定加载路径，开始新的训练。")

        model = SAC(
            policy=config.SAC_POLICY,
            env=env,
            learning_rate=linear_schedule(3e-4, 1e-5),
            buffer_size=config.SAC_BUFFER_SIZE,
            batch_size=config.SAC_BATCH_SIZE,
            target_entropy=config.SAC_TARGET_ENTROPY,
            policy_kwargs=dict(
            # 1. 激活函数：改用 Tanh，防止神经元坏死，输出更平滑
            activation_fn=nn.Tanh, 
            
            # 2. 网络架构：漏斗型，防止过拟合
            # pi = Actor网络 (策略), qf = Critic网络 (价值)
            # 建议尝试 [256, 128]，比默认的 [256, 256] 更适合低维输入
            net_arch=dict(pi=[256, 128], qf=[256, 128]),
            
            # 3. (可选) 启用 gSDE: 让探索更连贯，适合参数控制
            # 如果启用这个，上面的 use_sde 也要设为 True
            log_std_init=-2, 
            ),
    
            # [配合修改] 梯度裁剪 (一定要加！)
            # 防止 Critic 看到一个巨大的 Reward 提升时，梯度爆炸把网络冲坏
            use_sde=True,
            ent_coef=config.SAC_ENT_COEF,
            gamma=config.SAC_GAMMA,
            tau=config.SAC_TAU,
            train_freq=config.SAC_TRAIN_FREQ,
            learning_starts=config.SAC_LEARNING_STARTS,
            verbose=1,
            tensorboard_log=config.TENSORBOARD_LOG
        )
        if config.LOAD_REPLAY_BUFFER:
            # 你需要手动指定那个存有 40k 步数据的 buffer 路径
            # 假设它叫 "sac_final_model_replay_buffer.pkl" 或者之前的 checkpoint
            old_buffer_path = os.path.join(config.SAVE_PATH, "sac_final_model_replay_buffer.pkl")
            
            # 或者如果你有一个特定的 checkpoint buffer，请填在这里：
            # old_buffer_path = "./logs/sac_checkpoint_40000_steps_replay_buffer.pkl"

            if os.path.exists(old_buffer_path):
                main_logger.info(f"🧠 [方案C] 正在为新模型注入旧记忆: {old_buffer_path}")
                model.load_replay_buffer(old_buffer_path)
                main_logger.info(f"   - 成功恢复 {model.replay_buffer.size()} 条经验。")
            else:
                main_logger.warning(f"⚠️ 未找到旧 Buffer 文件: {old_buffer_path}，将从零开始收集数据。")
    if model: # 确保模型已加载
        # 1. 强制将“学习起始步数”设为 0
        # 原因：如果不改，假设原配置是 5000。步数归零后，当前步数(0) < 5000，
        # 模型会停止梯度更新，傻跑 5000 步来收集数据。
        # 设为 0 后，模型会发现 0 >= 0，从而第一步就开始利用旧 Buffer 更新网络。
        model.learning_starts = 0
        
        main_logger.info("🔧 已强制重置 learning_starts = 0，确保续训立即生效。")

    # ... (EvalCallback 设置保持不变) ...
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, # 每 10000 步保存一次
        save_path=config.SAVE_PATH,
        name_prefix="sac_checkpoint",
        save_replay_buffer=True, # <--- 关键：开启 Buffer 保存
        save_vecnormalize=True
    )
    # 5. 启动学习
    try:
        # 注意: reset_num_timesteps=False 表示接续之前的步数计数 (Tensorboard 不会断崖)
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=(model is None) # 如果是新模型就重置，如果是加载的就不重置
            # reset_num_timesteps=True # 如果是新模型就重置，如果是加载的就不重置
        )
    except KeyboardInterrupt:
        main_logger.warning("Training interrupted by user.")
    finally:
        # =================================================================
        # [修改] 保存逻辑：同时保存模型和 Replay Buffer
        # =================================================================
        final_model_path = os.path.join(config.SAVE_PATH, "sac_final_model.zip")
        final_buffer_path = os.path.join(config.SAVE_PATH, "sac_final_model_replay_buffer.pkl")
        
        model.save(final_model_path)
        main_logger.info(f"Final model saved to {final_model_path}")
        
        # 保存 Buffer (虽然文件很大，但对续训至关重要)
        main_logger.info("Saving replay buffer (this may take a while)...")
        model.save_replay_buffer(final_buffer_path)
        main_logger.info(f"Final replay buffer saved to {final_buffer_path}")
        
        env.close()
        eval_env.close()
        main_logger.info("Environments closed. Training finished.")

        # 关闭并恢复 stdout/stderr
        try:
            if log_file:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                log_file.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()