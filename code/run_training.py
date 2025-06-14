#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P3Defer 策略网络训练主脚本
用于执行策略网络的训练过程
"""

import os
import argparse
import torch
from train_policy import Config, train, set_seed
import random


def parse_args():
    parser = argparse.ArgumentParser(description="P3Defer 策略网络训练")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="每次更新的PPO轮数")
    parser.add_argument("--update_timestep", type=int, default=2000, help="策略网络更新间隔")
    parser.add_argument("--max_train_steps", type=int, default=100000, help="最大训练步数")
    
    # 模型路径
    parser.add_argument("--local_model", type=str, default="./models/gemma-2b", help="本地模型路径")
    parser.add_argument("--server_model", type=str, default="./models/gemma-7b", help="服务器模型路径")
    parser.add_argument("--server_api", type=str, default="http://127.0.0.1:5000/predict", help="服务器API地址")
    parser.add_argument("--use_api", action="store_true", help="是否使用API调用服务器模型")
    parser.add_argument("--use_mask", action="store_true", help="是否使用私有记忆模块(掩码)")
    
    # 其他参数
    parser.add_argument("--privacy_weight", type=float, default=0.5, help="隐私奖励权重λ")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点保存目录")
    parser.add_argument("--data_path", type=str, default="./data/gsm8k/train.jsonl", help="训练数据路径")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="使用的GPU ID，用逗号分隔")
    parser.add_argument("--verbose", action="store_true", help="是否输出详细训练信息")
    parser.add_argument("--log_interval", type=int, default=100, help="日志输出间隔步数")
    
    # 快速训练模式参数
    parser.add_argument("--fast_mode", action="store_true", help="是否使用快速训练模式，减少服务器模型调用")
    parser.add_argument("--server_sample_rate", type=float, default=0.3, help="快速模式下服务器模型的采样率(0-1)")
    parser.add_argument("--precompute", type=int, default=100, help="预计算本地模型回答的样本数量")
    parser.add_argument("--max_tokens", type=int, default=128, help="生成时的最大token数")
    
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化配置
    config = Config()
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.ppo_epochs = args.ppo_epochs
    config.update_timestep = args.update_timestep
    config.max_train_steps = args.max_train_steps
    config.local_model_path = args.local_model
    config.server_model_path = args.server_model
    config.server_api_url = args.server_api
    config.privacy_weight = args.privacy_weight
    config.seed = args.seed
    config.data_path = args.data_path
    config.use_api = args.use_api
    config.use_mask = args.use_mask
    
    # 设置GPU IDs
    config.gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")]
    
    # 设置日志参数
    config.verbose = args.verbose
    config.log_interval = args.log_interval
    
    # 打印训练配置
    print("\n" + "=" * 60)
    print("            P3Defer 策略网络训练配置              ")
    print("=" * 60)
    print(f"本地模型路径:          {config.local_model_path}")
    print(f"服务器模型路径:        {config.server_model_path}")
    print(f"服务器API:            {config.server_api_url}")
    print(f"使用API模式:          {config.use_api}")
    print(f"使用私有记忆模块(掩码): {config.use_mask}")
    print(f"批处理大小:           {config.batch_size}")
    print(f"学习率:               {config.learning_rate}")
    print(f"PPO更新轮数:          {config.ppo_epochs}")
    print(f"更新间隔:             {config.update_timestep}")
    print(f"最大训练步数:          {config.max_train_steps}")
    print(f"隐私权重λ:            {config.privacy_weight}")
    print(f"随机种子:             {config.seed}")
    print(f"数据路径:             {config.data_path}")
    print(f"检查点目录:           {args.checkpoint_dir}")
    print(f"使用GPU:              {config.gpu_ids}")
    print(f"设备:                {config.device}")
    print(f"详细输出模式:         {config.verbose}")
    print(f"日志输出间隔:         {config.log_interval}")
    print("=" * 60)
    
    # 开始训练
    import datetime
    start_time = datetime.datetime.now()
    print(f"\n开始训练时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    train(config)
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"结束训练时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总训练时长: {duration}")
    print("\n训练完成! 策略网络已保存到: {args.checkpoint_dir}/policy_final.pth")


if __name__ == "__main__":
    main() 