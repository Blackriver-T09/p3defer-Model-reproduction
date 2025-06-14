#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P3Defer 推理脚本
用于加载训练好的策略网络并进行预测
"""

import os
import json
import torch
import argparse
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_policy import (
    PolicyNetwork, 
    ValueNetwork, 
    StateEncoder, 
    PrivateMemory, 
    Config,
    set_seed
)


class P3DeferModel:
    def __init__(self, config, policy_path):
        self.config = config
        
        # 加载策略网络
        self.policy = PolicyNetwork(config.state_dim, config.hidden_dim, config.n_actions).to(config.device)
        
        # 加载训练好的模型参数
        checkpoint = torch.load(policy_path, map_location=config.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.eval()
        
        # 初始化其他组件
        print("加载本地模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.local_model_path)
        
        # 使用指定的GPU
        if len(config.gpu_ids) >= 1:
            local_gpu = config.gpu_ids[0]
        else:
            local_gpu = 0
            
        self.local_model = AutoModelForCausalLM.from_pretrained(
            config.local_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.local_model.eval()
        
        # 加载服务器模型或准备API连接
        if not config.use_api:
            print("加载服务器模型...")
            self.server_tokenizer = AutoTokenizer.from_pretrained(config.server_model_path)
            self.server_model = AutoModelForCausalLM.from_pretrained(
                config.server_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.server_model.eval()
        
        # 私有记忆模块（如果使用）
        if config.use_mask:
            self.private_memory = PrivateMemory()
        else:
            self.private_memory = None
            
        self.state_encoder = StateEncoder(config, self.tokenizer)
        
    def generate_local_response(self, query):
        """使用本地模型生成响应"""
        inputs = self.tokenizer(query, return_tensors="pt").to(self.local_model.device)
        with torch.no_grad():
            outputs = self.local_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.7
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取输入之后的响应部分
        response = response[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        return response
    
    def generate_server_response(self, query):
        """使用服务器模型生成响应"""
        if self.config.use_api:
            # API调用方式
            response = requests.post(
                self.config.server_api_url,
                json={"query": query}
            )
            if response.status_code == 200:
                return response.json()["output"]
            else:
                return f"Error: {response.status_code}"
        else:
            # 直接使用本地加载的模型
            inputs = self.server_tokenizer(query, return_tensors="pt").to(self.server_model.device)
            with torch.no_grad():
                outputs = self.server_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.7
                )
            response = self.server_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取输入之后的响应部分
            response = response[len(self.server_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            return response
    
    def predict(self, query):
        """使用P3Defer架构进行预测"""
        # 获取本地模型响应
        local_response = self.generate_local_response(query)
        
        # 编码状态
        state = self.state_encoder.encode_state(query, local_response)
        
        # 使用策略网络选择动作
        with torch.no_grad():
            action_probs = self.policy(state)
            action = torch.argmax(action_probs).item()
        
        # 根据选择的动作进行相应操作
        if action == 0:  # 返回本地响应
            final_response = local_response
            action_name = "使用本地响应"
        elif action == 1 or action == 2:  # 转发原始查询到服务器
            final_response = self.generate_server_response(query)
            action_name = "转发到服务器"
        
        return {
            "query": query,
            "local_response": local_response,
            "action": action_name,
            "action_probs": action_probs.tolist(),
            "final_response": final_response
        }


def parse_args():
    parser = argparse.ArgumentParser(description="P3Defer 推理")
    
    # 模型参数
    parser.add_argument("--policy_path", type=str, required=True, help="策略网络权重路径")
    parser.add_argument("--local_model", type=str, default="./models/gemma-2b", help="本地模型路径")
    parser.add_argument("--server_model", type=str, default="./models/gemma-7b", help="服务器模型路径")
    parser.add_argument("--server_api", type=str, default="http://127.0.0.1:5000/predict", help="服务器API地址")
    parser.add_argument("--use_api", action="store_true", help="是否使用API调用服务器模型")
    
    # 推理参数
    parser.add_argument("--input", type=str, help="输入查询")
    parser.add_argument("--input_file", type=str, help="输入查询文件路径")
    parser.add_argument("--output_file", type=str, default="results.jsonl", help="输出结果文件路径")
    
    # 其他参数
    parser.add_argument("--privacy_weight", type=float, default=0.5, help="隐私奖励权重λ")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="使用的GPU ID，用逗号分隔")
    
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化配置
    config = Config()
    config.local_model_path = args.local_model
    config.server_model_path = args.server_model
    config.server_api_url = args.server_api
    config.privacy_weight = args.privacy_weight
    config.seed = args.seed
    config.use_api = args.use_api
    
    # 设置GPU IDs
    config.gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")]
    
    # 打印GPU信息
    print(f"使用GPU: {config.gpu_ids}")
    print(f"GPU总数: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 初始化模型
    model = P3DeferModel(config, args.policy_path)
    
    # 处理输入
    if args.input:
        # 单个查询模式
        result = model.predict(args.input)
        
        print("\n===== P3Defer推理结果 =====")
        print(f"查询: {result['query']}")
        print(f"本地响应: {result['local_response']}")
        print(f"选择动作: {result['action']}")
        print(f"动作概率: {result['action_probs']}")
        print(f"最终响应: {result['final_response']}")
        
        # 保存结果
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
    elif args.input_file:
        # 批量查询模式
        results = []
        with open(args.input_file, "r", encoding="utf-8") as f:
            queries = f.readlines()
        
        for i, query in enumerate(queries):
            query = query.strip()
            if not query:
                continue
                
            print(f"处理查询 {i+1}/{len(queries)}...")
            result = model.predict(query)
            results.append(result)
        
        # 保存结果
        with open(args.output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"已将所有结果保存到 {args.output_file}")
    else:
        print("错误: 必须提供 --input 或 --input_file 参数")


if __name__ == "__main__":
    main() 