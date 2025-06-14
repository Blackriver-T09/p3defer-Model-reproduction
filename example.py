#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P3Defer 示例脚本
用于演示完整的P3Defer流程
"""

import os
import torch
from train_policy import (
    PolicyNetwork, 
    ValueNetwork, 
    StateEncoder, 
    PrivateMemory, 
    Config,
    LocalLLM,
    ServerLLM
)

# 检查保存目录
os.makedirs("checkpoints", exist_ok=True)

# 示例查询
example_queries = [
    "What is 7 * 8?",
    "John has 5 apples and gives 2 to Mary. How many apples does John have left?",
    "What is the square root of 144?",
    "My email is john@example.com and my phone is 555-123-4567. Can you help me with this math problem: 23 * 4?"
]

# 辅助函数：可视化动作选择
def visualize_action(query, local_response, action, action_probs, final_response):
    action_names = ["返回本地", "转发原始", "转发原始"]
    
    print("\n" + "="*50)
    print(f"查询: {query}")
    print("-"*50)
    print(f"本地模型回答: {local_response}")
    print("-"*50)
    print("策略网络决策:")
    for i, (name, prob) in enumerate(zip(action_names, action_probs)):
        marker = "✓" if i == action else " "
        print(f"{marker} {name}: {prob:.4f}")
    print("-"*50)
    print(f"最终响应: {final_response}")
    print("="*50)

def main():
    print("P3Defer 示例流程\n")
    
    # 初始化配置
    config = Config()
    
    # 打印GPU信息
    print(f"使用GPU: {config.gpu_ids}")
    print(f"GPU总数: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 为了演示，我们先创建一个随机策略网络
    print("初始化策略网络...")
    policy = PolicyNetwork(config.state_dim, config.hidden_dim, config.n_actions)
    
    # 保存一个随机初始化的策略网络（仅供演示）
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': ValueNetwork(config.state_dim, config.hidden_dim).state_dict(),
        'policy_optimizer': None,
        'value_optimizer': None,
    }, "checkpoints/demo_policy.pth")
    
    print("加载本地模型(Gemma-2B)...")
    try:
        local_llm = LocalLLM(config.local_model_path)
    except Exception as e:
        print(f"无法加载本地模型: {e}")
        print("请确保本地模型路径 '{config.local_model_path}' 存在，或修改 Config 中的路径。")
        return
    
    print("加载服务器模型(Gemma-7B)...")
    try:
        if config.use_api:
            # 使用API
            server_llm = ServerLLM(None)
            server_llm.use_api = True
            server_llm.api_url = config.server_api_url
        else:
            # 直接加载本地模型
            server_llm = ServerLLM(config.server_model_path)
    except Exception as e:
        print(f"无法加载服务器模型: {e}")
        print("将使用本地模型代替服务器模型进行演示。")
        server_llm = None
    
    private_memory = None
    if config.use_mask:
        private_memory = PrivateMemory()
        
    state_encoder = StateEncoder(config, local_llm.tokenizer)
    
    # 处理示例查询
    print("\n开始处理示例查询...")
    
    for query in example_queries:
        # 1. 本地模型生成响应
        local_response = local_llm.generate(query)
        
        # 2. 编码状态
        state = state_encoder.encode_state(query, local_response)
        
        # 3. 策略网络选择动作
        action_probs = policy(state).detach().cpu().numpy()
        action = torch.argmax(torch.tensor(action_probs)).item()
        
        # 4. 执行选定的动作
        if action == 0:  # 返回本地回答
            final_response = local_response
        elif action == 1 or action == 2:  # 转发给服务器
            try:
                if server_llm:
                    final_response = server_llm.generate(query)
                else:
                    print("服务器模型不可用，使用本地回答代替。")
                    final_response = local_response
            except Exception as e:
                print(f"服务器模型调用失败: {e}")
                print("确保服务器API运行中或服务器模型已加载。暂时使用本地回答作为替代。")
                final_response = local_response
        
        # 5. 可视化结果
        visualize_action(query, local_response, action, action_probs, final_response)
    
    print("\n示例流程完成！")
    print("如需训练策略网络，请运行:")
    print("  python run_training.py --max_train_steps 10000")
    print("如需进行推理，请运行:")
    print("  python inference.py --policy_path checkpoints/policy_final.pth --input '你的查询'")
    print("\nP3Defer 项目 - 谢谢使用!")


if __name__ == "__main__":
    main() 