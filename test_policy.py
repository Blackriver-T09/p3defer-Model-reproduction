#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P3Defer 策略网络测试脚本
用于验证策略网络和相关组件的功能
"""

import torch
from train_policy import (
    PolicyNetwork, 
    ValueNetwork, 
    StateEncoder, 
    PrivateMemory,
    Config
)

def test_policy_network():
    """测试策略网络的前向传播"""
    config = Config()
    policy = PolicyNetwork(config.state_dim, config.hidden_dim, config.n_actions)
    
    # 创建随机状态
    state = torch.rand(config.state_dim)
    
    # 前向传播
    action_probs = policy(state)
    
    print("===== 策略网络测试 =====")
    print(f"输入状态形状: {state.shape}")
    print(f"输出动作概率: {action_probs}")
    print(f"动作概率和: {action_probs.sum().item()}")  # 应该约为1
    assert abs(action_probs.sum().item() - 1.0) < 1e-5, "动作概率和应该等于1"
    print("策略网络测试通过!\n")

def test_value_network():
    """测试价值网络的前向传播"""
    config = Config()
    value = ValueNetwork(config.state_dim, config.hidden_dim)
    
    # 创建随机状态
    state = torch.rand(config.state_dim)
    
    # 前向传播
    state_value = value(state)
    
    print("===== 价值网络测试 =====")
    print(f"输入状态形状: {state.shape}")
    print(f"输出状态值: {state_value.item()}")
    print("价值网络测试通过!\n")

def test_private_memory():
    """测试私有记忆模块的掩码功能"""
    private_memory = PrivateMemory()
    
    # 测试不同类型的输入
    test_cases = [
        "Please send my records to john@gmail.com",
        "Dr. Smith said my blood pressure is 120/80",
        "My SSN is 123-45-6789",
        "Hello, my name is John Smith",
        "No private information here"
    ]
    
    print("===== 私有记忆模块测试 =====")
    for i, text in enumerate(test_cases):
        masked_text = private_memory.mask_private_info(text)
        print(f"原始文本 {i+1}: {text}")
        print(f"掩码后文本 {i+1}: {masked_text}")
        print()
    
    print("私有记忆模块测试通过!\n")

def test_state_encoder():
    """测试状态编码器"""
    config = Config()
    # 创建一个假的tokenizer
    class DummyTokenizer:
        def __init__(self):
            pass
    
    tokenizer = DummyTokenizer()
    state_encoder = StateEncoder(config, tokenizer)
    
    # 测试不同类型的输入
    test_cases = [
        {
            "x": "What is 2+2?", 
            "y_L": "To solve this problem, I need to add 2 and 2. 2+2=4. Therefore, the answer is 4."
        },
        {
            "x": "John has 5 apples and gives 2 to Mary. How many does he have left?", 
            "y_L": "John has 5-2=3 apples left."
        },
        {
            "x": "What is the square root of 144?", 
            "y_L": "I don't know."
        }
    ]
    
    print("===== 状态编码器测试 =====")
    for i, case in enumerate(test_cases):
        x, y_L = case["x"], case["y_L"]
        state = state_encoder.encode_state(x, y_L)
        
        print(f"测试案例 {i+1}:")
        print(f"输入 x: {x}")
        print(f"本地回答 y_L: {y_L}")
        print(f"状态向量形状: {state.shape}")
        print(f"状态向量前两维 (隐私): {state[:2].tolist()}")
        print(f"状态向量后两维 (质量): {state[-2:].tolist()}")
        print()
    
    print("状态编码器测试通过!\n")

if __name__ == "__main__":
    print("开始P3Defer组件测试...\n")
    
    # 测试策略网络
    test_policy_network()
    
    # 测试价值网络
    test_value_network()
    
    # 测试私有记忆模块
    test_private_memory()
    
    # 测试状态编码器
    test_state_encoder()
    
    print("所有测试通过!") 