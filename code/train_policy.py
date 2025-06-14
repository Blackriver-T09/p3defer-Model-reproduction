#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P3Defer 策略网络训练脚本
训练基于PPO的策略网络，用于决策本地LLM的回答是返回、转发或掩码后转发
"""

import os
import json
import torch
import numpy as np
import random
import requests
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Tuple, Optional, Union
import re
from collections import deque
import Levenshtein
import time


# 设置随机种子以确保结果可重现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 配置参数
class Config:
    def __init__(self):
        # 模型路径
        self.local_model_path = "./models/gemma-2b"
        self.server_model_path = "./models/gemma-7b"
        self.server_api_url = "http://127.0.0.1:5000/predict"
        
        # 训练参数
        self.batch_size = 16
        self.learning_rate = 3e-4
        self.ppo_epochs = 10
        self.max_train_steps = 100000
        self.update_timestep = 2000
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # PPO超参数
        self.ppo_batch_size = 64
        self.buffer_size = 10000
        
        # 隐私相关参数
        self.privacy_weight = 0.5  # 隐私奖励的权重λ
        
        # 网络架构参数
        self.state_dim = 128  # 状态向量维度 = privacy_embedding_dim + quality_embedding_dim
        self.hidden_dim = 256
        self.n_actions = 3  # a1: 返回本地, a2: 转发原始, a3: 转发原始（简化，不使用掩码）
        
        # 其他参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_ids = [0, 1, 2, 3]  # 使用前4张显卡
        self.seed = 42
        
        # 数据集路径
        self.data_path = "./data/gsm8k/train.jsonl"
        self.test_data_path = "./data/gsm8k/test.jsonl"
        
        # 是否使用API
        self.use_api = False  # 设置为False，不使用API而是直接加载本地模型
        
        # 是否使用私有记忆模块（掩码）
        self.use_mask = False  # 设置为False，不使用掩码
        
        # 日志参数
        self.verbose = False  # 详细输出模式
        self.log_interval = 100  # 日志输出间隔步数
        
        # 快速训练模式参数
        self.fast_mode = False  # 快速训练模式
        self.server_sample_rate = 0.3  # 服务器模型采样率
        self.precompute = 100  # 预计算样本数
        self.max_tokens = 128  # 最大生成token数


# 策略网络 π_θ
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, n_actions)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs


# 价值网络 V_ψ
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value


# 状态编码器
class StateEncoder:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        # 加载隐私标记列表（简化版，实际应包含更多隐私标记）
        self.privacy_tokens = self._load_privacy_tokens()
        
    def _load_privacy_tokens(self):
        # 这里是简化版的隐私标记列表
        # 在实际应用中，应该从更全面的隐私词库或通过规则生成
        return [
            "ssn", "password", "address", "phone", "email", "credit card", "bank", 
            "account", "patient", "medical", "doctor", "hospital", "social security",
            "john", "mary", "james", "robert", "david", "michael", "sarah", "linda",
            "gmail.com", "yahoo.com", "hotmail.com", "dr.", "mr.", "ms.", "mrs."
        ]
    
    def compute_privacy_embedding(self, x: str) -> torch.Tensor:
        """计算输入文本的隐私嵌入向量"""
        # 检测是否包含隐私标记
        contains_privacy = False
        for token in self.privacy_tokens:
            if token.lower() in x.lower():
                contains_privacy = True
                break
                
        # 生成简单的隐私嵌入向量 [has_privacy, no_privacy]
        privacy_embedding = torch.tensor([float(contains_privacy), float(not contains_privacy)], 
                                        device=self.config.device)
        
        # 扩展到更大的维度以增强表示能力（此处简单复制）
        privacy_dim = self.config.state_dim // 2
        privacy_embedding = privacy_embedding.repeat(privacy_dim // 2)
        
        return privacy_embedding
    
    def compute_quality_embedding(self, x: str, y_L: str) -> torch.Tensor:
        """计算本地模型回答的质量嵌入向量"""
        # 这里简单使用回答长度和特定关键词作为质量指标
        # 实际应用中可以使用更复杂的方法（如与黄金答案的相似度）
        
        # 1. 回答长度（标准化到0-1）
        length_score = min(len(y_L) / 500, 1.0)
        
        # 2. 是否包含数学解题步骤的指标词
        steps_keywords = ["step", "solve", "calculate", "equal", "therefore", "result", "answer"]
        steps_score = 0
        for keyword in steps_keywords:
            if keyword in y_L.lower():
                steps_score += 1/len(steps_keywords)
        
        # 3. 是否包含引用公式或方程·
        formula_patterns = [r"\d+\s*[+\-*/]\s*\d+", r"=\s*\d+", r"\(\d+\)"]
        formula_score = 0
        for pattern in formula_patterns:
            if re.search(pattern, y_L):
                formula_score += 1/len(formula_patterns)
        
        # 4. 是否包含最终答案
        answer_keywords = ["answer is", "result is", "solution is", "therefore"]
        answer_score = 0
        for keyword in answer_keywords:
            if keyword in y_L.lower():
                answer_score += 1/len(answer_keywords)
        
        # 组合成质量分数并转换为嵌入向量
        quality_score = (length_score + steps_score + formula_score + answer_score) / 4
        quality_embedding = torch.tensor([quality_score, 1 - quality_score], device=self.config.device)
        
        # 扩展到更大的维度
        quality_dim = self.config.state_dim // 2
        quality_embedding = quality_embedding.repeat(quality_dim // 2)
        
        return quality_embedding
    
    def encode_state(self, x: str, y_L: str) -> torch.Tensor:
        """将输入和本地模型回答编码为状态向量"""
        e_p = self.compute_privacy_embedding(x)
        e_q = self.compute_quality_embedding(x, y_L)
        s_t = torch.cat([e_p, e_q])
        return s_t


# 私有记忆模块M
class PrivateMemory:
    def __init__(self, privacy_tokens=None):
        # 初始化私有记忆（隐私标记列表）
        if privacy_tokens is None:
            self.privacy_tokens = [
                "ssn", "password", "address", "phone", "email", "credit card", "bank", 
                "account", "patient", "medical", "doctor", "hospital", "social security",
                "john", "mary", "james", "robert", "david", "michael", "sarah", "linda",
                "gmail.com", "yahoo.com", "hotmail.com", "dr.", "mr.", "ms.", "mrs."
            ]
        else:
            self.privacy_tokens = privacy_tokens
    
    def mask_private_info(self, text: str) -> str:
        """使用隐私掩码替换文本中的隐私信息"""
        masked_text = text
        
        # 1. 精确匹配
        for token in self.privacy_tokens:
            if token in masked_text.lower():
                pattern = re.compile(re.escape(token), re.IGNORECASE)
                masked_text = pattern.sub("[PRIVATE]", masked_text)
        
        # 2. 基于Levenshtein距离的模糊匹配
        words = re.findall(r'\b\w+\b', masked_text)
        for word in words:
            for token in self.privacy_tokens:
                # 如果单词与隐私标记的Levenshtein距离小于阈值，认为是相似的
                if len(word) > 3 and Levenshtein.distance(word.lower(), token.lower()) <= 2:
                    pattern = re.compile(r'\b' + re.escape(word) + r'\b')
                    masked_text = pattern.sub("[PRIVATE]", masked_text)
                    break
        
        # 3. 模式匹配（电子邮件、电话号码等）
        # 电子邮件
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        masked_text = email_pattern.sub("[PRIVATE_EMAIL]", masked_text)
        
        # 电话号码
        phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        masked_text = phone_pattern.sub("[PRIVATE_PHONE]", masked_text)
        
        return masked_text


# 经验回放缓冲区
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.actions), \
               np.array(self.probs), np.array(self.vals), \
               np.array(self.rewards), np.array(self.dones), batches


# PPO代理
class PPOAgent:
    def __init__(self, config):
        self.config = config
        self.policy = PolicyNetwork(config.state_dim, config.hidden_dim, config.n_actions).to(config.device)
        self.value = ValueNetwork(config.state_dim, config.hidden_dim).to(config.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config.learning_rate)
        self.memory = PPOMemory(config.ppo_batch_size)
    
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.config.device)
        action_probs = self.policy(state)
        value = self.value(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        probs = torch.squeeze(action_probs).detach().cpu().numpy()
        action = torch.squeeze(action).detach().cpu().numpy()
        value = torch.squeeze(value).detach().cpu().numpy()
        
        return action, probs, value
    
    def update(self):
        states, actions, old_probs, vals, rewards, dones, batches = self.memory.generate_batches()
        
        values = vals
        advantage = np.zeros(len(rewards), dtype=np.float32)
        
        # 计算广义优势估计(GAE)
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
                a_t += discount * (rewards[k] + self.config.gamma * values[k+1] * (1-int(dones[k])) - values[k])
                discount *= self.config.gamma * self.config.gae_lambda
            advantage[t] = a_t
        
        advantage = torch.tensor(advantage).to(self.config.device)
        values = torch.tensor(values).to(self.config.device)
        
        # PPO更新循环
        for _ in range(self.config.ppo_epochs):
            for batch in batches:
                states_batch = torch.tensor(states[batch], dtype=torch.float).to(self.config.device)
                old_probs_batch = torch.tensor(old_probs[batch]).to(self.config.device)
                actions_batch = torch.tensor(actions[batch]).to(self.config.device)
                
                # 计算新的动作概率
                action_probs = self.policy(states_batch)
                dist = torch.distributions.Categorical(action_probs)
                new_probs = dist.log_prob(actions_batch)
                old_probs_batch = torch.tensor(
                    [old_probs[batch[i]][actions_batch[i]] for i in range(len(batch))], 
                    dtype=torch.float
                ).to(self.config.device)
                
                # 计算比率和裁剪后的目标函数
                prob_ratio = torch.exp(new_probs - torch.log(old_probs_batch))
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = advantage[batch] * torch.clamp(
                    prob_ratio, 
                    1-self.config.clip_param, 
                    1+self.config.clip_param
                )
                
                # 计算损失函数
                policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + values[batch]
                value_preds = self.value(states_batch).squeeze()
                value_loss = F.mse_loss(value_preds, returns)
                
                entropy = dist.entropy().mean()
                total_loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
                
                # 优化策略网络和价值网络
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()
        
        self.memory.clear_memory()


# GSM8K数据集加载器
class GSM8KDataset:
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# 与本地LLM交互
class LocalLLM:
    def __init__(self, model_path, gpu_ids=[0]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"  # 使用自动分配而不是手动映射
        )
        self.model.eval()
    
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True  # 修复警告：当设置temperature时需要开启do_sample
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取输入之后的响应部分
        response = response[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        return response

    def quick_evaluate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # 只生成20个token快速评估
            outputs = self.model.generate(**inputs, max_new_tokens=20)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# 与服务器LLM通信（现在直接从本地加载模型，不使用API）
class ServerLLM:
    def __init__(self, model_path, gpu_ids=[1, 2, 3]):
        self.use_api = False
        self.api_url = None
        
        # 如果提供了model_path，则直接加载模型
        if model_path and not self.use_api:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 使用"auto"作为device_map，让模型自动分配到可用GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map="auto"  # 使用自动分配而不是手动映射
            )
            self.model.eval()
        else:
            # 仍然保留API调用方式作为备选
            self.api_url = "http://127.0.0.1:5000/predict"
    
    def generate(self, prompt):
        if self.use_api and self.api_url:
            # 使用API方式（备选）
            response = requests.post(
                self.api_url,
                json={"query": prompt}
            )
            if response.status_code == 200:
                return response.json()["output"]
            else:
                return f"Error: {response.status_code}"
        else:
            # 直接使用本地模型
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=False
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取输入之后的响应部分
            response = response[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            return response


# 奖励计算
class RewardCalculator:
    def __init__(self, config):
        self.config = config
    
    def compute_quality_reward(self, y: str, y_ref: str) -> float:
        """
        计算回答质量奖励，基于预测和参考答案的相似度
        """
        # 基于简单的文本重叠计算相似度（实际中可以使用更复杂的方法）
        y_tokens = set(y.lower().split())
        y_ref_tokens = set(y_ref.lower().split())
        
        if not y_ref_tokens:  # 防止除零错误
            return 0.0
        
        # Jaccard相似度 = 交集大小 / 并集大小
        intersection = len(y_tokens.intersection(y_ref_tokens))
        union = len(y_tokens.union(y_ref_tokens))
        
        similarity = intersection / union if union > 0 else 0
        return similarity
    
    def compute_privacy_reward(self, x: str, action: int, x_masked: Optional[str] = None) -> float:
        """
        计算隐私保护奖励
        action: 0=返回本地, 1=转发原始, 2=转发原始
        
        注意：现在a2和a3都转发原始查询，所以奖励计算方式相同
        """
        # 简单的隐私检测（检查是否包含隐私标记）
        contains_privacy = False
        privacy_tokens = [
            "ssn", "password", "address", "phone", "email", "credit card", "bank", 
            "account", "patient", "medical", "doctor", "hospital", "social security",
            "john", "mary", "james", "robert", "david", "michael", "sarah", "linda",
            "gmail.com", "yahoo.com", "hotmail.com", "dr.", "mr.", "ms.", "mrs."
        ]
        
        for token in privacy_tokens:
            if token.lower() in x.lower():
                contains_privacy = True
                break
        
        if not contains_privacy:
            # 不包含隐私内容
            return 1.0  # 最大奖励
        
        # 包含隐私内容
        if action == 0:  # 返回本地 - 好
            return 1.0
        elif action == 1 or action == 2:  # 转发原始 - 不好
            return -1.0
        
        return 0.0
    
    def compute_reward(self, x: str, y: str, y_ref: str, action: int, x_masked: Optional[str] = None) -> float:
        """
        计算总体奖励，结合质量和隐私
        """
        quality_reward = self.compute_quality_reward(y, y_ref)
        privacy_reward = self.compute_privacy_reward(x, action, x_masked)
        
        # 使用配置的权重λ来平衡质量和隐私
        total_reward = quality_reward + self.config.privacy_weight * privacy_reward
        return total_reward


# 训练循环
def train(config):
    # 设置随机种子
    set_seed(config.seed)
    
    # 打印GPU使用情况
    print(f"使用GPU: {config.gpu_ids}")
    print(f"GPU总数: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 加载模型和组件
    print("加载本地模型(Gemma-2B)...")
    local_llm = LocalLLM(config.local_model_path, gpu_ids=[config.gpu_ids[0]])
    
    print("加载服务器模型(Gemma-7B)...")
    if config.use_api:
        # 使用API
        server_llm = ServerLLM(None)
        server_llm.use_api = True
        server_llm.api_url = config.server_api_url
    else:
        # 直接加载本地模型，使用指定的GPU
        server_llm = ServerLLM(config.server_model_path, gpu_ids=config.gpu_ids[1:])
    
    private_memory = None
    if config.use_mask:
        private_memory = PrivateMemory()
    
    state_encoder = StateEncoder(config, local_llm.tokenizer)
    reward_calculator = RewardCalculator(config)
    ppo_agent = PPOAgent(config)
    
    # 加载数据集
    print(f"加载数据集: {config.data_path}")
    dataset = GSM8KDataset(config.data_path)
    print(f"数据集大小: {len(dataset)}个样本")
    
    # 训练步骤计数器
    step_counter = 0
    episode_rewards = []
    
    # 用于记录训练时间和估计剩余时间
    start_time = time.time()
    
    # 添加快速模式标志
    fast_mode = True  # 设置为True以启用快速训练模式
    server_sample_rate = 0.3  # 只有30%的概率使用服务器模型
    
    # 预先加载一些常见问题的回答，加速训练
    if config.fast_mode and config.precompute > 0:
        print("预处理常见问题...")
        precompute_count = min(config.precompute, len(dataset))
        precompute_indices = random.sample(range(len(dataset)), precompute_count)
        for idx in precompute_indices:
            sample = dataset[idx]
            x = sample["question"]
            # 预计算并缓存本地模型响应
            _ = local_llm.generate(x)
            if idx % 10 == 0:
                print(f"预处理进度: {idx+1}/{precompute_count}")
    
    # 主训练循环
    print("开始训练P3Defer策略网络...")
    print(f"总训练步数: {config.max_train_steps}")
    print(f"快速训练模式: {config.fast_mode}")
    print(f"服务器模型采样率: {config.server_sample_rate*100:.1f}%")
    print("=" * 50)
    
    response_cache = {}  # 缓存已生成的响应
    
    for step in range(config.max_train_steps):
        # 随机选择一个样本
        sample_idx = random.randint(0, len(dataset)-1)
        sample = dataset[sample_idx]
        x = sample["question"]
        y_ref = sample["answer"]
        
        # 在生成响应前检查缓存
        cache_key = x[:100]  # 使用问题前100个字符作为键
        if cache_key in response_cache:
            y_L = response_cache[cache_key]
        else:
            y_L = local_llm.generate(x)
            response_cache[cache_key] = y_L
        
        # 编码状态
        s_t = state_encoder.encode_state(x, y_L)
        
        # 选择动作
        action, action_probs, value = ppo_agent.choose_action(s_t.detach().cpu().numpy())
        
        # 根据动作生成最终响应
        x_masked = None
        if action == 0:  # 返回本地回答
            y = y_L
        elif action == 1 or action == 2:  # 转发给服务器
            # 注意：这里将a2和a3都当作直接转发，不使用掩码
            y = server_llm.generate(x)
        
        # 计算奖励
        reward = reward_calculator.compute_reward(x, y, y_ref, action, x_masked)
        episode_rewards.append(reward)
        
        # 存储状态转换
        done = True  # 每个样本作为独立的回合
        ppo_agent.memory.store_memory(s_t.detach().cpu().numpy(), action, action_probs, value, reward, done)
        
        # 更新步数
        step_counter += 1
        
        # 定期输出训练进度
        if step_counter % config.log_interval == 0 or step_counter == 1:
            # 计算进度和时间估计
            progress = step_counter / config.max_train_steps * 100
            elapsed_time = time.time() - start_time
            steps_per_sec = step_counter / elapsed_time
            remaining_steps = config.max_train_steps - step_counter
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            
            # 格式化时间
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
            
            # 计算最近的平均奖励
            recent_rewards = episode_rewards[-min(config.log_interval, len(episode_rewards)):]
            avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
            
            # 动作分布
            action_counts = [0, 0, 0]
            for i in range(min(config.log_interval, len(ppo_agent.memory.actions))):
                idx = -i-1  # 从最后一个开始
                if idx >= -len(ppo_agent.memory.actions):
                    action_counts[ppo_agent.memory.actions[idx]] += 1
            
            action_distribution = [count / sum(action_counts) * 100 for count in action_counts] if sum(action_counts) > 0 else [0, 0, 0]
            
            # 基本进度信息
            progress_info = (f"步骤: {step_counter}/{config.max_train_steps} ({progress:.2f}%) | "
                  f"平均奖励: {avg_recent_reward:.4f} | "
                  f"动作分布: [本地:{action_distribution[0]:.1f}%, 服务器:{action_distribution[1]:.1f}%, 服务器:{action_distribution[2]:.1f}%] | "
                  f"每秒步数: {steps_per_sec:.2f} | "
                  f"剩余时间: {format_time(eta_seconds)}")
            
            print(progress_info)
            
            # 详细模式下输出更多信息
            if config.verbose:
                print(f"  当前样本: {sample_idx}")
                print(f"  问题: {x[:50]}..." if len(x) > 50 else f"  问题: {x}")
                print(f"  本地回答: {y_L[:50]}..." if len(y_L) > 50 else f"  本地回答: {y_L}")
                print(f"  选择动作: {action} ({['本地', '服务器', '服务器'][action]})")
                print(f"  奖励: {reward:.4f}")
                
        # 达到更新时间间隔时，执行策略网络更新
        if step_counter % config.update_timestep == 0:
            print(f"\n开始第 {step_counter // config.update_timestep} 次策略更新...")
            update_start_time = time.time()
            ppo_agent.update()
            update_time = time.time() - update_start_time
            
            # 打印训练进度
            avg_reward = np.mean(episode_rewards[-config.update_timestep:])
            print(f"策略更新完成 | 用时: {update_time:.2f}秒 | 平均奖励: {avg_reward:.4f}")
            
            # 详细模式下输出更多信息
            if config.verbose:
                # 计算整体动作分布
                total_action_counts = [0, 0, 0]
                for action in ppo_agent.memory.actions[-config.update_timestep:]:
                    total_action_counts[action] += 1
                
                total_distribution = [count / sum(total_action_counts) * 100 for count in total_action_counts]
                print(f"  整体动作分布: [本地:{total_distribution[0]:.1f}%, 服务器:{total_distribution[1]:.1f}%, 服务器:{total_distribution[2]:.1f}%]")
                print(f"  已处理样本数: {step_counter}")
            
            # 保存模型检查点
            if step_counter % (config.update_timestep * 10) == 0:
                checkpoint_path = f"checkpoints/policy_step_{step_counter}.pth"
                torch.save({
                    'policy_state_dict': ppo_agent.policy.state_dict(),
                    'value_state_dict': ppo_agent.value.state_dict(),
                    'policy_optimizer': ppo_agent.policy_optimizer.state_dict(),
                    'value_optimizer': ppo_agent.value_optimizer.state_dict(),
                    'step': step_counter,
                    'avg_reward': avg_reward,
                    'timestamp': time.time()
                }, checkpoint_path)
                print(f"模型检查点已保存: {checkpoint_path}")
    
    # 训练结束，保存最终模型
    final_checkpoint_path = "checkpoints/policy_final.pth"
    torch.save({
        'policy_state_dict': ppo_agent.policy.state_dict(),
        'value_state_dict': ppo_agent.value.state_dict(),
        'policy_optimizer': ppo_agent.policy_optimizer.state_dict(),
        'value_optimizer': ppo_agent.value_optimizer.state_dict(),
        'step': step_counter,
        'avg_reward': np.mean(episode_rewards[-min(1000, len(episode_rewards)):]),
        'timestamp': time.time()
    }, final_checkpoint_path)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n策略网络训练完成!")
    print(f"总训练步数: {config.max_train_steps}")
    print(f"总训练时间: {format_time(total_time)}")
    print(f"平均每步时间: {total_time/config.max_train_steps:.4f}秒")
    print(f"最终模型已保存: {final_checkpoint_path}")
    print("=" * 50)


if __name__ == "__main__":
    # 创建保存检查点的目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # 初始化配置
    config = Config()
    
    # 开始训练
    train(config) 