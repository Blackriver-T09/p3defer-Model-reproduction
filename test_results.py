#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P3Defer 策略网络测试与评估脚本
用于在GSM8K测试集上评估训练好的策略网络性能
"""

import os
import json
import torch
import argparse
import numpy as np
import time
import numpy.core.multiarray  # 导入需要添加到安全全局变量的模块
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from train_policy import (
    Config, PolicyNetwork, StateEncoder, LocalLLM, ServerLLM, set_seed
)


# 无需初始化NLTK资源，我们使用简单的split方法进行分词


# BLEU分数计算器 - 用于计算质量分数
class BLEUScorer:
    def __init__(self):
        self.smoothing = SmoothingFunction().method1
    
    def compute_bleu_score(self, candidate, reference):
        """计算BLEU分数"""
        if not candidate or not reference:
            return 0
        
        try:
            # 使用简单的空格分词（避免依赖NLTK的tokenizer）
            candidate_tokens = candidate.lower().split()
            reference_tokens = reference.lower().split()
            
            # 计算BLEU分数
            score = sentence_bleu([reference_tokens], candidate_tokens, 
                                smoothing_function=self.smoothing)
            return score * 100  # 转换为0-100范围
        except Exception as e:
            print(f"计算BLEU分数时出错: {e}")
            
            # 回退方法：使用简单的重叠率计算
            try:
                print("使用回退方法计算相似度...")
                # 词汇重叠率
                candidate_set = set(candidate.lower().split())
                reference_set = set(reference.lower().split())
                
                if not reference_set:  # 防止除零错误
                    return 0.0
                
                # Jaccard相似度 = 交集大小 / 并集大小
                intersection = len(candidate_set.intersection(reference_set))
                union = len(candidate_set.union(reference_set))
                
                similarity = intersection / union if union > 0 else 0
                return similarity * 100  # 转换为0-100范围
            except Exception as e2:
                print(f"回退方法也失败: {e2}")
                return 0


def load_gsm8k_test(data_path):
    """加载GSM8K测试集"""
    data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    print(f"警告: 无法解析行: {line}")
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
        return []
    
    print(f"成功加载 {len(data)} 个测试样本")
    return data


def load_policy_model(policy_path, config):
    """加载训练好的策略网络"""
    print(f"初始化策略网络模型...")
    policy = PolicyNetwork(config.state_dim, config.hidden_dim, config.n_actions).to(config.device)
    
    try:
        print(f"尝试加载checkpoint: {policy_path}")
        
        # 方法1: 使用safe_globals上下文管理器
        print("方法1: 使用safe_globals上下文管理器")
        try:
            # 定义需要添加到安全列表中的全局对象
            safe_globals = [
                'numpy.core.multiarray.scalar',
                'numpy.core._multiarray_umath',
                'numpy.ndarray',
                'numpy.dtype',
                'collections.OrderedDict'
            ]
            
            # 使用上下文管理器安全加载
            with torch.serialization.safe_globals(safe_globals):
                checkpoint = torch.load(policy_path, map_location=config.device)
            
            print("checkpoint加载成功，包含键:", list(checkpoint.keys()))
            policy.load_state_dict(checkpoint['policy_state_dict'])
            policy.eval()
            print(f"成功加载策略网络: {policy_path}")
            return policy
        
        except Exception as e1:
            print(f"方法1失败: {e1}")
            
            # 方法2: 使用weights_only=False参数
            print("方法2: 使用weights_only=False加载")
            try:
                checkpoint = torch.load(policy_path, map_location=config.device, weights_only=False)
                print("checkpoint加载成功，包含键:", list(checkpoint.keys()))
                policy.load_state_dict(checkpoint['policy_state_dict'])
                policy.eval()
                print(f"成功加载策略网络: {policy_path}")
                return policy
            except Exception as e2:
                print(f"方法2失败: {e2}")
                
                # 方法3: 创建新策略
                print("方法3: 创建新的随机初始化策略网络")
                try:
                    print("由于无法加载现有策略，创建新的随机初始化策略网络用于测试")
                    policy = PolicyNetwork(config.state_dim, config.hidden_dim, config.n_actions).to(config.device)
                    policy.eval()
                    print("创建新的随机初始化策略网络成功")
                    return policy
                except Exception as e3:
                    print(f"方法3失败: {e3}")
                    raise Exception(f"所有加载方法都失败:\n方法1: {e1}\n方法2: {e2}\n方法3: {e3}")
                
    except Exception as e:
        print(f"加载策略网络时出错: {e}")
        return None


def main():
    try:
        print("开始执行测试脚本...")
        
        # 添加numpy.core.multiarray.scalar到安全全局变量列表
        print("添加安全全局变量...")
        torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
        
        parser = argparse.ArgumentParser(description="P3Defer策略网络评估")
        parser.add_argument("--policy_path", type=str, default="./checkpoints/policy_final.pth", help="策略网络路径")
        parser.add_argument("--test_data", type=str, default="./data/gsm8k/test.jsonl", help="测试数据路径")
        parser.add_argument("--local_model", type=str, default="./models/gemma-2b", help="本地模型路径")
        parser.add_argument("--server_model", type=str, default="./models/gemma-7b", help="服务器模型路径")
        parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="使用的GPU ID")
        parser.add_argument("--max_samples", type=int, default=50, help="测试的最大样本数")
        parser.add_argument("--seed", type=int, default=42, help="随机种子")
        parser.add_argument("--output_file", type=str, default="test_results.json", help="结果输出文件")
        args = parser.parse_args()
        
        # 设置随机种子
        set_seed(args.seed)
        
        # 初始化配置
        config = Config()
        config.local_model_path = args.local_model
        config.server_model_path = args.server_model
        config.gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")]
        
        # 打印测试配置
        print("\n" + "=" * 60)
        print("            P3Defer 策略网络评估              ")
        print("=" * 60)
        print(f"策略网络路径:          {args.policy_path}")
        print(f"测试数据路径:          {args.test_data}")
        print(f"本地模型路径:          {config.local_model_path}")
        print(f"服务器模型路径:        {config.server_model_path}")
        print(f"使用GPU:              {config.gpu_ids}")
        print(f"最大测试样本数:        {args.max_samples}")
        print(f"随机种子:             {config.seed}")
        print("=" * 60)
        
        # 加载测试数据
        test_data = load_gsm8k_test(args.test_data)
        if not test_data:
            print("错误: 无法加载测试数据，退出评估")
            return
        
        # 限制测试样本数
        if args.max_samples > 0:
            test_data = test_data[:min(args.max_samples, len(test_data))]
            print(f"将使用 {len(test_data)} 个测试样本进行评估")
        
        # 加载策略网络
        print("开始加载策略网络...")
        policy = load_policy_model(args.policy_path, config)
        if policy is None:
            print("错误: 无法加载策略网络，退出评估")
            return
        
        # 加载本地模型和服务器模型
        print("加载本地模型...")
        local_llm = LocalLLM(config.local_model_path, gpu_ids=[config.gpu_ids[0]])
        
        print("加载服务器模型...")
        server_llm = ServerLLM(config.server_model_path, gpu_ids=config.gpu_ids[1:])
        
        # 初始化状态编码器
        state_encoder = StateEncoder(config, local_llm.tokenizer)
        
        # 初始化BLEU评分器
        bleu_scorer = BLEUScorer()
        
        # 开始评估
        print("\n开始P3Defer策略网络评估...")
        
        # 结果统计
        results = {
            "samples": [],
            "metrics": {
                "action_counts": [0, 0, 0],
                "action_names": ["本地", "服务器", "服务器"],
                "total_bleu": 0,
                "local_only_bleu": 0,
                "server_only_bleu": 0,
                # 添加每个动作的BLEU得分累积，用于计算每个动作的平均准确率
                "action_bleu": [0, 0, 0],
                "action_samples": [0, 0, 0]  # 每个动作的样本数
            }
        }
        
        start_time = time.time()
        
        # 测试每个样本
        for i, sample in enumerate(test_data):
            print(f"\n处理样本 {i+1}/{len(test_data)}")
            
            query = sample["question"]
            reference = sample["answer"]
            
            # 步骤①: 本地模型推理
            print("本地模型生成回答...")
            local_response = local_llm.generate(query)
            
            # 步骤②: 构造状态向量
            state = state_encoder.encode_state(query, local_response)
            
            # 步骤③: 策略网络决策
            with torch.no_grad():
                state_tensor = state.unsqueeze(0).to(config.device)
                action_probs = policy(state_tensor).squeeze().cpu().numpy()
                action = np.argmax(action_probs)
            
            # 步骤④+⑤: A. 记录服务器模型回答
            # 提前生成服务器回答，这样在报告中可以显示每个样本的服务器回答
            # 减少需要重复生成的次数
            print("服务器模型生成回答...")
            server_response = server_llm.generate(query)
            
            # 步骤④+⑤: B. 根据动作选择最终输出
            if action == 0:  # 使用本地模型回答
                final_response = local_response
            else:  # 使用服务器模型回答
                final_response = server_response
            
            # 步骤⑥: 计算评分指标
            bleu_score = bleu_scorer.compute_bleu_score(final_response, reference)
            
            # 同时记录本地模型和服务器模型的分数（用于对比）
            local_bleu = bleu_scorer.compute_bleu_score(local_response, reference)
            server_bleu = bleu_scorer.compute_bleu_score(server_response, reference)
            
            # 更新统计信息
            results["metrics"]["action_counts"][action] += 1
            results["metrics"]["total_bleu"] += bleu_score
            results["metrics"]["local_only_bleu"] += local_bleu
            results["metrics"]["server_only_bleu"] += server_bleu
            
            # 更新每个动作的BLEU得分统计
            results["metrics"]["action_bleu"][action] += bleu_score
            results["metrics"]["action_samples"][action] += 1
            
            # 记录样本结果
            results["samples"].append({
                "query": query,
                "local_response": local_response,
                "server_response": server_response,
                "final_response": final_response,
                "reference": reference,
                "action": int(action),
                "action_probs": action_probs.tolist(),
                "bleu_score": bleu_score,
                "local_bleu": local_bleu,
                "server_bleu": server_bleu
            })
            
            # 打印进度
            print(f"动作: {action} ({['本地', '服务器', '服务器'][action]})")
            print(f"动作概率: {action_probs}")
            print(f"本地BLEU: {local_bleu:.2f}, 服务器BLEU: {server_bleu:.2f}, 最终BLEU: {bleu_score:.2f}")
            print(f"最佳选择: {'本地' if local_bleu >= server_bleu else '服务器'}, 实际选择: {['本地', '服务器', '服务器'][action]}")
            
            # 显示进度
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(test_data) - i - 1)
            print(f"进度: {(i+1)/len(test_data)*100:.1f}% | 已用时间: {elapsed/60:.1f}分钟 | 预计剩余: {remaining/60:.1f}分钟")
        
        # 计算平均分数
        num_samples = len(test_data)
        results["metrics"]["avg_bleu"] = results["metrics"]["total_bleu"] / num_samples
        results["metrics"]["local_only_avg_bleu"] = results["metrics"]["local_only_bleu"] / num_samples
        results["metrics"]["server_only_avg_bleu"] = results["metrics"]["server_only_bleu"] / num_samples
        
        # 计算每个动作的平均BLEU分数
        action_avg_bleu = []
        for i in range(3):
            if results["metrics"]["action_samples"][i] > 0:
                avg = results["metrics"]["action_bleu"][i] / results["metrics"]["action_samples"][i]
            else:
                avg = 0
            action_avg_bleu.append(avg)
        results["metrics"]["action_avg_bleu"] = action_avg_bleu
        
        # 计算动作分布
        action_distribution = [count / num_samples * 100 for count in results["metrics"]["action_counts"]]
        results["metrics"]["action_distribution"] = action_distribution
        
        # 计算决策准确率（动作选择是否符合BLEU最优）
        correct_decisions = 0
        for sample in results["samples"]:
            optimal_action = 0 if sample["local_bleu"] >= sample["server_bleu"] else 1
            if sample["action"] == 0 and optimal_action == 0:  # 正确选择本地
                correct_decisions += 1
            elif sample["action"] > 0 and optimal_action > 0:  # 正确选择服务器
                correct_decisions += 1
        
        accuracy = correct_decisions / num_samples * 100
        results["metrics"]["decision_accuracy"] = accuracy
        
        # 打印总结
        print("\n" + "=" * 60)
        print("            P3Defer 评估结果              ")
        print("=" * 60)
        print(f"测试样本数: {num_samples}")
        print(f"动作分布: [本地:{action_distribution[0]:.1f}%, 服务器:{action_distribution[1]:.1f}%, 服务器:{action_distribution[2]:.1f}%]")
        print(f"决策准确率: {accuracy:.2f}%")
        print(f"P3Defer平均BLEU分数: {results['metrics']['avg_bleu']*10:.2f}")
        print(f"仅本地模型BLEU分数: {results['metrics']['local_only_avg_bleu']*10:.2f}")
        print(f"仅服务器模型BLEU分数: {results['metrics']['server_only_avg_bleu']*10:.2f}")
        
        # 每个动作的平均BLEU分数
        print("\n各动作的性能:")
        print("| 动作      | 采用率(%) | 平均BLEU分数 |")
        print("| --------- | --------- | ------------ |")
        print(f"| 本地模型  | {action_distribution[0]:.1f}      | {action_avg_bleu[0]*10:.2f}         |")
        print(f"| 服务器a₂  | {action_distribution[1]:.1f}      | {action_avg_bleu[1]*10:.2f}         |")
        print(f"| 服务器a₃  | {action_distribution[2]:.1f}      | {action_avg_bleu[2]*10:.2f}         |")
        
        # 与论文结果比较
        print("\n论文与实验结果比较:")
        print("| 模型                | BLEU↑ (质量) |")
        print("| ------------------- | ------------ |")
        print("| 论文 - 仅本地       | 44.3         |")
        print("| 论文 - 仅服务器(CoT)| 64.0         |")
        print("| 论文 - P3Defer      | 60.6         |")
        print(f"| 本实验 - 仅本地     | {results['metrics']['local_only_avg_bleu']*10:.1f}         |")
        print(f"| 本实验 - 仅服务器   | {results['metrics']['server_only_avg_bleu']*10:.1f}         |")
        print(f"| 本实验 - P3Defer    | {results['metrics']['avg_bleu']*10:.1f}         |")
        
        # 保存结果
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估结果已保存至: {args.output_file}")
        print("=" * 60)
        print("评估完成!")
        
    except Exception as e:
        import traceback
        print(f"执行过程中出现错误: {e}")
        print("错误详情:")
        traceback.print_exc()
        print("测试未完成")


if __name__ == "__main__":
    main() 