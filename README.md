# P3Defer: Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning

本项目是论文《Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning》的实现，实现了一个三分支智能级联系统，结合了隐私保护机制、链式思维(CoT)和强化学习(PPO)策略网络。

## 项目架构

系统主要包括以下核心组件：

1. **本地模型Φ(L)**：轻量级LLM（Gemma-2B），处理用户的初步请求
2. **服务器模型Φ(S)**：更强大的LLM（Gemma-7B），处理复杂或需要高质量回答的请求
3. **策略网络π_θ**：基于PPO训练的决策网络，决定以下动作之一：
   - a1: 直接返回本地模型的回答
   - a2: 转发原始问题到服务器模型
   - a3: 转发原始问题到服务器模型（简化版，不使用掩码）
4. **私有记忆模块M**：（可选）负责检测和掩码敏感信息

## 性能优化

为提高训练和推理性能，本项目进行了以下优化：

1. **使用本地模型替代API**：在训练过程中直接加载本地模型，而不是通过API调用，大幅提高训练速度
2. **多GPU加速**：
   - 使用4张GPU并行训练
   - GPU 0用于本地模型(Gemma-2B)
   - GPU 1-3用于服务器模型(Gemma-7B)
3. **简化私有记忆模块**：通过配置可以选择是否使用掩码处理，以提高精度

## 项目结构

```
.
├── train_policy.py      # 策略网络核心实现
├── run_training.py      # 训练脚本
├── inference.py         # 推理脚本
├── test_policy.py       # 组件测试脚本
├── example.py           # 示例演示脚本
├── checkpoints/         # 模型检查点保存目录
├── data/                # 训练数据
│   └── gsm8k/           # GSM8K数据集
├── models/              # 预训练模型
│   ├── gemma-2b/        # 本地Gemma-2B模型
│   └── gemma-7b/        # 服务器Gemma-7B模型
└── server/              # 服务器模型API
    └── cloud_llm_server.py  # Flask服务器脚本（可选）
```

## 安装依赖

```bash
pip install torch transformers flask requests Levenshtein
```

## 使用方法

### 1. 启动服务器模型API（可选）

如果需要使用API模式，可以启动服务器：

```bash
cd server
python cloud_llm_server.py
```

### 2. 训练策略网络

```bash
python run_training.py --batch_size 16 --learning_rate 3e-4 --max_train_steps 10000
```

参数说明：
- `--batch_size`: 批处理大小
- `--learning_rate`: 学习率
- `--ppo_epochs`: 每次更新的PPO轮数
- `--update_timestep`: 策略网络更新间隔
- `--max_train_steps`: 最大训练步数
- `--local_model`: 本地模型路径
- `--server_model`: 服务器模型路径
- `--server_api`: 服务器API地址
- `--use_api`: 是否使用API模式（默认为False，直接加载本地模型）
- `--privacy_weight`: 隐私奖励权重λ
- `--seed`: 随机种子
- `--checkpoint_dir`: 检查点保存目录
- `--data_path`: 训练数据路径
- `--gpu_ids`: 使用的GPU ID列表，用逗号分隔，例如"0,1,2,3"

### 3. 推理

#### 单个查询

```bash
python inference.py --policy_path checkpoints/policy_final.pth --input "What is 2+2?"
```

#### 批量查询

```bash
python inference.py --policy_path checkpoints/policy_final.pth --input_file queries.txt --output_file results.jsonl
```

参数说明：
- `--policy_path`: 策略网络权重路径（必需）
- `--local_model`: 本地模型路径
- `--server_model`: 服务器模型路径
- `--server_api`: 服务器API地址
- `--use_api`: 是否使用API调用服务器模型
- `--input`: 输入查询
- `--input_file`: 输入查询文件路径
- `--output_file`: 输出结果文件路径
- `--privacy_weight`: 隐私奖励权重λ
- `--seed`: 随机种子
- `--gpu_ids`: 使用的GPU ID列表，例如"0,1,2,3"

### 4. 运行示例演示

```bash
python example.py
```

### 5. 测试组件

```bash
python test_policy.py
```

## 核心流程

1. 用户输入查询 `x`
2. 本地模型生成初步回答 `y_L`
3. 状态编码器生成状态向量 `s_t = [e_p^t, e_q^t]`
4. 策略网络根据状态向量选择动作 `a_t`
5. 根据选择的动作执行对应操作：
   - 若选择 `a1`，直接返回本地回答 `y_L`
   - 若选择 `a2`或`a3`，转发原始查询 `x` 到服务器，返回 `y_S`

## 引用

```
@article{p3defer2024,
  title={Privacy-preserved LLM Cascade via CoT-enhanced Policy Learning},
  author={Author},
  journal={arXiv preprint},
  year={2024}
}
``` 