#!/bin/bash

# P3Defer策略网络评估脚本
# 用于在GSM8K测试集上评估训练好的策略网络性能

# 默认测试配置
python test_results.py \
  --policy_path ./checkpoints/policy_final.pth \
  --test_data ./data/gsm8k/test.jsonl \
  --max_samples 50 \
  --output_file test_results.json

echo "==== 测试其他配置 ===="
echo "# 使用特定GPU"
echo "python test_results.py --policy_path ./checkpoints/policy_final.pth --gpu_ids 0,1"
echo ""
echo "# 测试更多样本"
echo "python test_results.py --policy_path ./checkpoints/policy_final.pth --max_samples 200"
echo ""
echo "# 使用不同的模型路径"
echo "python test_results.py --policy_path ./checkpoints/policy_step_20000.pth --local_model ./models/other-local-model --server_model ./models/other-server-model" 