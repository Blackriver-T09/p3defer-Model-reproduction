示例使用方法：
# 基本训练
python run_training.py --max_train_steps 10000 --log_interval 50
# 详细输出模式
python run_training.py --max_train_steps 10000 --log_interval 50 --verbose
# 调整GPU配置
python run_training.py --max_train_steps 10000 --gpu_ids 0,1
# 快速训练模式（推荐用于加速训练）
python run_training.py --max_train_steps 10000 --log_interval 50 --fast_mode --server_sample_rate 0.2 --precompute 200 --max_tokens 64
# 最快训练模式（最大程度减少模型生成）
python run_training.py --max_train_steps 10000 --log_interval 50 --fast_mode --server_sample_rate 0.1 --precompute 500 --max_tokens 32
