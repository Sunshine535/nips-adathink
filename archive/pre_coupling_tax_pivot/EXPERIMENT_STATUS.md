# 实验运行状态

## 正在运行的实验

### 27B完整测试 (GSM8K)
- **状态**: 运行中
- **进程**: PID 2648
- **内存**: 24GB
- **位置**: `/workspace/nips-adathink/results/fulltest_27b/`
- **日志**: `gsm8k_27b_restart.log`
- **预计样本**: 1,319
- **预计时间**: 2-4小时

### DeepSeek-R1-8B (MATH500)
- **状态**: 运行中
- **进程**: PID 1470
- **位置**: `/workspace/nips-adathink/results/deepseek/`

## 已完成实验

### 8B完整测试 (GSM8K)
- ✅ 1,319样本
- ✅ Adaptive: 52.77%
- ✅ Fixed-512: 65.20%
- 文件: `per_sample_gsm8k_Qwen3_8B_20260324_120316.csv`

## 环境修复

✅ 已将HuggingFace缓存从`/root/.cache`迁移到`/workspace/.cache`
✅ 使用`HF_HOME=/workspace/.cache/huggingface`环境变量
✅ 解决磁盘空间问题

## 下一步

等待27B实验完成后：
1. 下载结果CSV
2. 分析27B vs 8B性能对比
3. 完成最终论文数据
