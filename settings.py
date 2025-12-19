task_name = 'Test'  # 实验名称前缀，用于生成结果文件名
city = 'PHO'  # 选择数据集城市：PHO / NYC / SIN
gpuId = "cuda:0"  # 训练设备，设为 "cpu" 可强制用 CPU
seed = 2023  # 全局随机种子
batch_size = 1  # 训练 batch 大小（序列样本数）
log_steps = 2  # 前多少个 batch 打印详细日志

# 数据增强 / 预处理相关
enable_random_mask = True  # 是否对长序列做随机 mask
mask_prop = 0.1  # mask 比例（相对序列长度）
enable_enhance_user = True  # 是否启用 user embedding 增强

# 原版 SSL（与新对比学习互斥，关闭 use_contrastive 时才使用）
enable_ssl = True  # 是否启用原始 SSL 模块
enable_distance_sample = False  # SSL 负样本是否按最远距离采样
neg_sample_count = 5  # SSL 负样本数量
neg_weight = 1  # SSL loss 权重

# CALPRec 模块开关
use_gate = False  # 上下文门控开关（True 时使用 context gate）
use_contrastive = False  # 对比学习开关（True 切换到 InfoNCE）
neg_strategy = "random"  # 对比学习负样本策略：random / hard
hard_k = 5  # hard negative 选取的 top-k
tau = 0.2  # InfoNCE 温度
use_aux_cat = False  # 辅助类别预测开关
lambda_cl = 1.0  # 对比学习 loss 权重
lambda_cat = 1.0  # 辅助类别 loss 权重

# 动态天数窗口
enable_dynamic_day_length = False  # True 时使用动态天窗口数据与逻辑
sample_day_length = 14  # 动态模式下保留的最近天数范围 [3,14]

# 训练超参
lr = 1e-4  # 学习率
epoch = 25  # 训练轮数
if city == 'SIN':
    embed_size = 60  # 单特征 embedding 维度
    run_times = 3  # 重复运行次数（主程序中循环）
elif city == 'NYC':
    embed_size = 40
    run_times = 3
elif city == 'PHO':
    embed_size = 60
    run_times = 5

# 结果文件名自动拼接
output_file_name = f'{task_name} {city}' + "_epoch" + str(epoch)
if enable_dynamic_day_length:
    output_file_name = output_file_name + f"_DynamicDay{sample_day_length}"
else:
    output_file_name = output_file_name + "_StaticDay7"
if enable_random_mask:
    output_file_name = output_file_name + "_" + "Mask"
else:
    output_file_name = output_file_name + "_" + "NoMask"
if enable_enhance_user:
    output_file_name = output_file_name + "_" + "Enhance"
else:
    output_file_name = output_file_name + "_" + "NoEnhance"
if enable_ssl:
    if enable_distance_sample:
        output_file_name = output_file_name + "_" + "SSL" + "_" + "DistanceNegCount" + str(neg_sample_count)
    else:
        output_file_name = output_file_name + "_" + "SSL" + "_" + "NegCount" + str(neg_sample_count)
else:
    output_file_name = output_file_name + "_" + "NoSSL"
output_file_name = output_file_name + '_embeddingSize' + str(embed_size)
if use_gate:
    output_file_name = output_file_name + "_Gate"
if use_contrastive:
    output_file_name = output_file_name + "_CL-" + neg_strategy + f"-k{hard_k}-tau{tau}"
if use_aux_cat:
    output_file_name = output_file_name + f"_AuxCat-l{lambda_cat}"
