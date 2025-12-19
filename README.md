# CLSPRec

This is the pytorch implementation of paper "CLSPRec: Contrastive Learning of Long and Short-term Preferences for Next
POI Recommendation"

![model](model.png)

## Installation

```
pip install -r requirements.txt
```

## Valid Requirements

```
torch==2.0.1
numpy==1.24.3
pandas==2.0.2
Pillow==9.4.0
python-dateutil==2.8.2
pytz==2023.3
six==1.16.0
torchvision==0.15.2
typing_extensions==4.5.0
```

## Train

- Modify the configuration in settings.py

- Train and evaluate the model using python `main.py`.

- The training and evaluation results will be stored in `result` folder.

## 部署与运行（离线服务器详细步骤）

以下步骤假设服务器无法访问外网，请提前把代码仓库和所需的依赖包（如 PyTorch 对应 CUDA 版本的 whl 文件）拷贝到服务器。

1. 准备 Python 环境  
   - 推荐 Python 3.8–3.10。  
   - 创建虚拟环境：`python -m venv .venv`；激活：Linux `source .venv/bin/activate`，Windows `.\.venv\Scripts\activate`。  
   - 安装依赖：`pip install -r requirements.txt`。如需 GPU，请用与你的 CUDA 版本匹配的 `torch/torchvision` whl 包替换安装。

2. 数据放置  
   - 原始文件放在 `raw_data/`，命名示例：`PHO_checkin_with_active_regionId.csv`、`NYC_checkin_with_active_regionId.csv`、`SIN_checkin_with_active_regionId.csv`。  
   - 如使用距离负采样，还需对应的 `*_farthest_POIs.csv` 和 `*_poi_mapping.csv`（包含纬度经度）。

3. 生成处理数据  
   - 静态 7 天窗口（默认）：直接运行 `python data_preprocessor.py`，输出到 `processed_data/original/`。  
   - 动态窗口：在运行前设置 `settings.enable_dynamic_day_length = True`，可调整 `sample_day_length`（默认 14），然后运行 `python data_preprocessor.py`，输出到 `processed_data/dynamic_day_length/`。两套数据可共存。

4. 配置开关（settings.py）关键项  
   - `city`: 选择 `PHO/NYC/SIN`。  
   - 模块开关：`use_gate`（上下文门控）、`use_contrastive`（对比学习，`neg_strategy` 可选 `random/hard`，`hard_k`、`tau` 调参）、`use_aux_cat`（类别辅助任务）。默认均为 False，行为与原始 CLSPRec 一致。  
   - 其他：`batch_size`、`epoch`、`lr`、`seed` 等可按需调整。

5. 运行训练  
   - 进入仓库根目录，激活虚拟环境后执行：`python main.py`。  
   - 程序会从对应的 `processed_data/original` 或 `processed_data/dynamic_day_length` 读取数据（取决于 `enable_dynamic_day_length`），并将模型、日志写入 `results/`。  
   - 日志中会输出 loss、硬负样本均值、门控 α 统计等（取决于开关）。

6. 常见检查点  
   - 如果报 “找不到数据文件”，确认 `settings.enable_dynamic_day_length` 与已生成的数据目录一致。  
   - CUDA 可用性：启动时会根据 `settings.gpuId` 选择设备，如需 CPU 可将其设为 `"cpu"`。  
   - 若需重新生成数据，先删除或备份 `processed_data/*`，再按第 3 步运行。

## Cite Our Paper

### CIKM2023


    @inproceedings{CLSPRec2023,
        title = {{CLSPRec}: Contrastive Learning of Long and Short-term Preferences for Next {POI} Recommendation},
        doi = {10.1145/3583780.3614813},
        shorttitle = {{CLSPRec}},
        pages = {473--482},
        booktitle = {Proceedings of the 32nd {ACM} International Conference on Information and Knowledge Management, {CIKM} 2023, Birmingham, United Kingdom, October 21-25, 2023},
        publisher = {{ACM}},
        author = {Duan, Chenghua and Fan, Wei and Zhou, Wei and Liu, Hu and Wen, Junhao},
        editor = {Frommholz, Ingo and Hopfgartner, Frank and Lee, Mark and Oakes, Michael and Lalmas, Mounia and Zhang, Min and Santos, Rodrygo L. T.},
        date = {2023}
    }
