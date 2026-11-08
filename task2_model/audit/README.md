# audit/ — 排行榜方案（0.49893）的全部要素

这个目录是最终提交 `submission_ens.csv` 的完整来源。目标：**从公开数据 + 本目录脚本，复现排行榜 0.49893 的提交文件**。

目录内容：

| 文件 | 作用 |
|---|---|
| `build_text.py` | `train.csv` / `test.csv` → `parquet(id, text)`，`text = title + ". " + comment` |
| `recover_clean_labels.py` | 用公开 Grocery 源数据恢复训练集的**干净标签** `cy`（覆盖率 92.50%） |
| `ft3.py` | 微调编码器（编码器 + 线性回归头，MSE 对干净标签），输出验证/测试预测 |
| `build_submission.py` | 加权集成 + isotonic 校准，写出 `id,rating` 提交文件（支持 `--apply` 参数复现） |
| `preds/ftv_<tag>.npy` | 各模型在验证切片（`id % 40 == 0`）上的预测，共 192,117 行 |
| `preds/ftp_<tag>.npy` | 各模型在 10,000 行测试集上的预测 |
| `preds/ftvids_s1.npy` | 验证切片的 id 数组（各 tag 共用） |
| `preds/clean_y.npz` | 干净标签：`cy`（干净 1–5）、`matched`（是否匹配到源）、`ag`（交付标签） |
| `preds/ensemble_params.json` | 集成权重 + isotonic 断点，用于逐字节复现最佳提交 |
| `SOURCE_AND_LABEL_NOISE.md` | 标签噪声的完整证据链（在哪一层、如何恢复、下界推导） |
| `FINETUNE_RESULT.md` | 成绩轨迹、模型阵容、算力实测、踩坑记录 |

> 模型权重（`ft3_*.pt`，90–265 MB）**未提交**：GitHub 单文件 100 MB 限制。预测数组（每个 <1 MB）足以复现提交；重训命令见下文。

## 0. 数据获取

比赛数据（Kaggle，需要已加入该比赛）：

```python
import kagglehub
path = kagglehub.competition_download('comp-5434-20261-project-task-2')
print("Path to competition files:", path)
```

把 `train.csv` / `test.csv` / `sampleSubmission.csv` 放到 `comp5434project/Dataset2_metadata/`（该目录已在 `.gitignore` 中）。

源数据（Amazon Reviews 2023, Grocery_and_Gourmet_Food，恢复干净标签用，5.97 GB）：

```bash
curl -L -o /tmp/grocery.jsonl \
  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Grocery_and_Gourmet_Food.jsonl
```

## 1. 构造文本 parquet（约 10 秒 / 617 MB）

```bash
python build_text.py --src ../../Dataset2_metadata/train.csv --dst /content/train_text.parquet
python build_text.py --src ../../Dataset2_metadata/test.csv  --dst /content/test_text.parquet
```

## 2. 恢复干净标签（约 3 分钟）

```bash
python recover_clean_labels.py    # 读 /tmp/grocery.jsonl + train.csv，写 /tmp/clean_y.npz
```

预期输出（可核对）：

```
coverage 92.50%  agreement 80.0103%  implied p=0.1999
full-data 5-star share 0.5534 ; raw5 implied 0.6713
```

`preds/clean_y.npz` 就是这一步的产物（23 MB），仓库里已经放了一份，跳过第 0–2 步也能直接做第 4 步。

## 3. 微调四个编码器（Colab GPU）

把 `train_text.parquet`、`test_text.parquet`、`clean_y.npz` 传到 `/content/` 后：

```bash
# s1  MiniLM    2 epoch  maxlen 96   bs 512   （L4 约 2.2 h）
python ft3.py --epochs 2 --bs 512  --maxlen 96  --seed 1 --tag s1
# s2  同底座不同 seed / 截断（G4 约 37 min）
python ft3.py --epochs 2 --bs 1024 --maxlen 128 --seed 2 --tag s2
# d1  distilbert-base-uncased   （G4 约 65 min）
python ft3.py --model distilbert-base-uncased --epochs 2 --bs 768 --maxlen 128 --seed 3 --tag d1
# b1  bge-small-en-v1.5（单模型最强，0.5328）
python ft3.py --model BAAI/bge-small-en-v1.5  --epochs 2 --bs 768 --maxlen 128 --seed 5 --tag b1
```

每个 tag 产出 `ftv_<tag>.npy`（验证切片预测）、`ftvids_<tag>.npy`（id）、`ftp_<tag>.npy`（测试预测），下载到本目录 `preds/` 即可。

训练细节：AdamW，lr 2e-5 余弦调度 + 5% warmup，`clip_grad_norm 1.0`，只训 `id % 40 != 0` 且有干净标签的 693 万行。

## 4. 生成提交（两种模式，结果逐字节一致）

```bash
# (a) 只用仓库里的预测 + 参数复现（秒级）
python build_submission.py --apply --tags s1 s2 d1 b1 \
    --params-in preds/ensemble_params.json --out submission_ens.csv

# (b) 重新搜索权重并拟合 isotonic（给出 heldout 评估）
python build_submission.py --tags s1 s2 d1 b1 \
    --out submission_ens.csv --params-out preds/ensemble_params.json
```

输出：

```
best weights {'s1': 0.1774, 's2': 0.0051, 'd1': 0.2558, 'b1': 0.5618} validation rmse 0.5276
isotonic held-out rmse 0.5245 (fit 88808 / eval 88799)
test preds n=10000 mean 4.240 sd 1.246
```

`(a)` 与仓库里的 `submission_ens.csv` 用 `cmp` 比较无差异 —— 这是本方案的验收点。

## 5. 其它脚本（前期实验，结论已被采纳或否决）

| 文件 | 结论 |
|---|---|
| `cleaning_ablation.py` | 截尾 / EM 软标签 / 硬重标注全部中性或有害（`SOURCE_AND_LABEL_NOISE.md` §8） |
| `train_sgd.py` / `train_sgd_clean.py` / `train_sgd_dense.py` | 哈希词袋线性基线 1.2029；在干净标签上重训更差；加数值特征更差 |
| `eval_clean.py` / `calibrate_submission.py` | 校准评估：仿射 0.7515、isotonic 0.7043（词袋模型） |
| `entity_bias_full.py` | 实体偏置特征最多只值 0.05 RMSE，且测试侧没有 id |
| `encode_emb.py` | 冻结句向量路线（CPU 版：73 docs/s ⇒ 全量 29 h，已放弃，改用 Colab） |
| `check_no_leakage.py` | 训练/验证/测试划分的泄漏审计 |
