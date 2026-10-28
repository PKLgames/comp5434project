# COMP5434 Task 2 — V2 特征改进计划

> 状态：计划已评审通过，**已按本计划实施完毕**（2026-10-28）。G2 门槛结果为负，
> V2 未采纳，总表保持不变。实施细节与实测数字见文末「12. 执行记录」。

---

## 0. 当前状态

| 项 | 状态 |
| --- | --- |
| 计划评审 | 已通过 |
| 本机 Python 环境 | **已按要求撤回**（`uv`、`uv-python`、`uv-cache`、`.venv` 均已删除），本机当前没有可用 Python |
| 代码改动 | 无。`common.py` / `dataset.py` / `model.py` / `preprocess.py` / `train.py` / `predict.py` 均为原始状态 |
| 现有模型产物 | 未触碰。`artifacts/`、`runs/full3`、`runs/full_tmlp`、`runs/ablate_*`、`runs/ensemble*`、`results/validation_metrics.json` 全部保持原样 |
| 工作区已有改动 | 根 `README.md` 中 `新token化` → `tokenized` 的路径修正已存在于工作区（非本次操作产生），与本计划第 6 节文档项一致，保留 |

**执行前提：** 所有命令都需要一个可用的 Python 环境（numpy / scipy / pyarrow / scikit-learn / torch）。由于本机已删除该环境，实施与验证必须在具备 Python 的机器上（最终为 GPU 机器）进行，或按第 8 节重建环境。

---

## 1. 核查结论（计划的事实依据）

| 结论 | 证据 |
| --- | --- |
| `Task2_Token_Dataset/tokenized/` 与 `Task2_OriginData/output/` 的**特征 parquet 逐字节相同** | part 文件 UUID 一致；shard 0 的 SHA-256 相同（`2A5118A0…`）；总字节均为 **1,726,687,142**；`text_vectorization_summary.json` 哈希一致（`589E13D1…`） |
| 目录大小 2.02 GB vs 2.16 GB 的差异**不来自特征** | 差异只来自仅存在于 `Task2_OriginData` 的辅助表：`text_models/`、`entity_statistics/`、`split_assignments/` |
| 新数据集**没有新增任何列** | Parquet footer 列名：`id, rating` + 11 个基础数值 + 9 个实体统计 + `title_tfidf` + `comment_tfidf` |
| 当前模型已使用**全部**这些列 | `common.py` 使用全部 20 个数值列，派生 6 个，再用两个稀疏列；共 26 维 dense + 2 个哈希 TF-IDF（65,536 / 262,144 维，L2 归一化） |
| 当前成绩（待超越） | 单模型 **1.1573**，两模型堆叠集成 **1.1543**（验证集 1,526,196 行） |
| 过期路径残留 | `task2_model/README.md`、`task2_model/preprocess.py` 的文档字符串中仍有 `新token化`（根 `README.md` 已在工作区修正） |

**关键推论：** 所谓"新数据集"不携带新信息，因此 V2 的提升只能来自**在现有行内部派生的特征**，而不是新的上游数据。计划据此设计，并加入数据集指纹校验，以便将来真正不同的数据投放能被自动识别。

---

## 2. 目标与验收标准

**目标：** 在 `tokenized/` 之上增加一层可选的 V2 特征与 V2 训练 / 评估通路，与现有模型并行训练，**只有确实优于 1.1543 时才采纳**。

| 编号 | 标准 |
| --- | --- |
| **G1（硬性）** | 现有模型可证明未被改动：`artifacts/`、`runs/*/best.pt`、`results/validation_metrics.json` 内容不变，脚本默认行为不变 |
| **G2（准入门槛）** | 在与现有方法学完全一致、随机种子匹配的 1.52 M 行 / 1 epoch 子集消融上，V2 特征必须优于 V1；否则停止并汇报数字 |
| **G3（目标）** | 全量数据下 V2 堆叠集成 `val RMSE ≤ 1.1533`，且 ΔRMSE 的配对 bootstrap 95% 置信区间不含 0 |
| **G4** | **负结果同样算交付**：用数据记录清楚，总表保持不变 |

---

## 3. 设计：V2 到底增加什么

唯一可用的原料是两个 L2 归一化的哈希 TF-IDF 行。当前模型使用了它们的**数值**（EmbeddingBag）以及每个 token 的一个线性标量，但**从未利用其形态**。形态信息计算成本几乎为零，且模型无法从 rank-256 投影中还原。

### 3.1 V2 核心特征块（7 个，默认启用）

全部由已有 CSR 数组计算，不需要新数据、不使用标签、不做跨行统计。

| 特征 | 定义 | 动机 |
| --- | --- | --- |
| `title_nnz` / `comment_nnz` | `log1p(data > 0 的条目数)` | minDocFreq 过滤后真正保留的**唯一**词数，与 `*_word_count`（统计全部 token）并不等价 |
| `title_weight_sum` / `comment_weight_sum` | `log1p(Σ data)` | 分散度：单一主导词时 ≈1，内容分散时很小 |
| `title_weight_max` / `comment_weight_max` | `max(data)`，取值 (0,1] | 最强信息词的主导程度 |
| `title_comment_cosine` | 在共同哈希下标上的 `t · c` | 跨列交互；两个 EmbeddingBag 是分离的、wide 通路是相加的，模型自己无法构造该特征。任一侧为空时置 0.0 |

### 3.2 V2 可选交互块（4 个，默认关闭）

`user_unseen+product_unseen+parent_unseen`、`votes_log × log1p(comment_char_count)`、`purchased × log1p(comment_char_count)`、`review_year − 2023`。作为独立消融单独度量，绝不悄悄混入核心块。

### 3.3 明确的非目标

- 不做按 token 的平均评分编码：会泄漏，且与 wide 线性通路重复。
- 不对 MSE 做类别重加权：RMSE 最优解就是条件均值，重加权反而有害。
- 不使用 `id`：看似随机字符串，且有泄漏风险。

---

## 4. 架构：零拷贝叠加层

不另写第二份 2.3 GB 缓存，V2 作为现有只读 `artifacts/` 的**叠加层**：

```
artifacts/（保持不变，只读）                artifacts_v2/（新增，仅 KB）
  vocab_*.npy, {train,valid}/shard_*.npz     meta.json   <- v1 meta + base_dir + 7 列
                                             stats.json  <- 单遍训练集拟合的 mean/std
        │                                         │
        └──────────► ShardBatcher ◄───────────────┘
              读取基础 shard，按批次由 t_*/c_* 数组
              计算 7 个 V2 列并追加，再对 33 维统一标准化
```

这 7 个列都是基础 shard 内已有数组的纯函数，因此每批次开销仅几毫秒（相对约 8 分钟 / epoch），也无需第二份缓存。

### 4.1 `meta.json` 结构（version 2，纯增量）

```json
{
  "version": 2,
  "base_dir": "C:/coding/comp5434project/artifacts",
  "dense_columns": ["…26 个 v1 列名…", "title_nnz", "comment_nnz", "…"],
  "dense_stats": { "rows": 6104774, "mean": ["…33…"], "std": ["…33…"] },
  "v2": {
    "feature_set": "v2",
    "columns": ["…7…"],
    "fit_split": "train",
    "base_fingerprint": "…sha256…"
  },
  "splits": { },
  "vocab": { },
  "train_rating_mean": 0.0,
  "train_rating_distribution": []
}
```

v1 的 `meta.json` 一个字段都不改，依旧有效；`base_dir` 与新增列只是"不存在"而已。

---

## 5. 向后兼容约定（"不影响当前模型"）

1. **不写入** `artifacts/`、`runs/full3`、`runs/full_tmlp`、`runs/ablate_*`、`runs/ensemble*`、`task2_model/results/`。所有新产物写入 `artifacts_v2/` 与 `runs/v2_*`。
2. `common.DENSE_COLS` 严格保持现有 26 项与顺序。V2 列名放入新增的 `V2_TEXT_SHAPE_COLS`；`build_dense_matrix(table, feature_set="v1")` 默认行为与输出完全不变。
3. **不修改 `model.py`**：dense 塔本就只依赖 `n_dense`，state_dict 键不变，`runs/full3/best.pt` 与 `runs/full_tmlp/best.pt` 继续可加载。
4. `preprocess.py` / `train.py` / `predict.py` 新增开关（`--feature-set`、`--interactions`）默认等于当前行为；README 中的命令仍可复现旧结果。
5. 唯一一处共享代码改动是有意为之且行为中性：`dataset.py` 改为从 `meta["dense_columns"]` 读取列名，而非 import `DENSE_COLS`。现有每个 `meta.json` 都已存有同样内容的该列表，故 v1 行为不变，而 V2 缓存得以自描述。
6. `predict.py` 第 127 行目前用 `os.path.join(args.data, "valid", …)` 直接重开 shard，绕过了 loader。改为使用 `loader.shard_dir` 以支持叠加层（对 v1 结果无影响）。

---

## 6. 实施清单（按文件）

### 6.1 新增 `task2_model/features_v2.py`

纯函数、无额外依赖：

- `text_shape_block(t_indptr, t_indices, t_data, c_indptr, c_indices, c_data) -> np.ndarray[rows,7]`，基于 offset 的 bincount 向量化实现，处理空行（`nnz = 0`、`cosine = 0.0`），并**排除 `data == 0` 的词表未命中条目**。
- `interaction_block(dense_v1) -> np.ndarray[rows,4]`。
- `V2_TEXT_SHAPE_COLS`、`V2_INTERACTION_COLS`、`all_columns(interactions: bool)`。

### 6.2 `task2_model/common.py`

增加两组列注册表；扩展 `build_dense_matrix(..., feature_set)`；新增 `dense_schema(feature_set, interactions)`；在 `FEATURE_GROUPS` 中注册 `text_shape` 与 `interactions`，使现有消融机制（`select_dense_columns`）无需改动即可复用。默认参数保持 v1。

### 6.3 `task2_model/preprocess.py`

新增 `--feature-set {v1,v2}`（默认 `v1`）与 `--interactions`。v2 模式下：

1. 要求 `--reuse` 指向 v1 缓存；
2. 单遍扫描训练 shard，拟合新列的 mean/std（仅训练集，避免验证泄漏）；
3. 写出轻量的 `artifacts_v2/meta.json` 与 `stats.json`；
4. 记录 `base_fingerprint`。

同时修正文档字符串中过期的 `新token化` 路径。

### 6.4 `task2_model/dataset.py`

`ShardBatcher` 从 meta 读取 `feature_set` / `interactions`；按 `meta.get("base_dir", artifacts)` 解析 `shard_dir`；在既有标准化 + 截断之前追加计算块。`steps_per_epoch`、`num_rows` 与批处理语义不变。

### 6.5 `task2_model/train.py`

透传 feature set；将 `dense_schema` 写入 checkpoint（仅新增键，旧 loader 忽略）。消融模式增加 `text_shape` / `interactions`；默认值不变。

### 6.6 `task2_model/predict.py`

改用 `loader.shard_dir` 读取 `y`，以支持叠加层；校验 `--data` 的 `dense_schema` 与 checkpoint 一致，不一致时给出明确报错。

### 6.7 新增 `task2_model/compare.py`

读取两份 `val_preds.npz`，输出 ΔRMSE / ΔMAE、分评分差值，以及 5,000 次配对 bootstrap 置信区间，写 `comparison.json`。这是 G3 客观判定的依据。

### 6.8 新增 `task2_model/stack.py`

把 README 第 6.4 节中手工拟合的权重推广到 N 个 checkpoint：5 折 out-of-fold 拟合权重 + 截距，沿用现有 OOF 方法学。

### 6.9 测试

- `tests/test_v2_features.py`：用手算期望值校验 nnz / sum / max / cosine；覆盖空行、单 token 边界，以及 `data == 0` 必须被排除的情形。
- `tests/test_backcompat.py`：校验 `common.DENSE_COLS` 等于 26 项 v1 元组；用真实 `artifacts/meta.json` 构造 batcher 并断言 dense 维度为 26；加载 `runs/full3/best.pt` 跑一次前向；断言 V2 通路从不改动 `artifacts/`（mtime 检查）。
- `tests/test_smoke_v2.py`：`--limit-train 1 --limit-valid 1` 构建叠加层并跑一步训练，断言 loss 有限且 `dense.shape[1] == 33`。

### 6.10 文档

- 修正 `task2_model/README.md` 与 `preprocess.py` 中的 `新token化`（根 `README.md` 已修正）。
- 新增"数据集来源"小节，记录逐字节相同这一发现及其证据。
- 新增 V2 章节与 V2 消融 / 结果表；**数字只在实测后填写**。

---

## 7. 验证方式

1. **数值回归（改动前必须先做）**：`predict.py --data artifacts --checkpoint runs/full3/best.pt --max-shards 1` 记录 RMSE；改动后在同一 shard 上必须得到相同值。
2. 三个测试文件（6.9）全部通过。
3. `pip check` 无依赖冲突；三个主脚本的 `--help` 与旧参数完全兼容。
4. `git diff` 中不出现 `artifacts/`、`runs/` 下已存在文件的任何改动。

---

## 8. 执行顺序

### 第 0 步 — 准备 Python 环境（本机环境已撤回，需在目标机器上重建）

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r task2_model/requirements.txt
# GPU 机器：先装 CUDA 版 torch，再装其余依赖
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

校验：`python -c "import torch, numpy, scipy, pyarrow, sklearn"` 且 GPU 机器上 `torch.cuda.is_available()` 为 `True`。

### 第 1 步 — 数值回归基线

在任何代码改动之前，先跑第 7 节第 1 项的 `--max-shards 1` 预测并记录 RMSE。

### 第 2 步 — CPU 迭代（分钟级）

构建 V2 叠加层，跑 smoke 测试，然后执行 **G2 门槛**：在 16 个训练 / 2 个验证 shard 上，用相同种子与超参对比 V1、V2 核心、V2 核心 + 交互，各 1 epoch。

### 第 3 步 — GPU 全量（仅当 G2 通过）

```bash
python task2_model/preprocess.py --feature-set v2 --reuse artifacts --out artifacts_v2
python task2_model/train.py --data artifacts_v2 --out runs/v2_full --epochs 3 --batch-size 8192
python task2_model/predict.py --data artifacts_v2 --split valid \
  --checkpoint runs/v2_full/best.pt --out runs/v2_full/val_preds.npz
python task2_model/stack.py --data artifacts_v2 --out runs/v2_stack \
  --checkpoint runs/full3/best.pt --checkpoint runs/full_tmlp/best.pt \
  --checkpoint runs/v2_full/best.pt
python task2_model/compare.py --base runs/ensemble_stack/val_preds.npz \
  --candidate runs/v2_stack/val_preds.npz --out runs/v2_stack/comparison.json
```

### 第 4 步 — 决策

若 `compare.py` 显示 ΔRMSE ≤ −0.001 且置信区间不含 0，则把 V2 行写入结果表并记录采纳配置；否则如实记录负结果，总表数字保持不变。

---

## 9. 风险与应对

| 风险 | 应对 |
| --- | --- |
| 形态特征与 NN 已能推断的信息高度重复，收益可能接近 0 | G2 子集门槛在消耗 GPU 之前用约 10 分钟 CPU 就能识别 |
| 叠加层的 `base_dir` 间接寻址可能破坏 `predict.py` / `linear_baseline.py` | 两者统一走 `loader.shard_dir`，并由 v1 回归测试锁定 |
| 标准化统计在 v1（26）与 v2（33）之间错位 | 统计始终按完整 schema 单遍拟合，并在加载时断言 `len(mean) == len(dense_columns)` |
| 磁盘压力 | 叠加层仅新增 KB 级；预检在可用空间不足时给出明确报错并中止 |
| 纯 CPU 单 epoch 需 8–9 分钟 | GPU 全量之前的所有迭代都用子集或 shard 限制 |

---

## 10. 前提假设与不做的事

- "不影响当前模型" = 既不改动现有 artifacts / checkpoint / 结果，也不采纳未经验证的改动；第 5、8 节已分别覆盖。
- 即使 `tokenized/` 逐字节相同，仍将其作为 V2 的规范数据源；指纹校验会在将来出现真正不同的数据投放时提示直接重跑同一流程。
- 不做的事：不重新运行 Part A（仓库内没有清洗后的原始文本，也没有 Spark 环境）；不删除或重建现有 `artifacts/`；不改动 `model.py` 的网络结构。

---

## 11. 回滚方式

V2 全部为增量：删除 `artifacts_v2/`、`runs/v2_*`，并把 `features_v2.py`、`compare.py`、`stack.py`、`tests/` 移除即可恢复。`common.py` / `dataset.py` / `preprocess.py` / `train.py` / `predict.py` 的改动均以"默认等于旧行为"为约束，反向 diff 后旧命令仍可复现 1.1543。因 `model.py` 与 `artifacts/` 从未改动，现有模型不受回滚影响。

---

## 12. 执行记录（2026-10-28）

### 12.1 环境

本机为 Linux + Python 3.10.12（无 GPU，16 核，7 GB RAM）。按第 8 节第 0 步用 `uv` 重建 `.venv`：

| 包 | 版本 |
| --- | --- |
| torch | 2.14.1+cpu |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| pyarrow | 25.0.1 |
| scikit-learn | 1.7.2 |

`pip check` 无冲突。**注意**：torch 版本高于产出原结果的 2.9.1，同一配置的绝对 RMSE 会漂移约
3e-4，因此 G2 在同一环境内重跑 V1 作为对照，而不是直接对比 1.166614。

### 12.2 第 1 步：数值回归基线（改动前）

用 `git archive HEAD` 取出改动前的代码，在 `valid` 第 0 个 shard（143,543 行）上跑
`predict.py --data artifacts --checkpoint runs/full3/best.pt --max-shards 1`：RMSE **1.1560205**。
改动后同一命令的预测向量与改动前**逐位相同**（`np.array_equal` 为 True，最大绝对差 0.0），
并已固化为 `tests/test_backcompat.py` 中的 SHA-256 断言。

### 12.3 实施情况

第 6 节清单全部落地：`features_v2.py`、`common.py`（列注册表 + `dense_schema` + 感知 schema 的
`select_dense_columns`）、`preprocess.py`（`--feature-set v2` 叠加层）、`dataset.py`（从 meta 读列名、
`base_dir` 间接寻址、标准化前叠加）、`train.py`、`predict.py`、`compare.py`、`stack.py`、`tests/`
（3 个文件 25 条用例全部通过）、文档。

与计划的**两处有意偏离**，均已在代码注释与 `task2_model/README.md` 第 10 节说明：

1. `build_dense_matrix` 未增加 `feature_set` 参数。它的输入是 parquet table，而 V2 的 7 个文本形态列
   是**缓存里 CSR 数组**的函数；让同一函数同时接受"原始哈希下标"和"压缩后下标"两种约定，正是
   训练/推理不一致的典型来源。改由 `features_v2.build_matrix` 统一从缓存数组计算，保证预处理与
   训练走完全相同的代码路径。
2. 叠加层默认拟合**完整 37 列**（`--interactions` 默认开启），把交互块变成"运行期选择"而非
   "缓存 schema"。这样一次扫描、一个目录即可同时支持核心与 +交互两种消融。

另外修正了 `train.py` 的一个既有缺陷：`args.json` 原先在解析特征集之前写出，会留下 `null`；
现在在解析之后写出。

### 12.4 第 2 步：G2 门槛（负结果）

1.52 M 行 / 1 epoch / 种子 42，超参与既有消融完全一致：

| 方案 | dense 列数 | val RMSE | val MAE | ΔRMSE vs V1（95% CI） |
| --- | ---: | ---: | ---: | --- |
| V1 对照 | 26 | 1.166958 | 0.889986 | — |
| V2 核心（+7 形态列） | 33 | 1.167099 | 0.889837 | **+0.000142** [−0.000154, +0.000431] |
| V2 核心 + 交互 | 37 | 1.166775 | 0.891777 | **−0.000183** [−0.000473, +0.000104] |

两个方案都**没有**跨过 G2 门槛，置信区间均含 0；交互组虽然 RMSE 略降，但 MAE 显著变差
（+0.00179，CI 不含 0）。

**因此按 G2/G4 停止**：不执行第 3 步全量训练与三模型堆叠，总表数字维持 1.1543。
数字已归档到 `task2_model/results/v2_gate.json`。

### 12.5 实施中发现的数据事实（值得单独记录）

1. `Task2_Token_Dataset/tokenized/` 与 `Task2_OriginData/output/` 的特征 parquet **逐字节相同**
   （64 + 11 个文件，训练 shard 0 的 SHA-256 均为 `2a5118a0…`，字节数 1,726,687,142 + 430,329,320），
   已独立复核，与第 1 节结论一致。
2. `user_unseen` / `product_unseen` / `parent_product_unseen` 在**训练集上恒为 0**（6,104,774 行上
   `mean = 0`、方差为 0），却在验证集上非零。这是结构性的：unseen 标记的含义就是"未出现在训练期
   实体统计里"。所以这 3 列**学不到任何信号**，第 3.2 节的 `unseen_sum` 交互是死列——这也解释了
   为什么交互组只有 MAE 变差。
3. `title_comment_cosine` 必须在**原始哈希空间**计算。两列的压缩下标互不相干，若直接对存储下标做
   点积，只有 0.22% 的行非零（死特征）；映射回哈希值后是 26.8%。这一点已由
   `tests/test_v2_features.py::test_cosine_uses_hashed_ids_not_stored_ids` 锁定。
