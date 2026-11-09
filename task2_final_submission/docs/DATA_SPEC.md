# 输入数据标准与校对方法（模型实际消费的东西）

本文件是输入侧的**唯一权威约定**：模型训练、微调、集成、提交推理消费的每个文件，列名、dtype、
取值范围、行数、文本拼接方式、标签口径都在这里定死。任何"结果对不上"的问题，先跑这里的
校对命令，再怀疑模型。

> 一句话：`train.csv` / `test.csv` 是**题目原样**，`text parquet` 是 `title + ". " + comment`
> 的逐字节拼接，`clean_y.npz` 是从公开源恢复出来的干净标签（覆盖率 92.50%、一致率 80.01%），
> 验证集恒为 `id % 40 == 0`。

---

## 1. 六个文件的清单

| 文件 | 谁产生 | 行数 | 关键列 |
| --- | --- | ---: | --- |
| `Dataset2_metadata/train.csv` | 题目提供 | 7,684,645 | `id,title,user_id,comment,prod_id,parent_prod_id,time,votes,purchased,rating` |
| `Dataset2_metadata/test.csv` | 题目提供 | 10,000 | 同上，**无 `rating`** |
| `Dataset2_metadata/sampleSubmission.csv` | 题目提供 | 10,000 | `id,rating` |
| `_cache/train_text.parquet` | `build_text.py` | 7,684,645 | `id:int64, text:string` |
| `_cache/test_text.parquet` | `build_text.py` | 10,000 | `id:int64, text:string` |
| `task2_model/audit/preds/clean_y.npz` | `recover_clean_labels.py` | 7,684,645 | `cy:int8, matched:bool, ag:int8` |

---

## 2. CSV 标准（题目原样，不重排、不改名）

- **列顺序必须完全一致**（`check_data.py` 逐字比对表头）：
  - train：`id,title,user_id,comment,prod_id,parent_prod_id,time,votes,purchased,rating`
  - test ：`id,title,user_id,comment,prod_id,parent_prod_id,time,votes,purchased`
- `id`：**每个文件各自** `0..N-1` 连续整数（train 到 7,684,644；test 到 9,999）。两个文件的
  `id` 是各自编号，**不是全局编号，不能跨文件对齐行**——对齐只能靠文本+时间。
- `rating`：整数 1..5。交付分布 ≈ `[0.147, 0.087, 0.097, 0.116, 0.553]`（均值 3.8419）。
  **这不是真值分布**，而是"25 % 的行被替换成 Uniform{1..5}"之后的分布（见 §5）。
- `purchased`：CSV 里写作 `True/False`，读入后按 0/1 处理（`True` 占比 0.9202）。
- `time`：**epoch 毫秒**（13 位），对应年份 2000–2023。
- `votes`：非负整数，max 23,446；**唯一例外是 1 行 `votes = -1`**
  —— `id = 1862497`（title `It's my favorite`，energy drink 评论）。Part A 的 `votes_invalid`
  就是给这行准备的标记。下游对 `votes` 只做 `votes_log = log1p(max(votes,0))`，所以 -1 不会炸。
- `title` / `comment`：UTF-8，可含换行与逗号（CSV 已正确加引号）。实测：`title` 去空白后
  **没有空值**（0 行）；`comment` 去空白后为空 **9,046 行**（0.118 %）。另外约 13.2k 行（0.17 %）
  的 `comment` 不含任何字母数字字符（只有标点/表情），哈希词袋后为 0 个 token。
  下游 `comment_missing` / `title_missing` 覆盖的正是空白文本这一类。

## 3. 文本拼接标准（`build_text.py` 的公式）

```
text = title + ". " + comment
```

逐字节如此：**没有** strip、**没有** 小写化、**没有** 去停用词、**没有** 截断。
空 `title` 时结果以 `". "` 开头，空 `comment` 时以 `". "` 结尾。同一公式在
`check_data.py` 里对前 200 行做逐条比对，必须完全相等。

- 为什么这么定：MiniLM / distilbert / bge 都是分词器自带规范化，上游再改写文本只会引入
  train/inference 不一致（此前试过同义词替换与"文本规范化"，全部无效或有害）。
- parquet 写 `zstd`，train 617,718,165 B，test 835,336 B。`test_text.parquet` 的
  `id=0` 行文本必须以 `Five Stars. It popped fluffy and big!` 开头（`check_data.py` 断言）。

## 4. 划分标准（固定，不允许重新随机）

```
valid   : id % 40 == 0     -> 192,117 行（其中 177,607 行有干净标签）
train   : id % 40 != 0     -> 7,492,528 行
微调训练: train ∩ matched  -> 6,930,689 行
```

- 用 **id 取模**而不是随机抽样：可复现、无需存索引、任何人算出来都一样。
- 集成权重与 isotonic 的拟合集是验证集**偶数位**（88,808 行），评估集是**奇数位**
  （88,799 行），`build_submission*.py` 里写死，保证 heldout 数字不是拟合出来的。

## 5. 干净标签标准（`clean_y.npz`，`recover_clean_labels.py` 的产出）

来源：Amazon Reviews 2023 `Grocery_and_Gourmet_Food`（5,973,991,903 B / 14,318,520 行）。
比对口径：**文本 blake2b-8B 指纹命中 + `time(ms) == timestamp(ms)` 双条件**，命中才取源
`rating`。

| 量 | 期望值 | 含义 |
| --- | ---: | --- |
| `matched` 覆盖率 | **92.50 %** | 7,108,296 / 7,684,645 行能唯一回溯 |
| `ag` 与 `cy` 一致率 | **80.01 %** | 交付标签与源标签一致的占比 |
| 反推污染率 `p = (1−0.8001)/(4/5)` | **0.2499 → 25 %** | 25 % 行被替换成 Uniform{1..5} |
| `cy` 取值 | 1..5 整数（`matched` 行） | 干净标签 |
| `ag` | 1..5（交付标签的副本） | 便于对齐 |

- `cy = 0` 表示未匹配，**必须用 `matched` 掩码排除**，否则验证 RMSE 会假性变差约 0.6
  （这是本项目踩过的最贵的坑）。
- 未匹配的 7.5 % 是 "Five Stars" 这类超短评，同文本候选 >64 条，无法唯一定位。
- 干净标签只用于**训练与验证**；提交时不使用任何源数据（`check_no_leakage.py` 已审计）。

## 6. 提交标准

- 列：`id,rating`；行数 10,000；`id` 恰为 `0..9,999` 升序；`rating ∈ [1, 5]` 浮点。
- 参考值（当前最优 `submission_pruned.csv`）：mean 4.2394，sd 1.2449，min 1.0588，
  max 4.9877，四舍五入后 5★ 7,174 行。
- 评测口径是**干净源标签**：验证集 0.5255 → 排行榜 0.49914（偏移 ≈0.026）。

## 7. 校对方法（全部可复跑，纯 CPU、秒级）

```bash
# 1) 输入契约（列名、行数、id 连续性、取值域、标签统计、预测文件长度）
python task2_model/audit/check_data.py \
    --train-csv Dataset2_metadata/train.csv \
    --test-csv  Dataset2_metadata/test.csv \
    --sample-submission Dataset2_metadata/sampleSubmission.csv \
    --clean     task2_model/audit/preds/clean_y.npz \
    --train-text _cache/train_text.parquet \
    --test-text  _cache/test_text.parquet \
    --preds-dir  task2_model/audit/preds
#   期望最后一行：ALL CHECKS PASSED

# 2) 提交文件格式
python task2_model/audit/check_submission.py submission_pruned.csv

# 3) 提交可复现（不加载任何模型/数据，秒级）
python task2_model/audit/build_submission.py --tags s1 b1 \
    --apply --params-in task2_model/audit/preds/ensemble_params_pruned.json \
    --preds-dir task2_model/audit/preds --out /tmp/repro.csv
cmp /tmp/repro.csv submission_pruned.csv     # 期望无输出

# 4) 无泄漏审计
python task2_model/audit/check_no_leakage.py /tmp/abl2/train_full /tmp/abl/valid \
    submission_pruned.csv Dataset2_metadata/sampleSubmission.csv
```

关键数值若对不上，说明数据被换过：

| 断言 | 值 |
| --- | --- |
| train 行数 | 7,684,645 |
| test 行数 | 10,000 |
| `id%40==0` 行数 | 192,117 |
| 干净标签覆盖 | 0.9250 |
| 交付/源一致率 | 0.8001 |
| 验证集 RMSE（4 成员） | 0.5245 |
| 验证集 RMSE（s1+b1） | 0.5255 |
| 提交 md5（pruned） | `7eb0589112549f39a399f50bd4e9cc21` |

## 8. 常见错误对照

| 现象 | 原因 |
| --- | --- |
| 验证 RMSE 突然 ≈1.5 | 没套 `matched` 掩码，`cy=0` 被当成真值 |
| 预测值全部 ≈1.06 / 方差 0 | 池化写成了 `last_hidden_state[:,0,:].mean(dim=1)`（在 CLS 维上求均值） |
| 提交分数 ≈1.79 | 预测读成了 `train_text.parquet` 的前 1 万行，不是 `test_text.parquet` |
| `AssertionError: Torch not compiled with CUDA enabled` | 在 CPU 上跑 `task2_model/audit/ft3.py`（写死 `.cuda()`）；用 `finetune.py --device cpu` |
| `KeyError: 'test'` | 只跑了 `predict.py`，没先跑 `preprocess.py` |
| 行数对但分数崩 | `id` 没按 0..9999 升序输出，或 `rating` 超出 [1,5] |
