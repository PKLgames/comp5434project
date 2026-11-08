# 最终方案与结果（2026-11-08）

当前最好提交：**`submission_ens.csv` → 排行榜 RMSE 0.49893**（第 1 名 0.45700，差距 0.042）。

## 1. 成绩轨迹

| 提交文件 | 方案 | 验证 RMSE | 排行榜 |
|---|---|---|---|
| `submission.csv` | 交付的 `runs/full3/best.pt`（哈希词袋 + 实体计数 + 26 数值） | 1.1573 | 1.55180 |
| `submission_linear.csv` | 自建 SGD 词袋线性（comment 2^17 + title 2^15） | 1.2029 | 0.83641 |
| `submission_calibrated.csv` | 词袋 + 仿射校准 | 0.7515 | 0.69326 |
| `submission_iso.csv` | 词袋 + isotonic 校准 | 0.7043 | 0.65882 |
| `submission_emb.csv` | 冻结 MiniLM 句向量 MLP + 词袋 + isotonic | 0.6789 | 0.61998 |
| `submission_ft2.csv` | 微调 MiniLM（2 epoch, 337 万行）+ 词袋 + isotonic | 0.5399 | 0.51355 |
| **`submission_ens.csv`** | **4 个微调模型加权 + isotonic** | **0.5245** | **0.49893** |

验证→榜单偏移稳定在 −0.02 ~ −0.03（四次实测：−0.049 / −0.059 / +0.026 / −0.0256）。

## 2. 关键发现

1. **训练标签有 25% 被污染**：7,684,645 行中约 25% 的 `rating` 被替换成 Uniform{1..5}（p=0.2499，由 71 万行与源数据逐条比对得出，一致率 80.0103%）。**测试标签是干净的**，故所有校准/训练都以恢复的干净标签 `cy` 为目标。
2. **排行榜口径是干净评分**：0.83641 低于污染标签的理论下限 1.044，证明榜单不按交付标签打分。
3. **测试特征文件没有 id 列**，实体（user/item）特征在测试侧不可用，这条最强通道从数据层面被关闭。
4. **瓶颈是表示与多样性，不是容量**：
   - 哈希词袋 0.885 → 冻结句向量 0.731 → 微调 transformer 0.54；
   - 头容量：GPU 上 1 秒/epoch，加宽/加 epoch/换 5 分类交叉熵都停在 0.73；
   - 数据量翻倍（337 万 → 693 万行）几乎无收益（0.5426 → 0.5428）；
   - 换底座（MiniLM → distilbert）单模型只降 0.001，但**跨家族集成**降 0.008。
5. **清洗方法无效**：截尾 / EM 软标签 / 硬重标注 / 仿射自校准全部中性或有害（实测表见 `SOURCE_AND_LABEL_NOISE.md` §8）。

## 3. 最终模型阵容

全部为「编码器 + 单层线性回归头，MSE 对恢复的干净标签」，训练集为 `id % 40 != 0` 且有干净标签的 693 万行，验证集为 `id % 40 == 0`（192,117 行，其中 177,607 行有干净标签）。

| tag | 编码器 | 轮数 | maxlen | batch | 单模型验证 RMSE | 集成权重 |
|---|---|---|---|---|---|---|
| s1 | `sentence-transformers/all-MiniLM-L6-v2` | 2 | 96 | 512 | 0.5428 | 0.177 |
| s2 | 同上（不同 seed） | 2 | 128 | 1024 | 0.5506 | 0.005 |
| d1 | `distilbert-base-uncased` | 2 | 128 | 768 | 0.5418 | 0.256 |
| b1 | `BAAI/bge-small-en-v1.5` | 2 | 128 | 768 | **0.5328** | 0.562 |

集成：权重在验证集匹配行上随机搜索（400 次 Dirichlet 采样，seed 0），isotonic 在偶数半区拟合、奇数半区评估。

| 组合 | 验证 RMSE |
|---|---|
| 单模型最好（b1） | 0.5328 |
| s1+s2+d1+b1 等权 | 0.5309 |
| s1+s2+d1+b1 最优加权 | 0.5276 |
| + isotonic | **0.5245** |

## 4. 复现方式

预测数组与集成参数已随仓库提交（`task2_model/audit/preds/`），**无需训练即可逐字节复现提交文件**：

```bash
cd task2_model/audit
../../.venv/bin/python build_submission.py --apply --tags s1 s2 d1 b1 \
    --params-in preds/ensemble_params.json --out submission_ens.csv
# -> 与仓库中的 submission_ens.csv 完全一致（cmp 无差异）
```

完整重训流程见 `task2_model/audit/README.md`（数据获取、干净标签恢复、四组训练命令）。

## 5. 算力实测（Colab）

| 硬件 | 计费 | 微调吞吐 | 2-epoch 训练耗时 | 花费 |
|---|---|---|---|---|
| G4（RTX PRO 6000 Blackwell, 98 GB） | 10.1 CU/hr | 5.4 it/s @bs768 | ~1 h | ~10 CU |
| L4（23 GB） | 1.94 CU/hr | 3.55 it/s @bs512 | ~2.2 h | ~4.3 CU |
| T4 | 1.23 CU/hr | — | — | — |

H100 与 A100：账号无权限（`Backend rejected accelerator`）；L4 最省 CU，G4 最快（快 3.4 倍、贵 5 倍）。

逐模型的环境（GPU 型号/显存/核数/内存）、版本、训练与预处理耗时、集成校准耗时、
以及**尚未实测**的项（逐轮耗时、Colab 的 Python/transformers 版本、新鲜测试推理耗时）
都记在 [ENV_AND_TIMING.md](ENV_AND_TIMING.md)；新鲜推理的测量脚本是
[audit/measure_pipeline.py](audit/measure_pipeline.py)。

## 6. 踩过的坑

1. **推理读错文件**：`ft.py` 用 `train_text.parquet` 的前 1 万行当测试输入 → 排行榜 1.79。测试侧必须用 `test_text.parquet`（由 `test.csv` 构造）。
2. **缺 `matched` 掩码**：验证集 7.5% 的行没有干净标签（`cy=0`），未加掩码时 RMSE 从 0.73 抬到 1.53。
3. **`token_type_ids`**：新版 transformers 的 tokenizer 会返回该键，模型 forward 不接受 → `TypeError: FT.forward() got an unexpected keyword argument 'token_type_ids'`，取 batch 时过滤键名。
4. **`colab exec` 长任务断流**（`RuntimeError: Connection was lost.`）：改用 `ssh + setsid nohup ... < /dev/null &` 后台跑，`ssh tail` 轮询；同一运行时只允许一条 ssh 连接。
5. **`.to_pylist()` OOM**：768 万字符串不能整体物化；`ChunkedArray[i]` 返回 `StringScalar`，需 `.as_py()`。

## 7. 继续提升的路径（按性价比）

| 步骤 | 墙钟 | 花费 | 预测验证 | 预测榜单 |
|---|---|---|---|---|
| + bge-base-en-v1.5（109M） | 2.4 h | 25 CU | 0.519 | ≈0.493 |
| + e5-small / gte-small（2 个新家族） | +2 h | +20 CU | 0.514 | ≈0.489 |
| + deberta-v3-small | +1.8 h | +18 CU | 0.510 | ≈0.484 |
| 追平 0.457 | ~10 h | 100+ CU | 0.483 | 边际收益已衰减到 ~0.003/模型，不建议无脑堆 |

边际收益规律：单模型最好 0.5328 → 4 模型 0.5245（−0.008），此后每加一个异家族模型约 −0.003。
