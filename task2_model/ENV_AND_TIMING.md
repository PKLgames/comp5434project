# 运行环境与耗时（实测记录）

本文档记录最终方案（LB 0.49893）的运行环境与各阶段耗时。每条数据都标注来源类型：

- **实测**：有日志/命令输出可查；
- **推导**：由实测吞吐 × 已知步数/样本数算出，并给出校验（推算值与实测总量互相对照）；
- **未记录**：当时没有采集，现已无法回溯；
- **估算**：由实测吞吐外推，明确标注、不作为结论使用。

证据源：本机终端输出、`comp5434project/runs/gpu_execution/environment.json`、以及
**Colab CLI 的会话日志** `colab-cli/home/.config/colab-cli/history/{work,work2}.jsonl`
（记录 `session_created` / `session_terminated` / `file_operation` / `execution` 事件，
带精确 UTC 时间戳与 exec 的 stdout/stderr/traceback）。凡标注「日志证据」的条目都能在该日志里复核。

---

## 1. 训练硬件（按模型）

| tag | 加速器 | 显存 | vCPU | 内存 | maxlen / batch | 吞吐 | 训练总耗时（2 轮） |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| s1 | **L4** | 23,034 MiB | 12 | 52 GB | 96 / 512 | 3.55 it/s | 7,992 s（2 h 13 min） |
| s2 | **G4**（NVIDIA RTX PRO 6000 Blackwell） | 97,887 MiB | 48 | 176 GB | 128 / 1024 | 未记录（由总时长反推 ≈6.0 it/s） | 2,264 s（38 min） |
| d1 | **G4** | 97,887 MiB | 48 | 176 GB | 128 / 768 | 4.82 it/s | 3,869 s（1 h 5 min） |
| **b1** | **G4** | **97,887 MiB** | **48** | **176 GB** | **128 / 768** | **5.39 it/s** | **3,484 s（58 min）** |

- **b1 确认跑在 G4 上**（RTX PRO 6000 Blackwell，97,887 MiB 显存，48 vCPU，176 GB 内存），不是 L4 也不是 T4。b1 是单模型最强的一个（验证 0.5328），也是集成权重最大的一项（0.562）。
- L4 的磁盘为 113 GB；G4 磁盘未记录。
- **云端 CPU 型号：未记录，且无法事后补齐**。当时只采集了 `nproc`（L4=12、G4=48）与内存（52 GB / 176 GB），没有跑 `lscpu`；Colab 会话销毁后主机信息不可再查，日志里也没有任何 CPU 型号字符串（已对两个 JSONL 全文检索 `Xeon`/`EPYC`/`model name`，零命中）。要补只能在下一次会话执行 `lscpu | grep 'Model name'` 并记录。唯一可确认的是核数与内存量级（见上表），据此可判断 G4 是 48 vCPU / 176 GB 的高核数机型。
- GPU 型号在日志里有直接输出：`torch.cuda.get_device_name(0)` → `NVIDIA L4`（2026-11-07T15:59:16Z 的 bench 单元）。

## 2. 软件版本

| 环境 | Python | PyTorch | CUDA | transformers | 其他 |
| --- | --- | --- | --- | --- | --- |
| Colab GPU（s1/s2/d1/b1 训练） | **3.13**（日志证据） | **2.14.1+cu130** | **13.0**（由 `cu130` 构建号确定） | **5.x**（日志证据，精确小版本未打印） | — |
| 本机 CPU（预处理、集成、校准、分词计时） | 3.10.12 | 2.14.1+cpu | 无 GPU | 5.20.2 | numpy 2.2.6, scipy 1.15.3, sklearn 1.7.2, pyarrow 25.0.1；i7-1260P 16 线程，7 GB 内存 |
| Part-B GPU 盒子（`runs/gpu_execution/environment.json`） | 3.11.8 | 2.6.0+cu124 | 12.4 | 未装 | numpy 2.4.6, scipy 1.17.1, pyarrow 26.0.0, sklearn 1.9.1；RTX 3060 Laptop 6 GB |

**Colab Python = 3.13**：2026-11-07T14:53:48Z 的执行记录里，tokenizer 抛出的 traceback 逐帧路径为
`/usr/local/lib/python3.13/dist-packages/transformers/tokenization_utils_base.py`、
`/tmp/ipykernel_1179/…`，即解释器为 3.13（同一日志中 3 处 `python3.13` 命中，无其它版本）。

**Colab transformers = 5.x**：同一 traceback 里出现 `transformers/tokenization_utils_tokenizers.py`
（v5 起引入的 tokenizer 实现模块，4.x 时代为 `tokenization_utils_fast.py`），且加载模型时打印
`Loading weights: 0%|…| 0/103 [00:00<?, ?it/s]` 进度条（v5 的 safetensors 分片加载提示）。
精确小版本当时没有 `print(transformers.__version__)`，**不可再补**（会话已销毁）。

**Colab PyTorch 2.14.1+cu130**：在 T4 会话里实测 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
→ `2.14.1+cu130 True`；L4/G4 与 T4 同镜像族但训练时没有逐台复测。

## 3. 训练耗时明细

总时长是各次运行进程的 wall-clock（含验证与保存）。**逐轮耗时当时没有在 epoch 边界打印**，
但可以由「实测步数/轮 × 实测吞吐」推导，并用实测总量校验：

| tag | steps/epoch | 实测 it/s | 推导 每轮 | 推导 2 轮 | 实测 2 轮 | 差值（= 2 次 40k 验证 + 192k 全量预测 + 保存） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s1 | 13,537 | 3.55 | **3,813 s** | 7,626 s | 7,992 s | +366 s |
| s2 | 6,768 | 未记录 | **≈1,132 s**（= 实测 2,264 / 2） | 2,264 s | 2,264 s | — |
| d1 | 9,024 | 4.82 | **1,872 s** | 3,744 s | 3,869 s | +125 s |
| b1 | 9,024 | 5.39 | **1,674 s** | 3,348 s | 3,484 s | +136 s |

- `steps/epoch = ceil(6,930,689 / batch)`，训练行数 6,930,689 = 全部 matched 行（7,108,296）减去验证集里的 matched 行（177,607）。
- 差值项的量级与 s1 对得上：192,117 行全量预测在 L4 上按 3.6k docs/s ≈ 53 s、两次 40k 子集验证 ≈ 22 s、保存 90–265 MB ≈ 3 s，合计约 78 s；剩余约 290 s 是启动期的模型/分词器下载（L4 首次拉 MiniLM 90.9 MB + 空转）。G4 上 d1/b1 的 +125/+136 s 同理（下载 265 MB / 133 MB + 40k×2 验证）。
- 结论：**每轮 ≈ 总时长的一半**（两轮同配置、无早停），误差来自启动下载与末尾全量预测，不超过总量的 5%。

**每个模型的完整 wall-clock 窗口**（UTC，取自会话日志；北京时间 = UTC+8）：

| 阶段 | 起始 | 结束 | 窗口 |
| --- | --- | --- | ---: |
| s1（L4）上传完成 → 预测文件下载 | 2026-11-08T02:58:34Z | 05:13:31Z | 8,097 s |
| s2（G4-a）上传完成 → 下载 | 04:49:04Z | 05:29:57Z | 2,453 s |
| d1（G4-b）上传完成 → 下载 | 05:36:57Z | 06:45:10Z | 4,093 s |
| b1（G4-b）d1 下载完成 → 下载 | 06:45:13Z | 07:45:52Z | 3,639 s |

窗口比上表的训练时长多 150–350 s，差额即 VM 启动、617 MB 数据重组、模型下载与收尾。

## 4. 预处理耗时（本机实测）

| 步骤 | 耗时 | 说明 |
| --- | ---: | --- |
| `build_text.py` 训练集 | **13.0 s** | 7,684,645 行 → 617,718,165 B parquet（另一次 9 s，受磁盘缓存影响） |
| `build_text.py` 测试集 | **0.59 s** | 10,000 行 → 835,336 B |
| 源数据下载（Grocery 5.97 GB） | **≈ 245 s** | 24.36 MB/s，随网络波动，重跑时要重新下载 |
| `recover_clean_labels.py` | **155 s** | 对 1,431 万条源文本建 blake2b 索引 + 768 万次查找校验（此前一次 140 s） |
| 预处理合计 | **≈ 414 s** | 不含下载；含下载约 11 min |

## 5. 集成与校准耗时（本机实测，10,000 行测试集）

| 步骤 | 耗时 |
| --- | ---: |
| 权重搜索（400 次 Dirichlet 采样）+ isotonic 拟合（88,808 行）+ 写 CSV | **3.3 s** |
| `--apply` 重建（不改权重，仅按已存参数写文件） | **1.5 s** |

## 6. 全流程总耗时

- **CPU 侧**：数据构造 13.6 s + 干净标签恢复 155 s + 集成/校准 3.3 s ≈ **172 s**（加上 6 GB 下载约 7 min）。
- **GPU 侧训练**：7,992 + 2,264 + 3,869 + 3,484 = **17,609 s ≈ 4.9 h**。
- **会话开销**：3 次 Colab 会话（L4 一次、G4 两次），每次约 200–340 s（含 617 MB 分 13 片上传与远端重组）。
- **跨会话 wall clock 约 6–7 h**。
- **训练是否并行：没有**。四个模型顺序训练，任一时刻只有一个模型在跑（s1 在 L4 会话；s2 在 G4-a；d1、b1 在 G4-b 依次执行）。

## 7. 完整测试推理耗时（10,000 条 → 提交 CSV）

### 7.1 交付版 full3（权重在位，本机 CPU 端到端实测，10,000 行）

| 阶段 | 命令 | 实测 |
| --- | --- | ---: |
| 测试特征转换（复用 artifacts 词表/IDF） | `preprocess.py --test-dir test_features_with_entity --reuse artifacts` | **0.442 s** |
| 前向 + 写 CSV | `predict.py --split test --checkpoint runs/full3/best.pt` | **2.733 s** |
| **合计（10,000 行 → submission CSV）** | | **3.175 s** |

这是目前仓库里**唯一能端到端复测**的推理链路；`wc -l` 校验输出为 10,001 行（表头 + 10,000 条）。

### 7.2 最终版（4 成员微调集成）

4 个 `ft3_*.pt`（90.9 / 90.9 / 265 / 133 MB）随 Colab 会话销毁、未下载，整链无法复测。
改为**实测单成员完整推理**（相同架构、相同序列长度；微调头只是 384→1 的线性层，成本可忽略），
再乘成员数并加上融合阶段：

| 阶段（CPU 16 线程，10k 条，maxlen 96，bs 256） | 实测 |
| --- | ---: |
| 读 `test_text.parquet`（10k 行） | **0.052 s** |
| 加载 tokenizer | **0.797 s** |
| 加载模型权重（MiniLM 384 维） | **0.564 s** |
| 分词 10k 条 | **0.876 s** |
| 前向 10k 条（按长度排序分桶） | **43.529 s**（229.7 docs/s） |
| **单成员合计** | **45.819 s** |

| 组合 | 计算方式 | 结果 | 类型 |
| --- | --- | ---: | --- |
| 四成员 · 本机 CPU | 4×43.5 前向 + ≈5.4 加载/分词 + 1.5 融合 | **≈181 s** | 实测外推 |
| 四成员 · L4 | 前向 6,122–5,369 docs/s ⇒ 1.6–1.9 s/成员 + 权重加载 10–20 s | **18–28 s** | 吞吐推导 |
| 四成员 · G4 | 前向约快 2–3× | **12–18 s** | 吞吐推导 |

- GPU 前向数来自 `bench_encode.py` 在 L4 上的实测吞吐；CPU 单成员为本次实测（229.7 docs/s，与早先 `bench_emb.py` 的 226 docs/s 一致）。
- 融合 + isotonic + 写盘：本机实测 **1.5 s**（含 npz 读取），纯计算约 0.2 s。
- 首次运行若需从 HF 下载权重（每模型 90–265 MB，24 MB/s），再加 15–30 s。
- 若把真实 checkpoint 之一上传回本机，`audit/measure_pipeline.py` 可按同一分段重跑并直接打印 TOTAL。

## 8. 如何补齐仍缺的项

1. **云端 CPU 型号**：下一次会话首条命令 `lscpu | grep -i 'model name'`（或 `cat /proc/cpuinfo`），30 s 成本；
2. **新鲜推理整体耗时**：已在本机实测单成员与交付版链路（见 §7）；若要严格复测四成员整链，需把任一 `ft3_*.pt` 传回本机后跑 `task2_model/audit/measure_pipeline.py`；
3. **transformers 精确版本**：在 `ft3.py` 开头加 `print(sys.version, torch.__version__, transformers.__version__)`，下次训练自动留下。

> 结论：环境（GPU 型号/显存/核数/内存）、版本（Python 3.13、torch 2.14.1+cu130、transformers 5.x）、
> 训练总量与逐轮推导、预处理、集成校准、全流程总量都有实测或可校验的推导；
> **测试集推理已有实测**：交付版 full3 端到端 10,000 行 = **3.175 s**（CPU），最终版单成员 = **45.819 s**（CPU，229.7 docs/s），
> 四成员在 L4 / G4 上按实测吞吐推导为 18–28 s / 12–18 s。
> 仍缺的只有：**云端 CPU 型号**（日志无、只能下次 `lscpu`）与 **transformers 精确小版本**（只到 5.x）。

## 9. 待办（未执行，用户要求保留）

**仓库瘦身**：把 `comp5434project/` 从 14 GB 压到约 27 MB（仅保留模型相关文件），删除清单已核对：

| 目录 | 大小 | 说明 |
| --- | ---: | --- |
| `Task2_Token_Dataset/` | 2.1 GB | Part-A 的 hashed 特征缓存，可重新生成 |
| `Task2_OriginData/` | 2.2 GB | Part-A 中间产物（entity_statistics 等） |
| `runs/`（除 `gpu_execution/`） | 2.0 GB | 旧版 61M 参数 checkpoint（8 个 × ~240 MB） |
| `Dataset2_metadata/` | 1.9 GB | 竞赛原始 CSV（已 gitignore，用时再下载） |
| `colab/.cache/` | 467 MB | 上传缓存 |
| `.git/`（含 LFS 1.3 GB） | 1.3 GB | 历史与 LFS 对象，可 `git gc` / 重建仓库 |

保留：`task2_model/`（含 `audit/preds/` 26 MB）、`submission_ens.csv`、`test_features_with_entity/`。
执行前需逐项确认，删除会破坏历史复现路径（旧 checkpoint 需从 commit `8a82faa` 取回）。
