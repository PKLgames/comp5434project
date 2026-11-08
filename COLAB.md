# 在 Colab 上运行 Task 2（适配说明）

本项目原本假设「本机 venv + pip install + 本地磁盘」。在 Colab 上跑只需要替换
**启动方式**，模型代码一行都不用改。本文给出最短可用流程，以及每一步在消耗什么。

相关脚本都在 [`colab/`](colab/) 下，全部只依赖你已经装好的 `colab` 命令行
（见 `google-colab-cli` 的 `colab ssh`）。

---

## 0. 为什么这样适配

| 本机做法 | Colab 上改成 | 原因 |
| --- | --- | --- |
| `python -m venv` + `pip install -r requirements.txt` | **不装**，直接用镜像自带的 torch/pyarrow/scipy/numpy | Colab 镜像已带 `torch 2.11+cu130`、`pyarrow 23`、`numpy 2.4`、`scipy 1.16`；装一遍要 2~5 分钟且会顶掉 CUDA 版本 |
| 数据放本地磁盘 | 一次性推到 `/content/comp5434project` | `/content` 是临时盘，但同一 session 内一直有效；不挂 Drive，不付 Drive 的 OAuth/挂载时间 |
| `pip install torch --index-url cu124` | 省略 | 上面那步已经解决 |
| 长训任务 | 先小样本 smoke（`--max-shards 1`）再全量 | 验证集全量只要 ~40 秒；先确认代码在 py3.13/torch2.11 下能跑通，避免白跑 |

**不挂 Drive** 的意思：验证集缓存（471 MB）通过 `colab ssh` 的二进制 stdin 一次性
流进去，走的是 WebSocket 隧道本身，没有 Contents API 的 base64 放大，也没有
Drive 挂载失败/重新授权的问题。代价是每次新建 session 要重推一次（约 60 秒，
打包结果缓存在 `colab/.cache/`，第二次不用重新 tar）。

---

## 1. 一次性的准备

`colab` 不在 PATH 上，而且**不能裸调**：CLI 把会话状态、keyring 令牌和日志都写在 `$HOME`
下，真正通过认证的那份在 `colab-cli/home` 里；真实 `$HOME` 那份没有令牌，其日志路径又落在
可写工作区之外（实测报 `PermissionError: /home/pklgames/.config/colab-cli/colab.log`）。
所以调用前要有这两个变量（`colab/_common.sh` 已代劳，直接跑 `./colab/*.sh` 就不用管）：

```bash
export HOME=/home/pklgames/coding/colab-cli/home
export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring
export PATH=/home/pklgames/coding/colab-cli/bin:$PATH   # 或把 COLAB_BIN 设成全路径
colab sessions      # 期望输出: No active sessions found on server.
```

下面的密钥检查在 `$HOME/.ssh/`（即 `colab-cli/home/.ssh/`）下进行：

```bash
# 确认 CLI 和密钥（Colab 只接受 ed25519 / ecdsa，RSA 会被服务端拒绝）
colab --version
ls ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
```

## 2. 三个动作：起 → 跑 → 关

```bash
# ① 起一个 T4 会话（约 15 秒；没有 GPU 额度时去掉 --gpu T4 走 CPU）
colab new -s work --gpu T4

# ② 推代码 + 验证集缓存 + checkpoint（首次约 60 秒，之后走缓存）
./colab/push.sh work

# ③ 跑验证集，回收指标 JSON
./colab/run_valid.sh work

# ④ 用完立刻释放，这一步才真正停止计费
./colab/stop.sh work
```

实测输出（2026-11-07，T4 / torch 2.11.0+cu130）：

```
predicting 1,526,196 rows on cuda with 1 model(s)
val RMSE=1.1573 MAE=0.8646 rows=1,526,196
```

与仓库里发布的 `runs/full3`（torch 2.9.1/CPU，RMSE 1.1573 / MAE 0.8646）一致到小数点后
四位；逐条预测的配对差异见 `runs/colab_valid/comparison_vs_published.json`。
不同 torch 版本之间最后几位会有 1e-4 量级的差别，这是预期的，判断「是否变好」要用
`compare.py` 的配对 bootstrap，而不是肉眼比 RMSE。

## 3. 花费

| 项 | 实测 |
| --- | --- |
| 建 session | ~15 s |
| 推 467 MB 包（含 tar/gzip） | ~60 s（网络 4~5 MB/s） |
| 验证集全量打分（1.53 M 行） | ~35 s GPU |
| T4 计费率 | 1.23 CU / 小时 |
| 整轮实际计费 | < 0.05 CU（会话存活 ~3 分钟） |

真正贵的是「挂着不动」：T4 空转也是 1.23 CU/h。所以流程固定为
**起 → 推 → 跑 → 立刻 stop**，不要留着过夜。`colab usage` 随时看余额和当前速率。

## 4. 训练怎么做（需要时可选用）

训练需要完整的 `artifacts/`（约 2.4 GB）和 `runs/*/best.pt`，推送慢得多：

```bash
./colab/push.sh work --full          # 推 artifacts/ + 两个 best.pt（数分钟）
./colab/shell.sh work                # 进远端 shell，目录已在 /content/comp5434project
python task2_model/train.py --data artifacts --out runs/colab_full \
  --epochs 3 --batch-size 8192 --valid-batch-size 16384 \
  --text-dim 256 --dense-dim 128 --hidden-dim 256 \
  --dropout 0.15 --emb-dropout 0.1 --seed 42 --save-val-preds
```

注意：Colab 运行时只有 2 个 vCPU，`--num-threads 16`（GPU_RUNBOOK 里的值）在
这里没有意义，可以省略；`--batch-size 8192` 在 16 GB 的 T4 上没问题。

## 5. 排错

| 现象 | 原因 / 处理 |
| --- | --- |
| `colab: command not found` | 没设 PATH / `COLAB_BIN`，见 §1 |
| `PermissionError: .../.config/colab-cli/colab.log` | 没设 `HOME=colab-cli/home`，见 §1 |
| `no SSH public key found` | 没有 ed25519/ecdsa 密钥，或指定 `COLAB_SSH_ARGS="-i <key>"` |
| `Session 'work' not found` | 会话已被回收/停止：重新 `colab new -s work --gpu T4` 并 `push.sh` |
| 传输中断（tar 报错/连接关闭） | 重跑 `./colab/push.sh work`（会重新解包，幂等） |
| `checkpoint/cache mismatch` | 推漏了 `artifacts/meta.json` 或推的是 V2 缓存；重跑 push |
| 想先花 5 秒验证代码 | `EXTRA_ARGS="--max-shards 1" ./colab/run_valid.sh work` |
| 想核对逐条预测 | `FETCH_PREDS=1 ./colab/run_valid.sh work`，再跑 `compare.py` |

## 6. 文件清单

```
colab/
  _common.sh     公共变量/函数（会话名、远端路径、ssh 包装）
  push.sh        本地 → 远端，tar 流式传输；--full 推训练集
  run_valid.sh   远端跑验证集，回收 metrics（可选拉 preds）
  shell.sh       交互式 shell（便于长训练）
  stop.sh        释放 VM（停止计费）
```

环境变量覆盖：`COLAB_SESSION`（默认 `work`）、`REMOTE_ROOT`（默认
`/content/comp5434project`）、`COLAB_BIN`（默认 `colab`）、`COLAB_SSH_ARGS`（额外
`colab ssh` 参数，如 `-i`）。
