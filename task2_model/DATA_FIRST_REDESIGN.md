> **2026-11-07 更新（本文部分结论已被取代）**：原始 CSV 到手后已确认源数据为
> Amazon Reviews 2023 / Grocery_and_Gourmet_Food，并发现 train.csv 有 **25% 的标签被替换为 Uniform{1..5}**。
> 因此本文中"标签方差只能解释 47%""信息上限 1.09–1.10"的解读需按噪声模型重算：
> 硬下限 0.707，oracle 下限 1.056，文本下限 1.140。详见 `task2_model/SOURCE_AND_LABEL_NOISE.md`。

# Dataset 2 — source verification, measured ceiling, and a data-first redesign

All numbers below were recomputed from the files in this repo (see `diag_linear.py`,
`diag_overlap2.py`, `diag_noise_floor.py`, `diag_floor2.py`).

---

## 1. Where the data comes from

**Verdict: Amazon Reviews 2023 (McAuley Lab, UCSD — arXiv 2403.03952), a derived subset of the
raw per-category review dumps. Schema confidence ~95 %, dataset-family confidence ~85 %,
exact category < 30 %.**

The raw dump has exactly 10 fields, and all 9 assignment columns map onto them 1:1:

| assignment | Amazon-2023 | note |
|---|---|---|
| `rating` | `rating` | exact |
| `title` | `title` | exact |
| `user_id` | `user_id` | exact |
| `comment` | `text` | renamed |
| `prod_id` | `asin` | renamed |
| `parent_prod_id` | `parent_asin` | renamed |
| `time` | `timestamp` | renamed |
| `votes` | `helpful_vote` | renamed |
| `purchased` | `verified_purchase` | renamed |
| `id` | — | synthetic re-index of the course file |

`images` is the only source field dropped. The assignment's own sentence "same products with
different colors share identical parent ASIN" is literally Amazon's definition of `parent_asin`;
`parent_asin` + `verified_purchase` + `helpful_vote` together exist only in the 2023 dump
(the 2014/2018 dumps have no parent ASIN and no verified-purchase flag). Datafiniti, Kaggle
mirrors, JD/Taobao are excluded on the same grounds.

**Why the exact category cannot be pinned down by row count.** The corpus is 7,684,645 train +
10,000 test = **7,694,645 reviews**, 3,873,422 users, 576,493 products, 459,631 parent products
⇒ **1.99 reviews/user and 13.35 reviews/product**. No raw category has that size (nearest:
Amazon_Fashion 2.5 M, Software 4.9 M, Industrial_and_Scientific 5.2 M, Baby_Products 6.0 M,
Arts_Crafts_and_Sewing 9.0 M, Grocery 14.3 M). The per-user and per-item densities also fall
below every raw category (raw ones run ~2.0–3.0 reviews/user and 18.8–55 reviews/item),
so the course file is a **filtered / k-core / merged subset**, not a raw dump — which is
precisely why row-count matching cannot succeed.

Distribution fingerprint (5-star shares, ours = .147/.086/.097/.116/.554) is closest to
Amazon_Fashion (L2 0.051), All_Beauty (0.067), Health_and_Personal_Care (0.076), Software (0.085)
— but all of those are far too small to be the source at 7.69 M rows.

**Consequence:** the public source cannot be used to re-derive our rows, because the ids in the
delivered files are 16-char anonymised hashes (`KW8FBJKJZE2PXK2A`) and the raw `title`/`comment`
text no longer exists anywhere in this repo.

---

## 2. What the delivered feature medium can actually express ("the ceiling")

The only committed data are hashed TF-IDF vectors + 26 numeric columns. Raw text and raw ids are
gone, and `test_features_with_entity/part-*.parquet` has **no user/product ids at all**, so no
identity feature can ever be applied at inference even if we built one.

Measured on the 6,104,774-row train split, using exact duplicate-text groups as a direct estimate
of the irreducible noise `E[Var(y | text)]`:

| quantity | value |
|---|---|
| Var(y) | 2.2949 (sd 1.5149) → constant-mean RMSE 1.5149 |
| unique comment fingerprints | 5,266,552 (86.3 %) |
| rows inside duplicate-text groups | 922,882 (15.1 %) in 84,660 groups |
| pooled within-duplicate-text variance | 1.3169 → **RMSE 1.148** |
| stratum-weighted (by comment nnz) floor | 1.1007 → **RMSE 1.049** |
| text + all 26 dense features, duplicate groups | 1.0918 → **RMSE 1.045** |
| shipped two-model ensemble | **1.1543** |
| shipped single hybrid NN | 1.1573 |

So the Bayes-optimal predictor on this medium sits at roughly **RMSE 1.05–1.15**, and the shipped
1.154 is already within ~0.05–0.10 of it. Ridge ablations agree on the structure:

| model (762 k train rows) | RMSE |
|---|---|
| constant mean | 1.5151 |
| dense (26 numeric) only | 1.8732 — worse than the constant |
| text only (α = 10) | 1.2140 |
| text + dense | 1.2335 — worse than text alone |

Every dense feature correlates with `rating` at |r| ≤ 0.11. **The numeric block carries almost no
signal; the text block carries ~47 % of the variance and that is all there is.** Training a deeper,
wider, better-regularised network on the same tensors cannot cross that bound — which is exactly
why the V2 overlay and the ensemble both landed within 0.003 of the single model.

---

## 3. Why RMSE 0.4–0.5 is arithmetically impossible here

With Var(y) = 2.295, RMSE 0.45 implies **R² = 1 − 0.2025/2.295 = 0.912**, i.e. 91 % of the
variance of individual 1–5★ ratings explained. On real review data no published model reaches
that: J3R / NARRE / DeepCoNN score 0.775–1.00 on 5-core Amazon; Kang et al. 2023 (arXiv
2305.06474) report global-mean 0.948, item-mean 0.934, user-mean 0.853 and a fine-tuned
Flan-T5-XXL at 0.830 on Amazon-Books. Even an oracle that knows the *true item mean* only reaches
~1.33 (the within-item sd), measured on Amazon-2023 Software; an oracle knowing the *true user
mean* reaches ~1.14.

A 0.4–0.5 number appears in practice only under one of these conditions, none of which applies to
`dataset_discribe.txt`:

1. the target is an **item-average** rating (variance shrinks ~1/n), not a per-review rating;
2. **leakage** — dupes across train/test, or user/item means computed over train+test;
3. a **different task** (e.g. MovieLens-style dense per-user ranking, where the best is still 0.78).

I verified there is no accidental leak here: fingerprinting all 10,000 test rows against the
6,104,774 visible train rows gives **21 exact matches (0.21 %)**, indistinguishable from the
background duplicate rate (0.33 %) — the test set is disjoint. So 1.154 is an honest number and
0.4–0.5 is not reachable from the delivered medium by any model.

---

## 4. The redesign that follows from the data, not from model complexity

The binding constraint is *information*, not capacity. The redesign is therefore a medium redesign;
model work only resumes once a richer medium exists.

### Stage 0 — reopen the medium (blocking)
Recover `train.csv` / `test.csv` (or the Part-A `clean_train_latest` / `train_removed_latest`
directory, which still has `title`, `comment`, `user_id`, `prod_id`, `parent_prod_id`). Nothing
downstream can improve until this happens: identity features are the single largest missing
channel, and they cannot be faked from what is committed. If the original Windows machine is
still available, copy `C:\coding\comp5434project\Task2数据交接` (or re-run the Part-A Spark job on
the raw CSV) into this repo.

### Stage 1 — the three feature channels, rebuilt from raw ratings
1. **Text**: replace the 2^18 unigram-only hashing with uni+bigram TF-IDF (≥1 M features) and a
   learned low-rank projection; keep the per-row L2 norm only as one extra numeric column, not as a
   replacement for term frequency.
2. **Identity biases** (this is where the 1.33/1.14 oracle numbers come from): out-of-fold
   user / item / parent-item mean ratings with empirical-Bayes shrinkage toward the global mean,
   plus user-minus-item residuals and review-count priors. Must be OOF and must use only the train
   split; this is also the only channel that can express "this user is lenient" and "this product is
   bad".
3. **Context**: `votes`, `purchased`, review age and item-age at review time (already partially
   present); these are weak alone but interact with item bias.

### Stage 2 — model
A single model that consumes all three channels — linear/wide path over text, embedding-bag over
biases, small MLP head — trained on 6.1 M rows with early stopping on the 1.53 M validation split.
The existing `HybridRatingModel` is already this shape; it simply lacks channel 2.

**Expected outcome on that richer medium: RMSE 0.95–1.05** (text ≈ 1.03, user+item bias ≈ 1.30,
combined 0.95–1.00 by the additivity measured in the literature and in the Software-category probe).
A fine-tuned transformer would add ~0.03–0.05 but is not trainable on the available CPU.

### Stage 3 — calibration only
Ordinal head / isotonic calibration to fix the 5★ mass and improve MAE by ~0.01–0.02. Not a lever.

### What not to do
Do not spend compute on deeper networks, more epochs, extra ensembles or extra dense features on
the current artifacts: the ceiling analysis bounds any such gain at ~0.05 RMSE, and the last two
repo attempts (V2 overlay, ensemble) each produced ≤0.003.

---

## 5. Bottom line

* Source: Amazon Reviews 2023 (McAuley Lab), derived subset — verified on schema, not on row count.
* Best possible on the committed medium: ≈1.05–1.15 RMSE; current 1.154 is at that cap.
* Target 0.4–0.5: impossible (needs R² = 0.91); on a full medium expect 0.95–1.05.
* The only real redesign is to reopen the raw medium (text + ids), then rebuild the three channels.
