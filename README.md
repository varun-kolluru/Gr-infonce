# Gr-InfoNCE: Using Explicit Graded Feedback in Contrastive Loss for Recommender Systems

Code for **Graded-InfoNCE (Gr-InfoNCE)**, a weighted InfoNCE loss for two-tower retrieval that uses the explicit rating of each user–item pair as its weight. Positives are weighted by the normalized rating `W`, and in-batch negatives by `1/W`. The loss adds no parameters, no adversarial loop and no extra temperature.

On MovieLens-1M, Gr-InfoNCE raises Hit@20 from 6.32% (best baseline) to 10.50% and NDCG@20 from 2.29% to 4.10%. It also gives the best Hit@20 on Yahoo! R3 and Coat, and reaches its best validation accuracy in fewer epochs than every baseline. A light MLP reranker on the frozen embeddings lifts MovieLens Hit@20 further to 16.52%.

> **Naming note.** In the notebooks the proposed loss is called `MyLoss` (function `weighted_clip_loss`), and an earlier name, *SC-InfoNCE*, still appears in some comments. All of these refer to **Gr-InfoNCE** in the paper.

---

## Method in brief

For a batch of `B` observed positive pairs, let `s_ij = u_i · v_j / τ` (τ = 0.07, L2-normalized embeddings). The weight for every pair is looked up from a rating table built from the training data:

```
W_ij = clip( R[u_i, v_j] / r_max , ε , 1 ),   ε = 1e-2   (R = 0 if the pair was never rated)
```

Implemented in log-additive form, followed by standard cross-entropy over the batch:

```
positive logit:  s_ii + log W_ii
negative logits: s_ij − log W_ij        (i.e. weight 1 / W_ij)

L = ½ ( CE(rows) + CE(columns) )         # symmetric user-side / item-side loss
```

A negative's weight falls from `1/ε = 100` (never rated) to `1` (maximum rating).

**Two-stage pipeline**

1. **Stage 1 – two-tower retrieval.** Each tower concatenates side features with a 64-d ID embedding and passes them through an MLP (256 → 128, LayerNorm + ReLU) to a 256-d L2-normalized output. Scoring is by dot product over all unseen items.
2. **Stage 2 – light reranking head (MovieLens only).** With the Stage-1 embeddings frozen, a Linear or MLP head reranks each user's top-100 candidates. Its input is `[h, v, h ⊙ v, hᵀv]`, where `h` is a recency-decayed average (γ = 0.9) of the user's history embeddings. The head is trained with binary cross-entropy (5 random negatives per positive).

---

## Repository contents

| Notebook | What it does |
|---|---|
| `movielens-earlystopping.ipynb` | MovieLens-1M: Stage-1 comparison (Random / BCE / InfoNCE / AdvInfoNCE / Gr-InfoNCE) **and** the Stage-2 Linear / MLP reranking heads |
| `yahoo-earlystopping.ipynb` | Yahoo! R3: Stage-1 comparison |
| `coat-earlystopping.ipynb` | Coat: Stage-1 comparison |
| `ablations-movielens.ipynb` | Negative-weighting ablation on MovieLens (`linear`, `recip`, `exp_linear`, `exp_recip`) |
| `ablations-yahoo.ipynb` | Negative-weighting ablation on Yahoo! R3 |
| `ablations-coat.ipynb` | Negative-weighting ablation on Coat |

The ablation variants replace the negative weight `g(W)` with `1 − W` (`linear`), `1 / W` (`recip`, **the Gr-InfoNCE formula**), `e^(1−W)` (`exp_linear`) and `e^(1/W)` (`exp_recip`, which collapses to NaN loss).

---

## Datasets

| Dataset | Files used | Source |
|---|---|---|
| **MovieLens-1M** | `ratings.dat`, `users.dat`, `movies.dat` | [GroupLens](https://grouplens.org/datasets/movielens/1m/) |
| **Yahoo! R3** | `user.txt` (train, MNAR), `random.txt` (randomly exposed) | [Yahoo! Webscope R3](https://webscope.sandbox.yahoo.com/) |
| **Coat** | `train.csv`, `random.csv`, `user_feat_onehot.csv`, `item_feat_onehot.csv` | [Schnabel et al., 2016](https://www.cs.cornell.edu/~schnabts/mnar/) |

Interactions with rating ≥ 4 count as positives, and ratings are divided by the dataset maximum (5) to give `W`.

**Splits**

- **MovieLens:** per-user leave-one-out by timestamp. The latest interaction is the test item, the second latest is the validation item, and everything earlier is training data. User features are gender, age bucket and occupation; item features are genres.
- **Yahoo! R3 / Coat:** the biased training file is used in full for training. The randomly exposed file is split once, with a fixed seed (`SPLIT_SEED = 2026`), into 20% validation (early stopping) and 80% test. The split is identical across all methods and seeds.

---

## Setup

```bash
pip install torch numpy pandas matplotlib jupyter
```

A GPU is recommended for MovieLens (about 30 s per epoch in the saved logs). Coat runs in well under a minute even on CPU.

The notebooks were developed and run on **Kaggle**, so the data paths are hard-coded to `/kaggle/input/...`. To run them elsewhere, download the datasets and edit the path constants:

| Notebook | Where to change paths |
|---|---|
| MovieLens notebooks | the three `pd.read_csv(...)` calls inside `load_movielens()` |
| Yahoo notebooks | the two `pd.read_csv(...)` calls inside `load_yahoo()` |
| Coat notebooks | the `COAT_DIR` constant |

## Running

Open a notebook and run all cells in order. Each notebook is self-contained.

- **Do not skip or reorder cells in `movielens-earlystopping.ipynb`.** The Stage-2 cell reuses the trained Gr-InfoNCE encoders (`myloss_artifacts`) kept in memory from Stage 1, so nothing is retrained.
- Every method is run with seeds **42, 43, 44**, and results are reported as mean ± std over the three seeds.
- Saved outputs:

| Notebook | Output |
|---|---|
| `movielens-earlystopping` | `movielens_stage1_hit20_results_mean_std_earlystop.csv`, `movielens_stage2_hit20_ndcg20_results.csv` |
| `yahoo-earlystopping` | `yahoo_stage1_hit20_results_mean_std_earlystop.csv` |
| `coat-earlystopping` | `coat_stage1_hit20_results_mean_std_earlystop.csv` |
| `ablations-movielens` | `ablation_negweight_movielens_hit20_results.csv` (+ bar plot) |
| `ablations-yahoo` | `ablation_negweight_hit20_results.csv` (+ bar plot) |
| `ablations-coat` | `coat_ablation_negweight_hit20_results.csv` (+ bar plot) |

In the saved Kaggle runs, MovieLens Stage 1 took about 1 h 45 min and Stage 2 about 20 min. Yahoo! R3 took about 30 min and Coat about half a minute.

---

## Hyperparameters

| | Stage 1 | Stage 2 (MovieLens) |
|---|---|---|
| Embedding / ID dim | 256 / 64 | – |
| Hidden sizes | (256, 128) | MLP head (128, 64) |
| Temperature τ | 0.07 | – |
| Batch size | 128 | 512 |
| Optimizer | AdamW, lr 1e-3, wd 1e-2, cosine annealing | Adam, lr 1e-3, wd 1e-4 |
| Gradient clipping | 5.0 | – |
| Max epochs / patience | 40 / 7 (early stopping on validation Hit@20) | 40 / 7 |
| Weight floor ε | 1e-2 | – |
| Other | – | 100 candidates, decay γ = 0.9, 5 negatives per positive, ≤ 10 steps per user |

All trainable losses share the same backbone, optimizer and schedule, so differences come only from the loss. AdvInfoNCE uses its adversarial settings `lr = 5e-5`, `T_adv = 5`, `E_adv = 200`, `K = 64`.

**Evaluation:** all-ranking protocol. Every unseen item is ranked for each user, with training items masked out. Hit@20 and NDCG@20 are averaged over users and reported at each seed's best-validation epoch.

---

## Results

Stage 1, all-ranking, Hit@20 / NDCG@20 in % (mean ± std over 3 seeds; best per column in bold):

| Method | MovieLens H@20 | MovieLens N@20 | Yahoo! R3 H@20 | Yahoo! R3 N@20 | Coat H@20 | Coat N@20 |
|---|---|---|---|---|---|---|
| Random | 0.54 ± 0.12 | 0.17 ± 0.05 | 3.43 ± 0.25 | 0.82 ± 0.02 | 17.34 ± 1.37 | 3.02 ± 0.03 |
| BCE | 5.53 ± 0.22 | 2.03 ± 0.10 | 13.21 ± 1.60 | 3.81 ± 0.53 | 22.12 ± 3.73 | 3.89 ± 0.91 |
| InfoNCE | 6.21 ± 0.45 | 2.26 ± 0.13 | 13.45 ± 0.59 | 3.95 ± 0.11 | 21.52 ± 0.90 | 3.87 ± 0.48 |
| AdvInfoNCE | 6.32 ± 0.42 | 2.29 ± 0.12 | 13.73 ± 0.24 | 3.95 ± 0.10 | 20.63 ± 0.78 | **4.30 ± 0.08** |
| **Gr-InfoNCE** | **10.50 ± 0.26** | **4.10 ± 0.08** | **14.07 ± 0.82** | **4.39 ± 0.25** | **22.87 ± 1.19** | 3.96 ± 0.35 |

On Yahoo! R3 and Coat the Hit@20 gains are within about one standard deviation, so they are suggestive rather than conclusive.

Mean best epoch (epoch at which best validation accuracy is reached; lower is faster): Gr-InfoNCE reaches it in **4.3 / 7.3 / 1.0** epochs on MovieLens / Yahoo! R3 / Coat, versus 10.7–11.0 / 12.3–25.3 / 2.0–4.7 for the baselines.

Stage 2 on MovieLens (frozen Gr-InfoNCE embeddings, top-100 candidates reranked):

| Model | Hit@20 | NDCG@20 |
|---|---|---|
| Gr-InfoNCE backbone only | 10.50 ± 0.26 | 4.10 ± 0.08 |
| + Linear head | 15.17 ± 1.45 | 6.35 ± 0.50 |
| + MLP head | **16.52 ± 0.82** | **6.90 ± 0.26** |

---

## Citation

The paper is currently under review. A citation entry will be added after the review period.
