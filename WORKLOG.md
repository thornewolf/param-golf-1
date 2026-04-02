# Parameter Golf Worklog

## Session: 2026-04-01

### Goal
Build a competitive submission for the Parameter Golf challenge — 10-minute training on H100 GPUs, 16MB artifact limit, scored on bits-per-byte (BPB). Baseline: 1.2244 BPB. Top leaderboard: 1.1194 BPB.

### Starting Point
- Forked `parameter-golf` repo, created `train_gpt_thorne.py` based on the baseline `train_gpt.py`
- RunPod with 1xH100 initially, later 8xH100

### Techniques Implemented (in order)

**Batch 1 — Core techniques (1-10):**
1. Sliding window evaluation (stride=64, full context per scored token)
2. Int6 quantization (per-row, adaptive clip search across 5 percentiles)
3. QAT with Straight-Through Estimator (late QAT at LR scale < 0.15)
4. Zstandard compression (level 22, replacing zlib)
5. NTK-aware RoPE scaling (extrapolate to longer eval sequences)
6. FP16 embedding passthrough (skip quantization for tok_emb)
7. Stochastic Weight Averaging (available but disabled — EMA works better)
8. Exponential Moving Average (decay=0.985)
9. Multi-Token Prediction (2 auxiliary heads, excluded from export)
10. Bigram Hash Embedding (XOR hash, 2048 vocab, 128 dim)

Plus: SmearGate, orthogonal init with projection scaling, decoupled Muon weight decay, AdamW for embeddings/scalars.

**Batch 2 — Architecture upgrades (11-15):**
11. SmearGate (learned adjacent-token blending)
12. XSA on last 4 layers (exclude self-attention, GQA-aware)
13. Flash Attention 3 (with SDPA fallback, B,T,H,D layout)
14. Partial RoPE (16 of 64 dims rotated, rest position-free)
15. Value Embeddings at layers 9,10 (shared table, per-layer scales)

**Batch 3 — Optimizer and eval-time:**
- Parallel Muon with parameter banking (4 contiguous 3D banks, async reduce-scatter/all-gather)
- Legal Score-First TTT (chunk-sequential SGD adaptation at eval time)

**Custom — EMA Backup:**
- Snap model weights to EMA weights at specified training steps, reset EMA, continue training from smoother basin

### Run History

| Run | GPUs | Config | Sliding BPB | Notes |
|-----|------|--------|-------------|-------|
| v1 | 1x | 2048 seq, 786k batch, warmdown=3500 | N/A | Disk quota error |
| v2 | 1x | 2048 seq, 786k batch, warmdown=1200 | N/A | 745 steps, LR crushed, quant gap 1.6→3.0 |
| v3 | 1x | 1024 seq, 262k batch, warmdown=500 | N/A | 1834 steps, 1.2980 post-int6 |
| v5 | 8x | 1024 seq, 524k batch, no backup | 1.2148 | 10106 steps, sliding window hurt (train/eval mismatch) |
| v6 | 8x | same + backup@4999 | 1.2062 | Backup helped sliding -0.009 |
| v7 | 8x | same + backup@4999,7499 | 1.2094 | Double backup slightly worse than single |
| v8 | 8x | +FA3+XSA+VE+PartialRoPE, no backup | 1.2234 | 1.1893 post-int6, sliding still hurt at 1024 |
| v8b | 8x | same + backup@4999 | 1.2035 | Best 1024 sliding, backup helps |
| v11 | 8x | 2048 seq, MTP=2, backup@3500 | **1.1484** | Big jump from training at 2048 |
| v12 | 8x | same + Parallel Muon banking | **1.1483** | Banking didn't help step time |
| v12-ttt | 8x | +TTT lr=0.002 | N/A | TTT never ran (old commit) |
| v12-ttt2 | 8x | +TTT lr=0.002, correct commit | 1.3500 | TTT destroyed model (RoPE cache bug fixed mid-run) |
| v13 | 8x | +TTT lr=0.0001, epochs=1, freeze=6 | 1.4594 | TTT still destroys model — incompatible with our architecture |
| v14 | 8x | backup@2500,5000,7500, no TTT | 1.1489 | 3 backups slightly worse than 1 |

### Key Findings

1. **Training at 2048 seq_len was the biggest single win** — sliding window went from hurting (-0.03) to helping (-0.024) once train/eval seq lengths matched.

2. **EMA backup at midpoint helps ~0.009 BPB** on sliding window. Single backup > multiple backups.

3. **Parallel Muon banking added no speedup** on 8xH100 — the per-layer all_reduce was already fast enough. Same results, more code complexity.

4. **TTT is incompatible with our model** — SGD adaptation destroys weights even at very low LR. The reference that achieved -0.002 from TTT used LeakyReLU² which has gradients everywhere; our ReLU² has dead neurons that can't recover from SGD perturbation.

5. **Warmdown tuning is critical** — warmdown_iters must be << total steps or LR is suppressed the entire run. Got burned by warmdown=3500 on 745-step runs.

6. **QAT makes quantization lossless or even beneficial** — post-int6 BPB is often better than pre-quant due to regularization effect.

### Final Best Result
```
Sliding window BPB: 1.1483 (v12, 8xH100)
Post-int6 BPB:      1.1720
Post-EMA BPB:       1.1678
Artifact size:      17.3MB (over 16MB limit — needs model shrink for valid submission)
```

### Remaining Gap to SOTA (1.1194)
- 0.029 BPB gap
- Main missing pieces: LeakyReLU² activation, better hyperparameter tuning across many runs
- TTT contributed only -0.002 in the reference — not a big lever
- Artifact size still over 16MB — need to reduce model size (fewer layers or smaller MLP) for a valid submission

### Files
- `train_gpt_thorne.py` — main training script (1996 lines)
- `requirements.txt` — added `zstandard` dependency
- RunPod machines used: `n6pv6dzjx6to4v`, `cee8dcaf0e23`
