# How KV Cache Size Affects Batching and Cost
*and how compression techniques like SVD/quantization help*

---

## Reproducing the Results

All experiments run on `bert-base-uncased` (downloaded automatically from HuggingFace). No GPU required — CPU is sufficient.

**Setup**

```bash
cd compression
pip install -e .
```

**Run the full report** (SVD + INT4 analysis, cost tables, HBM capacity estimates):

```bash
python report.py
```

**Run the SVD compression demo** (plots error vs. rank and error vs. memory):

```bash
python main.py
```

**Run the rank-16 tradeoff script** (single-head, layer 0):

```bash
python -m experiments.compression_tradeoff
```

**Optional — carbon footprint tracking:**

```bash
pip install codecarbon
python report.py   # carbon section auto-enables
```

---

During inference, there are two limiting factors built into the hardware: compute and memory. The compute bottleneck is characterised by the maximum operating speed (FLOPs/s) of a system, while bandwidth is the speed at which the system can pull from memory (bytes/s). When the ratio of the two surpasses ~600 FLOPs/byte, we are "compute bound" and have reached the peak performance, or ridge point, of the hardware. Under this ridge point, we are "bandwidth bound."

<img width="446" height="339" alt="Screenshot 2026-05-26 at 10 34 53 AM" src="https://github.com/user-attachments/assets/412ddea6-4e46-4a60-ab34-64d95cb5d898" />

During prefill (the processing of a user's prompt), stored weights Wq, Wk, Wv, and Wo are pulled from memory once, and many QKV matrices are computed via batching. Batching is the concatenation of many user requests to allow for compute to happen all at once and reduce the time we wait for data to arrive from HBM. Therefore, prefill is compute bound.

**Prefill arithmetic intensity = seq_len FLOPs/byte**

> Note: whether prefill is compute or bandwidth bound depends on the length of the input. At long contexts, batching no longer becomes possible. For example, if a 70B model takes up 140GB of storage just for the weights, it will require 2 H100 GPUs (each 80GB) just to hold the weights.
>
> 160GB of storage − 140GB for weights = 20GB for inference, which is roughly 60k token max_context.
>
> *Ex: Llama 3 70B, 80 layers, 8 KV heads (Grouped Query Attention), d_head = 128, FP16*
> `20 GB / (80 × 8 × 128 × 2 × 2 bytes) = 60k tokens`

Alternatively, in decode (the generation of tokens), the stored K and V matrices must be pulled from memory for each layer for each token.

**Decode arithmetic intensity = 1 FLOPs/byte**

---
## Memory-Error Tradeoff
<img width="635" height="474" alt="Screenshot 2026-03-14 at 4 27 38 PM" src="https://github.com/user-attachments/assets/3ee02b2e-8218-4862-a56a-c4e27a11906b" />

As expected, error increases as we use less storage for singular value approximation.

Reconstruction error increases monotonically as rank r decreases, consistent with the **Eckart–Young theorem**. This curve defines the Pareto frontier between compression ratio and approximation quality, and can be used to select a rank budget given an acceptable error threshold.

<img width="330" height="329" alt="Screenshot 2026-05-26 at 10 35 38 AM" src="https://github.com/user-attachments/assets/4e76d29b-82f9-44b1-9bc6-1f11002025df" />

---
## $/Mtok — Cost per Million Tokens

```
$/Mtok = GPU cost per hour / tokens generated per hour
```

While both prefill and decode have cost, decode (generation) is usually much more expensive. Decode is memory-bound, so throughput is determined by bandwidth (bytes/s).

```
tokens/s = (bytes/s) / model_size_bytes
At 3.35 TB/s and 140GB model => 24 tokens/s
H100 spot price = $2.50/hr
$28.90/Mtok generated (for one batch)
```

Therefore, cost is inversely proportional to bandwidth. However, it is also highly dependent on batching.

Batching reduces the time waiting for data from memory. As we know, decode is memory bound, so batching inversely affects cost.

| Batch size | Cost |
|---|---|
| 1 | $28.90/Mtok |
| 10 | $2.89/Mtok |
| 100 | $0.28/Mtok |

**Batching is the determinant of economics ($/Mtok). KV cache reduces capacity to batch.**

---

## KV Cache Memory

Full KV cache takes up:

```
n_layers × seq_len × model_dim × 2 (per K and V) × 2 bytes (in FP16)
```

Researchers have noticed that K and V matrices encode redundant information and can be compressed without significantly compromising performance.

---

## Singular Value Decomposition

### Problem
Inference cost in transformers models O(n^2). KV cache compression reduces this quadratic growth -> linear, but not all dimensions in the stored KV matrices are equally significant. We can prune the insignificant directions to reduce storage. 
full-rank are either given higher rank or ignored entirely.

Singular Value Decomposition is a method to rewrite a matrix as a product of 3 matrices: a rotational matrix, a scaling matrix, and another rotational matrix to reverse the first one.

```
M = U @ diag(S) @ Vᵀ
```

Where M is (s × d), s = seq_len, d = model_dim:

| Matrix | Shape |
|---|---|
| U | s × s |
| S | s × d |
| V | d × d |

S is a diagonal matrix, where the diagonals are descending singular values. When we take the top r singular values in S, we prune all singular values after r, and notice that performance does not meaningfully degrade.

After truncation to rank r:

| Matrix | Shape |
|---|---|
| M | r × d |
| U | s × r |
| S | r × r |
| V | r × d |

We initially had a full rank diagonal matrix and 2 large square matrices. Now, we have a small diagonal matrix and 2 thin matrices.

```
Store U×S as an (s × r) matrix  =  s × r × 2 bytes
Store V as an (r × d) matrix    =  r × d × 2 bytes
Total                           =  r(s + d) × 2 bytes
```
### Decay of Singular Values
<img width="634" height="476" alt="Screenshot 2026-03-18 at 6 43 39 PM" src="https://github.com/user-attachments/assets/1ea24226-a0e1-4e77-9e72-cc029bb960d0" />
The decay profile determines compressibility:

Exponential decay → energy is concentrated in a few directions; aggressive rank reduction is safe
Linear decay → singular values are roughly equally significant; the matrix is near full-rank and compression will incur higher error

In this project, we used adaptive rank, which takes the highest r such that the error between full rank and adapted rank is <10%.

```
Full KV           =  2 × s × d × 2 bytes           per layer
Adapted rank SVD  =  2 × r × (s + d) × 2 bytes     per layer

Ratio = (2 × s × d × 2) / (2 × r × (s+d) × 2)
      = s × d / (r × (s+d))

If seq_len = 182, model_dim = 64:
  = 11648 / (r × 246)
```

### When does SVD stop being helpful?

When `r × (s+d) > s × d`, i.e. when `r > s×d / (s+d)`.

In this model, at rank 64, SVD memory exceeds full KV memory.

### Results by layer

| Layer | Rank r | Full (KB) | SVD (KB) | Ratio | Attn error |
|---|---|---|---|---|---|
| 0 | 8 | 1092.0 | 185.2 | 5.9× | 9.93% |
| 1 | 16 | 1092.0 | 370.5 | 2.9× | 2.56% |
| 2 | 32 | 1092.0 | 741.0 | 1.5× | 4.45% |
| 3 | 16 | 1092.0 | 370.5 | 2.9× | 5.65% |
| 4 | 16 | 1092.0 | 370.5 | 2.9× | 4.87% |
| 5 | 32 | 1092.0 | 741.0 | 1.5× | 6.68% |
| 6 | 32 | 1092.0 | 741.0 | 1.5× | 5.36% |
| 7 | 32 | 1092.0 | 741.0 | 1.5× | 7.55% |
| 8 | 64 | 1092.0 | 1482.0 | 0.7× | 0.00% |
| 9 | 16 | 1092.0 | 370.5 | 2.9× | 9.20% |
| 10 | 32 | 1092.0 | 741.0 | 1.5× | 3.93% |
| 11 | 64 | 1092.0 | 1482.0 | 0.7× | 0.00% |

### Totals (all 12 layers · all 12 heads)

| | |
|---|---|
| Full KV | 13,104.0 KB → 73,728.0 bytes/token |
| SVD KV | 8,336.2 KB → 46,902.9 bytes/token |
| Ratio | 1.57× |
| Reduction | 26,825.1 bytes/token (36.4%) |
| Avg error | 5.02% (relative ‖Δattn‖ / ‖attn‖₂) |

### Cost per 1B tokens — SVD vs. full (memory-bandwidth model)

| GPU | BW (TB/s) | $/hr | Full $/1B | SVD $/1B | Saved $/1B | Saving % |
|---|---|---|---|---|---|---|
| H100 SXM5 | 3.35 | $3.00 | $0.0183 | $0.0117 | $0.0067 | 36.6% |
| A100 SXM4 | 2.00 | $2.00 | $0.0205 | $0.0130 | $0.0075 | 36.5% |
| A10G | 0.60 | $0.75 | $0.0256 | $0.0163 | $0.0093 | 36.3% |

> **Note:** Error was calculated as the difference between the full rank matrix and SVD matrix, but downstream performance was not tested.
>
> Layer 1 showed 2.56% error, while layer 9 showed 9.20%. Certain layers are more sensitive to loss of data.

---

## 4-bit Quantization

In FP16, each weight and stored value takes up about 2 bytes of memory. By taking FP16 down to INT4, we reduce each element's storage by 4×, down to 0.5 bytes.

### Storage formulas

```
float16  :  2.0000  bytes / element
int4     :  0.5000  bytes / element  (4 bits, packed 2-per-byte)
           + 2 / 128  bytes / element  (one float16 scale per group)
          =  0.5156  effective bytes / element

Ratio    =  2 / 0.5156  =  3.88×
```

### Error by layer

| Layer | INT4 error | SVD error |
|---|---|---|
| 0 | 9.23% | 9.93% |
| 1 | 7.28% | 2.56% |
| 2 | 16.82% | 4.45% |
| 3 | 7.29% | 5.65% |
| 4 | 7.87% | 4.87% |
| 5 | 29.94% | 6.68% |
| 6 | 21.43% | 5.36% |
| 7 | 36.78% | 7.55% |
| 8 | 60.35% | 0.00% |
| 9 | 8.88% | 9.20% |
| 10 | 9.90% | 3.93% |
| 11 | 44.97% | 0.00% |

Layer 8 has 60% error and layer 11 has 44% when quantized. We can conclude that INT4 quantization affects loss more significantly than adapted rank SVD (error <10%).

### Totals

| | |
|---|---|
| Full KV | 13,104.0 KB → 73,728.0 bytes/token |
| INT4 KV | 3,379.0 KB → 18,874.0 bytes/token |
| Ratio | 3.88× |
| Reduction | 54,854.0 bytes/token (74.4%) |
| Avg error | 21.73% (relative) |

### Cost per 1B tokens — INT4 vs. full (memory-bandwidth model)

| GPU | BW (TB/s) | $/hr | Full $/1B | INT4 $/1B | Saved $/1B | Saving % |
|---|---|---|---|---|---|---|
| H100 SXM5 | 3.35 | $3.00 | $0.0183 | $0.0047 | $0.0136 | 74.4% |
| A100 SXM4 | 2.00 | $2.00 | $0.0205 | $0.0053 | $0.0152 | 74.4% |
| A10G | 0.60 | $0.75 | $0.0256 | $0.0066 | $0.0190 | 74.4% |

---

## SVD + 4-bit Quantization

Stack SVD (reduce rank) then INT4 (quantize the low-rank factors):

```
ratio_combined  =  ratio_svd × ratio_int4
                =  1.57 × 3.88  =  6.09×
```

### Summary

| Method | Total KV | Bytes/token | Ratio | Avg error |
|---|---|---|---|---|
| Full (baseline) | 13,104.0 KB | 73,728.0 B | 1.00× | — |
| SVD only | 8,336.2 KB | 46,902.9 B | 1.57× | 5.02% |
| INT4 only | 3,379.0 KB | 18,874.0 B | 3.88× | 21.73% |
| SVD + INT4 | 2,150.8 KB | 12,096.2 B | 6.09× | SVD: 5.02% · INT4: 21.73% (independent measurements) |

### Cost per 1B tokens — memory-bandwidth model

`cost = bytes/token × price_per_hr / (bandwidth × 3600)`

| GPU | BW (TB/s) | $/hr | Full $/1B | SVD $/1B | INT4 $/1B | Comb $/1B | Max saving |
|---|---|---|---|---|---|---|---|
| H100 SXM5 | 3.35 | $3.00 | $0.0183 | $0.0117 | $0.0047 | $0.0030 | $0.0153 (84%) |
| A100 SXM4 | 2.00 | $2.00 | $0.0205 | $0.0130 | $0.0053 | $0.0034 | $0.0171 (83%) |
| A10G | 0.60 | $0.75 | $0.0256 | $0.0163 | $0.0066 | $0.0042 | $0.0214 (84%) |

---

## Compression vs. HBM Capacity

When compressed, how much memory is left for batching?

**H100 SXM5, 80 GB HBM**
Assumption: 50% of 80 GB HBM allocated to KV cache → KV budget: 40 GB

### Max concurrent sequences (seq_len = 182)

| Method | Sequences | vs. baseline |
|---|---|---|
| Full | 2,980 | 1.0× |
| SVD | 4,685 | 1.6× |
| INT4 | 11,560 | 3.9× |
| SVD + INT4 | 18,181 | 6.1× |

With SVD+INT4, you are freeing up capacity for 6.1× as many batches, which reduces cost by a factor of 6.1. However, the error from the full KV is significant across several layers, which can be harmful to the quality of the output. It may also not have a significant effect on the performance from a user's perspective.

A good tradeoff is Adaptive SVD, which returns ~5% error, but frees up 1.6× capacity. Coupling this with 8-bit quantization (yet to be tested) could be optimal.

---

## Next Steps

1. Verify that high error, or deviating from full KV, is actually harmful to downstream language tasks.
2. 8-bit quantization could be a good compromise between INT4 and no quantization — half the compression power of INT4, but with less error.
