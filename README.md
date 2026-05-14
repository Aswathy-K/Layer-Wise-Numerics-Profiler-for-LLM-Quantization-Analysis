# Layer-wise Numerics Profiler for LLM Quantization Analysis

A PyTorch hook-based tool that instruments every linear layer in a HuggingFace transformer model to capture weight and activation statistics during inference. Identifies quantization-sensitive layers before committing to a compression recipe — replacing the typical compress-then-eval loop with a principled, data-driven layer ranking.

---

## Motivation

Standard quantization evaluation measures perplexity on WikiText-2 after compression. This tells you *whether* a recipe degraded the model, but not *which layers* caused the degradation or *why*. This profiler surfaces the layer-level signal — outlier distributions, kurtosis, dynamic range — that makes it possible to design targeted mixed-precision recipes rather than applying uniform quantization across all layers.

---

## Features

- Hooks into every `nn.Linear` layer (all attention and FFN projections) via `register_forward_hook`
- Measures 7 statistics per layer for both weights and input activations
- Supports float, INT8, and INT4 (bitsandbytes) model loading
- Accepts custom SFT calibration data or falls back to WikiText-2
- `show_summary` — heatmap of top-N most sensitive layers
- `show_diff` — delta table comparing float vs quantized, flagging degraded layers
- Works fully in-memory in Colab — no JSON files required

---

## Installation

```bash
pip install torch transformers datasets bitsandbytes>=0.46.1
```

---

## Usage

```python
# Profile float model
stats_float = profile_model(
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_samples  = 32,
)

# Profile INT8 quantized model
stats_int8 = profile_model(
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantize   = "int8",
    n_samples  = 32,
)

# Visualise top-15 sensitive layers as a heatmap
df = show_summary(stats_float)

# Diff float vs INT8 — highlights degraded layers
delta = show_diff(stats_float, stats_int8, label_a="float", label_b="int8")
```

To use your own SFT calibration data (e.g. from [build_calib_dataset.py](build_calib_dataset.py)):

```python
stats = profile_model(
    model_name   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    calib_jsonl  = "calib_sft.jsonl",
    n_samples    = 64,
)
```

---

## Metrics

Each layer is profiled across both its **weight tensor** and **input activations**:

| Metric | Description | Why it matters for quantization |
|---|---|---|
| `outlier_pct` | % of values beyond 3σ from mean | Directly measures what breaks naive INT8 |
| `kurtosis` | Tail heaviness of the distribution | Values >> 3 indicate spike artifacts |
| `dynamic_range` | abs_max / std | How much quantization range outliers consume |
| `abs_max` | Maximum absolute value | Sets the clipping threshold for symmetric quant |
| `std` | Standard deviation | Spread of the useful signal |
| `mean` | Mean value | Checks for distribution shift post-quantization |

---

## Findings: TinyLlama-1.1B

### INT8 vs Float

- Activation `outlier_pct` was stable across all layers (delta < 0.05pp) — TinyLlama tolerates INT8 well
- All 15 most sensitive layers were `self_attn.o_proj`; no FFN layers appeared — consistent with the LLM.int8() paper's finding that outlier features concentrate in attention output projections
- Kurtosis dropped slightly after INT8 across all layers, explained by bitsandbytes' mixed-precision decomposition handling large outlier values in FP16 before they reach the residual stream
- `layers.0` weight kurtosis dropped dramatically (17.77 → 5.84), indicating an anomalous float weight distribution that INT8 effectively regularized

### INT4 vs Float

- `mlp.down_proj` layers appeared in the top sensitivity ranking for the first time — entirely absent from the INT8 analysis
- `layers.1.mlp.down_proj` showed a kurtosis delta of **+400.75** and dynamic range delta of **+20.9** — the largest changes in the entire model, indicating INT4 creates extreme spike artifacts in FFN output activations not present in float
- `q_proj`, `k_proj`, `v_proj` in layer 2 all showed identical deltas (kurtosis +59.6, dynamic range +0.8), pointing to a corrupted residual from `layers.1.mlp.down_proj` propagating upstream rather than independent projection degradation
- Standard `outlier_pct` threshold missed all INT4 degradation; **kurtosis and dynamic range were the informative signals**, motivating a composite degradation flag:

```python
delta["degraded"] = (delta["d_act_kurtosis"].abs() > 50) | \
                    (delta["d_act_dynamic_range"].abs() > 10)
```

### Mixed-precision recipe derived from findings

| Layers | Recommended precision |
|---|---|
| All layers | W4 (INT4) |
| `layers.0–2.mlp.down_proj` | W8 (INT8) — keep in higher precision |

Keeping only 3 layers at W8 recovers most of the INT4 accuracy loss at less than 2% size overhead — a targeted decision that is not visible from perplexity numbers alone.

---

## How It Works

### Forward hooks

`register_forward_hook` fires after every linear layer's forward pass. `input[0]` is the activation tensor entering the layer (shape `[batch, seq_len, hidden_dim]`) — what quantization actually sees at inference time. Weight stats are read directly from `module.weight`.

```python
def hook(module, inp, out):
    activation = inp[0]
    stats = _compute_stats(activation)
    self._stats[name]._act_buffer.append(stats)
```

Activation stats are averaged across all calibration forward passes to produce stable estimates.

### Calibration data

The profiler accepts either WikiText-2 (default) or a custom JSONL with `prompt`/`response` fields. The choice of calibration data affects which neurons are activated and at what magnitudes — see [build_calib_dataset.py](build_calib_dataset.py) for the companion pipeline that constructs domain-balanced SFT calibration sets.

---

## Files

```
layer_profiler.py        # profiler, hook logic, display functions
build_calib_dataset.py   # SFT calibration dataset builder (Track 2)
```

---

## References

- Dettmers et al. (2022) — [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- Dettmers et al. (2023) — [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078)
- Frantar et al. (2022) — [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- Lin et al. (2023) — [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
