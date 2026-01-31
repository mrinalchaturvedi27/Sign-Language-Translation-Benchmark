# Open-Source LLMs for Sign Language Translation 🚀

**Now supporting decoder-only LLMs: Qwen, Gemma, Llama, Mistral, Phi, and more!**

## 🎯 Why Use Open-Source LLMs?

✅ **State-of-the-art quality** - Modern 7B+ models outperform T5-Large
✅ **Multilingual** - Many models support 100+ languages
✅ **Active development** - Constantly improving
✅ **Flexible** - Can be instruction-tuned, chat-tuned, etc.
✅ **Cost-effective with LoRA** - Fine-tune on consumer GPUs

---

## 📦 Supported Open-Source Models

### Recommended Models (Best Quality-to-Speed Ratio)

| Model | Size | Languages | Memory (4-bit) | Quality |
|-------|------|-----------|----------------|---------|
| **Qwen2-7B** | 7B | Multilingual | ~6GB | ⭐⭐⭐⭐⭐ |
| **Gemma-7B** | 7B | English-focused | ~6GB | ⭐⭐⭐⭐ |
| **Mistral-7B** | 7B | Multilingual | ~6GB | ⭐⭐⭐⭐⭐ |
| **Llama-3.1-8B** | 8B | Multilingual | ~7GB | ⭐⭐⭐⭐⭐ |

### All Supported Models

<details>
<summary><b>Qwen Family (Alibaba)</b></summary>

| Model | Size | Best For |
|-------|------|----------|
| `Qwen/Qwen2-1.5B` | 1.5B | Fast experiments |
| `Qwen/Qwen2-7B` | 7B | **Recommended** |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Instruction-following |
| `Qwen/Qwen2-14B` | 14B | Best quality |
| `Qwen/Qwen2-72B` | 72B | SOTA (needs multi-GPU) |

**Pros:** Excellent multilingual support, very fast inference, great quality
**Cons:** None!

**Usage:**
```yaml
model:
  name: "Qwen/Qwen2-7B"
  tokenizer: "Qwen/Qwen2-7B"
  use_lora: true
```
</details>

<details>
<summary><b>Gemma Family (Google)</b></summary>

| Model | Size | Best For |
|-------|------|----------|
| `google/gemma-2b` | 2B | Fast experiments |
| `google/gemma-7b` | 7B | **Recommended** |
| `google/gemma-2-9b` | 9B | Better quality |
| `google/gemma-2-27b` | 27B | Best quality |

**Pros:** Excellent English performance, well-optimized
**Cons:** Less strong on multilingual

**Usage:**
```yaml
model:
  name: "google/gemma-7b"
  tokenizer: "google/gemma-7b"
  use_lora: true
```
</details>

<details>
<summary><b>Llama Family (Meta)</b></summary>

| Model | Size | Best For |
|-------|------|----------|
| `meta-llama/Llama-3.2-1B` | 1B | Mobile/Edge |
| `meta-llama/Llama-3.2-3B` | 3B | Fast experiments |
| `meta-llama/Llama-3.1-8B` | 8B | **Recommended** |
| `meta-llama/Llama-3.1-70B` | 70B | SOTA (multi-GPU) |

**Note:** Requires HuggingFace access token (gated models)

**Pros:** Industry standard, excellent quality, great community
**Cons:** Requires HF token, slightly larger than competitors

**Usage:**
```yaml
model:
  name: "meta-llama/Llama-3.1-8B"
  tokenizer: "meta-llama/Llama-3.1-8B"
  use_lora: true
  params:
    use_auth_token: true  # Set HF_TOKEN env variable
```
</details>

<details>
<summary><b>Mistral Family (Mistral AI)</b></summary>

| Model | Size | Best For |
|-------|------|----------|
| `mistralai/Mistral-7B-v0.3` | 7B | **Recommended** |
| `mistralai/Mixtral-8x7B` | 46B | MoE, best quality |

**Pros:** Very strong performance, efficient architecture
**Cons:** Fewer model sizes

**Usage:**
```yaml
model:
  name: "mistralai/Mistral-7B-v0.3"
  tokenizer: "mistralai/Mistral-7B-v0.3"
  use_lora: true
```
</details>

<details>
<summary><b>Phi Family (Microsoft)</b></summary>

| Model | Size | Best For |
|-------|------|----------|
| `microsoft/phi-2` | 2.7B | Fast experiments |
| `microsoft/phi-3-mini` | 3.8B | **Recommended for small** |
| `microsoft/phi-3-medium` | 14B | Better quality |

**Pros:** Very efficient, good quality for size
**Cons:** Smaller context window

**Usage:**
```yaml
model:
  name: "microsoft/phi-3-mini"
  tokenizer: "microsoft/phi-3-mini"
  use_lora: true
```
</details>

---

## 🚀 Quick Start

### 1. Choose Your Model

For most users, we recommend:
- **Best overall:** `Qwen/Qwen2-7B`
- **Best English:** `google/gemma-7b`
- **Industry standard:** `meta-llama/Llama-3.1-8B`

### 2. Use the Pre-made Config

```bash
# Qwen2-7B
bash train_multi_gpu.sh configs/qwen2_7b_isign.yaml 4

# Gemma-7B
bash train_multi_gpu.sh configs/gemma_7b_isign.yaml 4

# Llama-3.1-8B (requires HF token)
export HF_TOKEN="your_huggingface_token"
bash train_multi_gpu.sh configs/llama3.1_8b_isign.yaml 4

# Mistral-7B
bash train_multi_gpu.sh configs/mistral_7b_isign.yaml 4
```

---

## 💾 Memory Requirements

### Without Quantization (FP16 + LoRA)

| Model Size | Min GPU Memory | Recommended GPUs |
|------------|----------------|------------------|
| 1-3B | 16GB | 1x RTX 4090 |
| 7-8B | 32GB | 1x A100 or 2x RTX 4090 |
| 13-14B | 48GB | 2x A100 |
| 70B+ | 160GB+ | 4x A100 80GB |

### With 4-bit Quantization + LoRA

| Model Size | Min GPU Memory | Recommended GPUs |
|------------|----------------|------------------|
| 1-3B | 6GB | 1x RTX 3060 |
| 7-8B | 10GB | 1x RTX 3080 |
| 13-14B | 16GB | 1x RTX 4090 |
| 70B+ | 48GB | 2x A100 |

---

## ⚙️ Configuration Guide

### Basic Configuration

```yaml
model:
  name: "Qwen/Qwen2-7B"  # Any decoder-only model
  tokenizer: "Qwen/Qwen2-7B"
  
  use_lora: true  # REQUIRED for 7B+ models
  lora_config:
    r: 16  # Rank (higher = more capacity, slower)
    lora_alpha: 32  # Scaling factor (usually 2x r)
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout: 0.1
```

### With 4-bit Quantization (Memory Efficient)

```yaml
model:
  name: "Qwen/Qwen2-7B"
  tokenizer: "Qwen/Qwen2-7B"
  
  use_lora: true
  load_in_4bit: true  # 4-bit quantization
  
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Advanced LoRA Settings

```yaml
model:
  use_lora: true
  lora_config:
    r: 32  # Higher rank for better quality
    lora_alpha: 64
    # Target ALL attention and MLP layers
    target_modules: [
      "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
      "gate_proj", "up_proj", "down_proj"       # MLP (Llama/Mistral)
    ]
    lora_dropout: 0.05  # Lower dropout for larger models
```

---

## 🎓 Best Practices

### LoRA Hyperparameters

| Parameter | Small Models (1-3B) | Medium (7-8B) | Large (13B+) |
|-----------|---------------------|---------------|--------------|
| `r` | 8-16 | 16-32 | 32-64 |
| `lora_alpha` | 16-32 | 32-64 | 64-128 |
| `lora_dropout` | 0.1 | 0.05-0.1 | 0.05 |
| `target_modules` | q,v | q,k,v,o | All attention + MLP |

### Training Hyperparameters

| Parameter | Small Models | Medium | Large |
|-----------|-------------|--------|-------|
| Learning Rate | 3e-4 | 2e-4 | 1e-4 |
| Batch Size | 8-16 | 4-8 | 2-4 |
| Grad Accum | 2-4 | 4-8 | 8-16 |
| Warmup Ratio | 0.1 | 0.1 | 0.05 |

### Quantization Decision Tree

```
Do you have 40GB+ GPU memory?
├─ Yes → Use FP16 (no quantization)
└─ No → Use 4-bit quantization
    ├─ 24GB+ → load_in_8bit: true
    └─ <24GB → load_in_4bit: true
```

---

## 📊 Performance Comparison

Based on iSign dataset (118k samples):

| Model | BLEU-4 | Training Time (4xA100) | Memory |
|-------|--------|------------------------|--------|
| T5-Base | 3-5 | 12h | 12GB |
| T5-Large | 4-6 | 24h | 24GB |
| **Qwen2-7B (LoRA)** | **6-8** | **18h** | **10GB (4-bit)** |
| **Gemma-7B (LoRA)** | **5-7** | **20h** | **10GB (4-bit)** |
| **Llama-3.1-8B (LoRA)** | **6-8** | **22h** | **12GB (4-bit)** |
| **Mistral-7B (LoRA)** | **6-8** | **20h** | **10GB (4-bit)** |

*These are estimates - actual results depend on your data and hyperparameters*

---

## 🔧 Troubleshooting

### "OutOfMemoryError"

**Solution 1:** Enable 4-bit quantization
```yaml
model:
  load_in_4bit: true
```

**Solution 2:** Reduce batch size
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 16  # Keep effective batch same
```

**Solution 3:** Reduce LoRA rank
```yaml
lora_config:
  r: 8  # Instead of 16
```

### "Model download is slow"

Set HuggingFace cache:
```bash
export HF_HOME="/path/to/large/storage"
```

### "Access denied for Llama models"

1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B
2. Request access
3. Create HF token: https://huggingface.co/settings/tokens
4. Set token:
```bash
export HF_TOKEN="your_token_here"
```

### "bitsandbytes not working"

Install from source:
```bash
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir
```

---

## 💡 Pro Tips

1. **Start with 4-bit + LoRA** - Best memory/quality tradeoff
2. **Use Qwen2-7B** - Excellent quality, fast, no access restrictions
3. **Monitor WandB** - Track experiments for all models
4. **Experiment with target_modules** - More modules = better quality but slower
5. **Save merged model** - Merge LoRA weights after training for deployment

---

## 🔄 Merging LoRA Weights (For Deployment)

After training with LoRA, merge weights for inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "checkpoints/qwen2_7b_isign/best_model")

# Merge
merged_model = model.merge_and_unload()

# Save
merged_model.save_pretrained("qwen2_7b_sign_language_merged")
tokenizer.save_pretrained("qwen2_7b_sign_language_merged")
```

---

## 📚 Additional Resources

- **Qwen Documentation:** https://github.com/QwenLM/Qwen2
- **Gemma Documentation:** https://ai.google.dev/gemma
- **Llama Documentation:** https://github.com/meta-llama/llama3
- **Mistral Documentation:** https://docs.mistral.ai/
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **QLoRA Paper:** https://arxiv.org/abs/2305.14314

---

## ✅ Ready to Train!

Choose your model and run:

```bash
# Quick test with Qwen2-7B
bash train_single_gpu.sh configs/qwen2_7b_isign.yaml

# Production training with 4 GPUs
bash train_multi_gpu.sh configs/qwen2_7b_isign.yaml 4

# For Llama (requires HF token)
export HF_TOKEN="your_token"
bash train_multi_gpu.sh configs/llama3.1_8b_isign.yaml 4
```

Happy training! 🚀
