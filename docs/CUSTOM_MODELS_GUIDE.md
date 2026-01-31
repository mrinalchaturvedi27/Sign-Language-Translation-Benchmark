# Using Custom HuggingFace Models 🤗

This pipeline supports **ANY** Seq2Seq model from HuggingFace! Here's how to use custom models.

## Quick Example

Want to try Google's Pegasus model? Just create a config:

```yaml
# configs/pegasus_isign.yaml

data:
  train_path: "/DATA7/vaibhav/tokenization/train_split_unicode_filtered.csv"
  val_path: "/DATA7/vaibhav/tokenization/val_split_unicode_filtered.csv"
  test_path: "/DATA7/vaibhav/tokenization/test_split_unicode_filtered.csv"
  pose_dir: "/DATA7/vaibhav/isign/Data/iSign-poses_v1.1/"
  
  max_frames: 300
  max_length: 128
  step_frames: 5
  num_keypoints: 152

model:
  name: "google/pegasus-large"      # HuggingFace model ID
  tokenizer: "google/pegasus-large"  # Tokenizer (usually same)
  dropout: 0.1

training:
  num_epochs: 50
  batch_size: 16
  learning_rate: 5e-5
  # ... rest of training config
```

Then train:
```bash
bash train_multi_gpu.sh configs/pegasus_isign.yaml 4
```

## Finding Models on HuggingFace

1. Go to https://huggingface.co/models
2. Filter by task: "Translation" or "Summarization"
3. Filter by type: "Seq2Seq"
4. Copy the model ID (e.g., "facebook/bart-large")

## Popular Model Categories

### Translation Models
```yaml
# NLLB (200 languages!)
model:
  name: "facebook/nllb-200-distilled-600M"
  tokenizer: "facebook/nllb-200-distilled-600M"

# M2M100 (100 languages)
model:
  name: "facebook/m2m100_1.2B"
  tokenizer: "facebook/m2m100_1.2B"

# MarianMT (specific language pairs)
model:
  name: "Helsinki-NLP/opus-mt-en-de"
  tokenizer: "Helsinki-NLP/opus-mt-en-de"
```

### Summarization Models
```yaml
# Pegasus
model:
  name: "google/pegasus-large"
  tokenizer: "google/pegasus-large"

# LED (Long Document)
model:
  name: "allenai/led-large-16384"
  tokenizer: "allenai/led-large-16384"

# BART
model:
  name: "facebook/bart-large-cnn"
  tokenizer: "facebook/bart-large-cnn"
```

### Multilingual Models
```yaml
# mT5 (101 languages)
model:
  name: "google/mt5-xl"
  tokenizer: "google/mt5-xl"
  use_lora: true  # Large model - use LoRA!

# mBART (50 languages)
model:
  name: "facebook/mbart-large-50-many-to-many-mmt"
  tokenizer: "facebook/mbart-large-50-many-to-many-mmt"
```

## Model Configuration Options

```yaml
model:
  # Required
  name: "model-name-on-huggingface"
  tokenizer: "tokenizer-name"  # Usually same as model
  
  # Optional
  dropout: 0.1                  # Override model dropout
  freeze_encoder: false         # Freeze encoder weights
  freeze_decoder: false         # Freeze decoder weights
  
  # LoRA (for large models)
  use_lora: true
  lora_config:
    r: 16                       # Rank
    lora_alpha: 32              # Scaling
    target_modules: ["q", "v"]  # Which layers
    lora_dropout: 0.1
  
  # Model loading params
  params:
    torch_dtype: "auto"         # auto, float32, float16, bfloat16
    device_map: "auto"          # For multi-GPU
    load_in_8bit: false         # 8-bit quantization
    load_in_4bit: false         # 4-bit quantization
```

## Advanced: Quantized Models

For very large models on limited GPU memory:

```yaml
model:
  name: "google/mt5-xxl"  # 13B parameters!
  tokenizer: "google/mt5-xxl"
  
  use_lora: true  # Must use LoRA for fine-tuning quantized models
  
  params:
    load_in_8bit: true    # Load in 8-bit (requires bitsandbytes)
    # OR
    load_in_4bit: true    # Load in 4-bit (even more memory efficient)
    
  lora_config:
    r: 64
    lora_alpha: 128
    target_modules: ["q", "k", "v", "o"]
```

**Install quantization support:**
```bash
pip install bitsandbytes
```

## Checking Model Compatibility

Not all models work out of the box. Check if a model is compatible:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "your-model-name"

try:
    # Try loading
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check architecture
    print(f"✅ Model loaded successfully!")
    print(f"Model type: {model.config.model_type}")
    print(f"Hidden size: {model.config.d_model if hasattr(model.config, 'd_model') else model.config.hidden_size}")
    print(f"Vocab size: {len(tokenizer)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
```

## Recommended Models by Use Case

### Best for English Sign Language (ISL, ASL)
1. **T5-Large** - Best quality
2. **BART-Large** - Good alternative
3. **T5-Base** - Fastest with good quality

### Best for Multilingual
1. **mBART-Large-50** - 50 languages
2. **mT5-Large** - 101 languages
3. **M2M100-1.2B** - 100 languages

### Best for Limited Memory
1. **T5-Small** - 60M params
2. **T5-Base + LoRA** - Memory efficient
3. **distilled models** - Smaller versions

### Best for State-of-the-Art
1. **T5-3B + LoRA** - Highest quality
2. **mT5-XL + LoRA** - Multilingual SOTA
3. **T5-11B + LoRA + 8bit** - Absolute best (needs 40GB+ GPU)

## Troubleshooting

### Model doesn't load
- Check model name on HuggingFace
- Ensure it's a Seq2Seq model
- Try adding `trust_remote_code=true` to params

### OOM (Out of Memory)
- Use smaller model
- Enable LoRA
- Use quantization (8-bit or 4-bit)
- Reduce batch size

### Slow training
- Use smaller model for experiments
- Use more GPUs
- Increase num_workers

### Poor results
- Train longer
- Try different learning rate
- Use larger model
- Check data quality

## Example: Full Custom Model Config

```yaml
# configs/custom_nllb_isign.yaml

data:
  train_path: "/DATA7/vaibhav/tokenization/train_split_unicode_filtered.csv"
  val_path: "/DATA7/vaibhav/tokenization/val_split_unicode_filtered.csv"
  test_path: "/DATA7/vaibhav/tokenization/test_split_unicode_filtered.csv"
  pose_dir: "/DATA7/vaibhav/isign/Data/iSign-poses_v1.1/"
  max_frames: 300
  max_length: 128
  step_frames: 5
  num_keypoints: 152

model:
  name: "facebook/nllb-200-distilled-600M"
  tokenizer: "facebook/nllb-200-distilled-600M"
  dropout: 0.1
  freeze_encoder: false
  freeze_decoder: false
  use_lora: false
  
  params:
    torch_dtype: "auto"
  
  special_tokens:
    additional_special_tokens: ["<PERSON>", "<UNKNOWN>"]

training:
  num_epochs: 50
  batch_size: 16
  learning_rate: 5e-5
  weight_decay: 0.01
  betas: [0.9, 0.999]
  max_grad_norm: 1.0
  gradient_accumulation_steps: 2
  mixed_precision: true
  warmup_ratio: 0.1
  
  checkpoint_dir: "checkpoints/nllb_isign"
  save_every: 5
  eval_every: 1
  
  num_beams: 5
  max_gen_length: 128
  
  use_wandb: true
  project_name: "sign-language-translation"
  run_name: "nllb_isign"
  num_workers: 4

seed: 42
```

Train it:
```bash
bash train_multi_gpu.sh configs/custom_nllb_isign.yaml 4
```

---

## 🎓 Pro Tips

1. **Start small**: Test with T5-Small before using large models
2. **Use LoRA**: For models >1B parameters, always use LoRA
3. **Monitor WandB**: Track all experiments for comparison
4. **Save checkpoints**: Training can take hours/days
5. **Experiment**: Try multiple models to find the best one!

---

Happy experimenting with HuggingFace models! 🚀
