"""
Model Factory for Sign Language Translation - Universal Edition
Supports BOTH Seq2Seq AND Causal LM models from HuggingFace!
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

# Try to import PEFT for LoRA support
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. Install with: pip install peft bitsandbytes")


# ------------------ Feature Projection ------------------ #

class FeatureProjection(nn.Module):
    """Project pose features to model hidden size"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------ Wrapper Model ------------------ #

class SignLanguageTranslationModel(nn.Module):
    """
    Universal wrapper for Seq2Seq and Causal LM models
    """

    def __init__(
        self,
        model: nn.Module,
        feature_projection: nn.Module,
        tokenizer: PreTrainedTokenizer,
        is_encoder_decoder: bool,
    ):
        super().__init__()
        self.model = model
        self.feature_projection = feature_projection
        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        logger.info(
            f"Model type: {'Seq2Seq' if is_encoder_decoder else 'Causal LM'}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        projected = self.feature_projection(input_ids)

        if self.is_encoder_decoder:
            outputs = self.model(
                inputs_embeds=projected,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        # -------- Causal LM -------- #
        if labels is None:
            outputs = self.model(
                inputs_embeds=projected,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return {"loss": None, "logits": outputs.logits}

        text_embeds = self.model.get_input_embeddings()(labels)

        combined_embeds = torch.cat([projected, text_embeds], dim=1)

        batch_size = attention_mask.size(0)
        text_attention = torch.ones(
            batch_size,
            labels.size(1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        combined_attention = torch.cat(
            [attention_mask, text_attention], dim=1
        )

        outputs = self.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            return_dict=True,
        )

        logits = outputs.logits[:, projected.size(1):, :]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id
        )
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return {"loss": loss, "logits": logits}

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 5,
    ):
        projected = self.feature_projection(input_ids)

        if self.is_encoder_decoder:
            return self.model.generate(
                inputs_embeds=projected,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
            )

        generated = self.model.generate(
            inputs_embeds=projected,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return generated[:, projected.size(1):]


# ------------------ Model Factory ------------------ #

class ModelFactory:
    @staticmethod
    def create_model(
        model_name: str,
        num_keypoints: int,
        tokenizer: PreTrainedTokenizer,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> SignLanguageTranslationModel:

        logger.info(f"Loading model: {model_name}")

        config = AutoConfig.from_pretrained(model_name)
        is_encoder_decoder = config.is_encoder_decoder

        load_kwargs = kwargs.copy()

        if load_in_8bit or load_in_4bit:
            if not PEFT_AVAILABLE:
                raise ImportError("bitsandbytes + peft required")

            load_kwargs["device_map"] = "auto"
            load_kwargs["load_in_8bit"] = load_in_8bit
            load_kwargs["load_in_4bit"] = load_in_4bit

        if is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, **load_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )

        # -------- FIXED TOKEN RESIZE BLOCK -------- #
        try:
            config_vocab_size = None

            if hasattr(model.config, "vocab_size"):
                config_vocab_size = model.config.vocab_size

            if config_vocab_size is not None:
                if len(tokenizer) > config_vocab_size:
                    logger.warning(
                        f"Resizing embeddings: tokenizer={len(tokenizer)}, "
                        f"model={config_vocab_size}"
                    )
                    model.resize_token_embeddings(len(tokenizer))
            else:
                model.resize_token_embeddings(len(tokenizer))

        except Exception as e:
            logger.warning(
                f"Could not resize token embeddings: {e}"
            )

        # -------- Hidden Size -------- #
        if hasattr(config, "d_model"):
            hidden_size = config.d_model
        elif hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            raise ValueError("Cannot determine hidden size")

        if load_in_8bit or load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        if use_lora:
            lora_cfg = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": (
                    TaskType.SEQ_2_SEQ_LM
                    if is_encoder_decoder
                    else TaskType.CAUSAL_LM
                ),
                "target_modules": ["q_proj", "v_proj"],
            }

            if lora_config:
                lora_cfg.update(lora_config)

            model = get_peft_model(model, LoraConfig(**lora_cfg))
            model.print_trainable_parameters()

        elif freeze_encoder and is_encoder_decoder:
            for p in model.get_encoder().parameters():
                p.requires_grad = False

        elif freeze_decoder:
            for p in model.parameters():
                p.requires_grad = False

        feature_projection = FeatureProjection(
            input_dim=num_keypoints,
            output_dim=hidden_size,
            dropout=dropout,
        )

        wrapper = SignLanguageTranslationModel(
            model=model,
            feature_projection=feature_projection,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder,
        )

        total = sum(p.numel() for p in wrapper.parameters())
        trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)

        logger.info(f"Total params: {total:,}")
        logger.info(f"Trainable params: {trainable:,}")

        return wrapper
