"""
Model Factory for Sign Language Translation - Universal Edition
Supports BOTH Seq2Seq AND Causal LM models from HuggingFace!

Supported Architectures:
1. Seq2Seq (Encoder-Decoder):
   - T5, mT5, BART, mBART, Pegasus, M2M100
   
2. Causal LM (Decoder-Only):
   - Qwen (Qwen/Qwen2-7B, Qwen/Qwen2.5-7B-Instruct)
   - Gemma (google/gemma-2-9b, google/gemma-7b)
   - Llama (meta-llama/Llama-3.1-8B, meta-llama/Llama-2-7b)
   - Mistral (mistralai/Mistral-7B-v0.3)
   - Phi (microsoft/phi-3-mini)
   - Any other decoder-only model!
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Dict, Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)

# Try to import PEFT for LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. Install with: pip install peft")



class FeatureProjection(nn.Module):
    """Project pose features to model hidden size"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SignLanguageTranslationModel(nn.Module):
    """
    Universal wrapper for both Seq2Seq and Causal LM models
    Handles pose-to-text translation for any architecture
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_projection: nn.Module,
        tokenizer: PreTrainedTokenizer,
        is_encoder_decoder: bool
    ):
        super().__init__()
        self.model = model
        self.feature_projection = feature_projection
        self.tokenizer = tokenizer
        self.config = model.config
        self.is_encoder_decoder = is_encoder_decoder
        
        logger.info(f"Model type: {'Seq2Seq (Encoder-Decoder)' if is_encoder_decoder else 'Causal LM (Decoder-Only)'}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Pose sequences (batch, seq_len, num_keypoints)
            attention_mask: Mask (batch, seq_len)
            labels: Target tokens (batch, target_len)
        
        Returns:
            Dict with 'loss' and 'logits'
        """
        # Project features to model hidden size
        projected = self.feature_projection(input_ids)  # (batch, seq_len, hidden_size)
        
        if self.is_encoder_decoder:
            # Seq2Seq models (T5, BART, mBART)
            outputs = self.model(
                inputs_embeds=projected,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        else:
            # Causal LM models (Qwen, Gemma, Llama, Mistral)
            # For decoder-only: concatenate pose embeddings with text embeddings
            
            if labels is not None:
                # Get text embeddings
                text_embeds = self.model.get_input_embeddings()(labels)
                
                # Concatenate pose + text embeddings
                combined_embeds = torch.cat([projected, text_embeds], dim=1)
                
                # Create combined attention mask
                batch_size = attention_mask.size(0)
                text_attention = torch.ones(
                    batch_size, 
                    labels.size(1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                combined_attention = torch.cat([attention_mask, text_attention], dim=1)
                
                # Forward through model
                outputs = self.model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention,
                    labels=None,  # We'll compute loss manually
                    return_dict=True
                )
                
                # Compute loss on text portion only
                logits = outputs.logits[:, projected.size(1):, :]  # Text portion
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                outputs.loss = loss
                outputs.logits = logits
            else:
                # Inference mode
                outputs = self.model(
                    inputs_embeds=projected,
                    attention_mask=attention_mask,
                    return_dict=True
                )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 5,
        **kwargs
    ) -> torch.Tensor:
        """Generate translations using beam search"""
        projected = self.feature_projection(input_ids)
        
        if self.is_encoder_decoder:
            # Seq2Seq generation
            generated = self.model.generate(
                inputs_embeds=projected,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                **kwargs
            )
        else:
            # Causal LM generation
            generated = self.model.generate(
                inputs_embeds=projected,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # For decoder-only, remove the prompt portion
            generated = generated[:, projected.size(1):]
        
        return generated


class ModelFactory:
    """
    Universal Factory for HuggingFace Models
    Automatically detects and loads:
    - Seq2Seq models (T5, BART, mBART, Pegasus, M2M100)
    - Causal LM models (Qwen, Gemma, Llama, Mistral, Phi)
    """
    
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
        **kwargs
    ) -> SignLanguageTranslationModel:
        """
        Create model from HuggingFace - auto-detects architecture type
        
        Args:
            model_name: HuggingFace model name
                Seq2Seq: "t5-base", "facebook/bart-large", "facebook/mbart-large-50"
                Causal LM: "Qwen/Qwen2-7B", "google/gemma-7b", "meta-llama/Llama-3.1-8B"
            num_keypoints: Input feature dimension (e.g., 152 for pose keypoints)
            tokenizer: Tokenizer
            dropout: Dropout rate
            freeze_encoder: Whether to freeze encoder weights (Seq2Seq only)
            freeze_decoder: Whether to freeze decoder weights
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_config: LoRA configuration dict
            load_in_8bit: Load model in 8-bit (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit (requires bitsandbytes)
            **kwargs: Additional arguments for model loading
        
        Returns:
            SignLanguageTranslationModel
        
        Examples:
            # Seq2Seq model
            model = ModelFactory.create_model(
                model_name='t5-base',
                num_keypoints=152,
                tokenizer=tokenizer
            )
            
            # Causal LM model
            model = ModelFactory.create_model(
                model_name='Qwen/Qwen2-7B',
                num_keypoints=152,
                tokenizer=tokenizer,
                use_lora=True
            )
            
            # Large model with quantization
            model = ModelFactory.create_model(
                model_name='meta-llama/Llama-3.1-8B',
                num_keypoints=152,
                tokenizer=tokenizer,
                load_in_4bit=True,
                use_lora=True
            )
        """
        
        logger.info(f"Loading model: {model_name}")
        
        # Load config to detect model type
        config = AutoConfig.from_pretrained(model_name)
        is_encoder_decoder = config.is_encoder_decoder
        
        logger.info(f"Detected architecture: {'Seq2Seq (Encoder-Decoder)' if is_encoder_decoder else 'Causal LM (Decoder-Only)'}")
        
        # Prepare loading kwargs
        load_kwargs = kwargs.copy()
        
        # Add quantization config if requested
        if load_in_8bit or load_in_4bit:
            if not PEFT_AVAILABLE:
                raise ImportError("Quantization requires PEFT. Install with: pip install peft bitsandbytes")
            
            load_kwargs['device_map'] = 'auto'
            if load_in_8bit:
                load_kwargs['load_in_8bit'] = True
                logger.info("Loading model in 8-bit mode")
            elif load_in_4bit:
                load_kwargs['load_in_4bit'] = True
                logger.info("Loading model in 4-bit mode")
        
        # Load model based on architecture type
        try:
            if is_encoder_decoder:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Resize token embeddings if needed
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Resized token embeddings to {len(tokenizer)}")
        
        # Get hidden size from config
        if hasattr(config, 'd_model'):
            hidden_size = config.d_model
        elif hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        else:
            raise ValueError(f"Could not determine hidden size from config")
        
        logger.info(f"Model hidden size: {hidden_size}")
        
        # Apply dropout to model if specified
        if hasattr(model.config, 'dropout'):
            model.config.dropout = dropout
        if hasattr(model.config, 'attention_dropout'):
            model.config.attention_dropout = dropout
        if hasattr(model.config, 'activation_dropout'):
            model.config.activation_dropout = dropout
        
        # Prepare model for k-bit training if quantized
        if load_in_8bit or load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA if requested
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT not installed. Install with: pip install peft")
            
            logger.info("Applying LoRA for parameter-efficient fine-tuning")
            
            # Default LoRA config
            default_lora_config = {
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'v_proj'] if not is_encoder_decoder else ['q', 'v'],
                'lora_dropout': 0.1,
                'bias': 'none',
                'task_type': TaskType.SEQ_2_SEQ_LM if is_encoder_decoder else TaskType.CAUSAL_LM
            }
            
            # Update with user config
            if lora_config:
                default_lora_config.update(lora_config)
            
            # Convert task_type string to enum if needed
            if isinstance(default_lora_config.get('task_type'), str):
                default_lora_config['task_type'] = TaskType.SEQ_2_SEQ_LM if is_encoder_decoder else TaskType.CAUSAL_LM
            
            lora_cfg = LoraConfig(**default_lora_config)
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        
        # Freeze encoder if specified (Seq2Seq only, and not using LoRA)
        elif freeze_encoder and is_encoder_decoder:
            logger.info("Freezing encoder weights")
            for param in model.get_encoder().parameters():
                param.requires_grad = False
        
        # Freeze decoder if specified (and not using LoRA)
        elif freeze_decoder:
            logger.info("Freezing decoder weights")
            if is_encoder_decoder:
                for param in model.get_decoder().parameters():
                    param.requires_grad = False
            else:
                # For causal LM, freeze all transformer layers
                for param in model.model.parameters():
                    param.requires_grad = False
        
        # Create feature projection layer
        feature_projection = FeatureProjection(
            input_dim=num_keypoints,
            output_dim=hidden_size,
            dropout=dropout
        )
        
        # Wrap in our custom model
        wrapper = SignLanguageTranslationModel(
            model=model,
            feature_projection=feature_projection,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in wrapper.parameters())
        trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
        
        logger.info(f"Model created successfully!")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        return wrapper


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Example 1: T5-base
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = ModelFactory.create_model(
        model_name='t5-base',
        num_keypoints=152,
        tokenizer=tokenizer
    )
    
    print(f"\nT5-Base Model:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example 2: BART-large
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = ModelFactory.create_model(
        model_name='facebook/bart-large',
        num_keypoints=152,
        tokenizer=tokenizer
    )
    
    print(f"\nBART-Large Model:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example 3: mBART with frozen encoder
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
    model = ModelFactory.create_model(
        model_name='facebook/mbart-large-50',
        num_keypoints=152,
        tokenizer=tokenizer,
        freeze_encoder=True  # Only train decoder + projection
    )
    
    print(f"\nmBART-50 Model (frozen encoder):")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

