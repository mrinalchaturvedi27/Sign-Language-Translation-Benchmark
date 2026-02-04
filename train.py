"""
Main Training Script for Sign Language Translation
Easy to use - just modify config and run!
"""

import os
import sys
import yaml
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataloaders.sign_dataloader import create_dataloaders
from models.model_factory import ModelFactory
from src.trainers.trainer import Trainer, setup_distributed, cleanup_distributed


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str):
    """Main training function"""
    
    # Load config
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    is_main_process = (rank == 0)
    
    if is_main_process:
        logger.info(f"Training on {world_size} GPU(s)")
        logger.info(f"Config: {config}")
    
    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load tokenizer
    tokenizer_name = config['model']['tokenizer']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add special tokens if needed
    special_tokens = config['model'].get('special_tokens', {})
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    
    if is_main_process:
        logger.info(f"Loaded tokenizer: {tokenizer_name} (vocab_size={len(tokenizer)})")
    
    # Create dataloaders
    data_config = config['data']
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=data_config['train_path'],
        val_path=data_config['val_path'],
        test_path=data_config['test_path'],
        pose_dir=data_config['pose_dir'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        max_frames=data_config['max_frames'],
        max_length=data_config['max_length'],
        step_frames=data_config.get('step_frames', 1),
        num_keypoints=data_config['num_keypoints']
    )
    
    if is_main_process:
        logger.info(f"Created dataloaders:")
        logger.info(f"  Train: {len(train_loader.dataset)} samples")
        logger.info(f"  Val: {len(val_loader.dataset)} samples")
        logger.info(f"  Test: {len(test_loader.dataset)} samples")
    
    # Create model
    model_config = config['model']
    model = ModelFactory.create_model(
        model_name=model_config['name'],
        num_keypoints=data_config['num_keypoints'],
        tokenizer=tokenizer,
        dropout=model_config.get('dropout', 0.1),
        freeze_encoder=model_config.get('freeze_encoder', False),
        freeze_decoder=model_config.get('freeze_decoder', False),
        use_lora=model_config.get('use_lora', False),
        lora_config=model_config.get('lora_config', None),
        load_in_8bit=model_config.get('load_in_8bit', False),
        load_in_4bit=model_config.get('load_in_4bit', False),
        **model_config.get('params', {})
    )
    
    model = model.to(device)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created: {model_config['name']}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Wrap model with DDP for multi-GPU
    # Note: find_unused_parameters=True is needed because the model has different code
    # paths for encoder-decoder vs causal LM models (in SignLanguageTranslationModel.forward()),
    # which may result in some parameters not participating in the loss computation.
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if is_main_process:
            logger.info(f"Model wrapped with DDP (world_size={world_size})")
        # Synchronize all processes before continuing
        dist.barrier()
    
    # Create optimizer
    training_config = config['training']
    lr = float(training_config["learning_rate"])
    weight_decay = float(training_config.get("weight_decay", 0.0))

    betas = training_config.get("betas", (0.9, 0.999))
    betas = tuple(map(float, betas))
    
    # NOTE: Effective batch size = batch_size * gradient_accumulation_steps * world_size
    # If you want to keep the same effective batch size when scaling GPUs,
    # divide batch_size by world_size in the config.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        config=training_config,
        device=device,
        rank=rank,
        world_size=world_size,
        use_wandb=training_config.get('use_wandb', True)
    )
    
    # Train
    trainer.train()
    
    # Cleanup
    cleanup_distributed()
    
    if is_main_process:
        logger.info("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Sign Language Translation Model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    
    args = parser.parse_args()
    
    main(args.config)
