"""
Multi-GPU Trainer for Sign Language Translation
Supports: DDP, Mixed Precision, Gradient Accumulation, Checkpointing
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, List
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
import numpy as np
from datetime import datetime

from ..utils.metrics import compute_bleu, compute_rouge

logger = logging.getLogger(__name__)


class Trainer:
    """
    Multi-GPU Trainer with all the bells and whistles
    
    Features:
    - Distributed Data Parallel (DDP)
    - Mixed Precision Training (AMP)
    - Gradient Accumulation
    - Automatic Checkpointing
    - WandB logging
    - Learning Rate Scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        use_wandb: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
        # Training config
        self.num_epochs = config.get('num_epochs', 100)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.mixed_precision = config.get('mixed_precision', True)
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.save_every = config.get('save_every', 5)
        self.eval_every = config.get('eval_every', 1)
        
        # Generation config
        self.num_beams = config.get('num_beams', 5)
        self.max_gen_length = config.get('max_gen_length', 128)
        
        # Setup
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_bleu = 0.0
        
        # Mixed precision
        self.scaler = GradScaler("cuda") if self.mixed_precision else None
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(config.get('warmup_ratio', 0.1) * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # WandB
        self.use_wandb = use_wandb and self.is_main_process
        if self.use_wandb:
            wandb.init(
                project=config.get('project_name', 'sign-language-translation'),
                name=config.get('run_name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                config=config
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
        logger.info(f"Trainer initialized on rank {rank}/{world_size}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast("cuda", enabled=self.mixed_precision):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            loss_value = loss.item() * self.gradient_accumulation_steps
            total_loss += loss_value
            
            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
                
                if self.use_wandb and self.global_step % 10 == 0:
                    wandb.log({
                        'train/loss': loss_value,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/step': self.global_step
                    })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, split: str = 'val') -> Dict:
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []
        
        # Get the underlying model for generation (unwrap DDP if needed)
        model_for_generate = self.model.module if isinstance(self.model, DDP) else self.model
        
        if self.is_main_process:
            pbar = tqdm(data_loader, desc=f"Evaluating {split}")
        else:
            pbar = data_loader
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Compute loss
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs['loss'].item()
            
            # Generate predictions (use unwrapped model for generation)
            generated = model_for_generate.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_gen_length,
                num_beams=self.num_beams,
                length_penalty=0.6,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            # Decode
            labels_clean = torch.where(labels == -100, self.tokenizer.pad_token_id, labels)
            predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            references = self.tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
        
        # Compute metrics
        avg_loss = total_loss / len(data_loader)
        bleu_scores = compute_bleu(all_references, all_predictions)
        rouge_scores = compute_rouge(all_references, all_predictions)
        
        metrics = {
            f'{split}/loss': avg_loss,
            f'{split}/bleu1': bleu_scores['bleu1'],
            f'{split}/bleu2': bleu_scores['bleu2'],
            f'{split}/bleu3': bleu_scores['bleu3'],
            f'{split}/bleu4': bleu_scores['bleu4'],
            f'{split}/rouge_l': rouge_scores['rouge_l']
        }
        
        if self.is_main_process:
            logger.info(f"\n{split.upper()} Results:")
            logger.info(f"  Loss: {avg_loss:.4f}")
            logger.info(f"  BLEU-4: {bleu_scores['bleu4']*100:.2f}")
            logger.info(f"  ROUGE-L: {rouge_scores['rouge_l']*100:.2f}")
        
        return metrics, all_predictions, all_references
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        if not self.is_main_process:
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(1, self.num_epochs + 1):
            # Set epoch on DistributedSampler for proper shuffling across epochs
            if self.world_size > 1 and hasattr(self.train_loader, 'sampler'):
                sampler = self.train_loader.sampler
                if hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(epoch)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            if self.is_main_process:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Evaluate
            if epoch % self.eval_every == 0:
                val_metrics, val_preds, val_refs = self.evaluate(self.val_loader, split='val')
                
                # Test evaluation (optional, can be less frequent)
                if epoch % (self.eval_every * 2) == 0:
                    test_metrics, test_preds, test_refs = self.evaluate(self.test_loader, split='test')
                else:
                    test_metrics = {}
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        **val_metrics,
                        **test_metrics,
                        'epoch': epoch
                    })
                
                # Save checkpoint
                is_best = val_metrics['val/bleu4'] > self.best_val_bleu
                if is_best:
                    self.best_val_bleu = val_metrics['val/bleu4']
                    self.best_val_loss = val_metrics['val/loss']
                
                if epoch % self.save_every == 0 or is_best:
                    self.save_checkpoint(epoch, val_metrics, is_best=is_best)
        
        if self.is_main_process:
            logger.info("Training completed!")
            logger.info(f"Best Val BLEU-4: {self.best_val_bleu*100:.2f}")
            
            if self.use_wandb:
                wandb.finish()


def setup_distributed():
    """Setup for distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


# Example usage
if __name__ == "__main__":
    import os
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    print(f"Rank {rank}/{world_size} on device {device}")
