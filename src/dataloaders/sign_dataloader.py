"""
Generic Sign Language DataLoader
Supports: Pose sequences, Video frames, Multiple datasets
Easy to extend and modify
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pickle
from pose_format import Pose
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class SignLanguageDataset(Dataset):
    """
    Generic Sign Language Dataset
    
    Args:
        data_path: Path to CSV file with columns ['uid', 'text'] or ['video_path', 'text']
        pose_dir: Directory containing pose files
        tokenizer: Huggingface tokenizer for text
        max_frames: Maximum number of frames to use
        max_length: Maximum length of text sequence
        step_frames: Downsample frames (e.g., 5 means every 5th frame)
        add_noise: Whether to add Gaussian noise for augmentation
        use_video: If True, load video frames instead of poses
    """
    
    def __init__(
        self,
        data_path: str,
        pose_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_frames: int = 300,
        max_length: int = 128,
        step_frames: int = 1,
        add_noise: bool = False,
        noise_std: float = 0.01,
        use_video: bool = False,
        num_keypoints: int = 152,
        labels: Optional[List[List[int]]] = None,
    ):
        self.data_path = Path(data_path)
        self.pose_dir = Path(pose_dir)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_length = max_length
        self.step_frames = step_frames
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.use_video = use_video
        self.num_keypoints = num_keypoints
        
        # Load data
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df)} samples from {data_path}")
        
        # Use provided labels or tokenize
        if labels is not None:
            self.labels = labels
        else:
            self.labels = self._tokenize_texts(self.df['text'].tolist())
    
    def _tokenize_texts(self, texts: List[str]) -> List[List[int]]:
        """Tokenize all texts"""
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encodings['input_ids'].tolist()
    
    def _load_pose(self, uid: str) -> Optional[np.ndarray]:
        """Load pose file and extract keypoints"""
        pose_path = self.pose_dir / f"{uid}.pose"
        
        if not pose_path.exists():
            logger.warning(f"Pose file not found: {pose_path}")
            return None
        
        try:
            with open(pose_path, 'rb') as f:
                pose = Pose.read(f.read())
            
            # Extract and process keypoints (customize based on your extraction logic)
            keypoints = self._extract_keypoints(pose)
            
            # Downsample
            keypoints = keypoints[::self.step_frames]
            
            # Add noise if needed
            if self.add_noise:
                noise = np.random.normal(0, self.noise_std, keypoints.shape)
                keypoints = keypoints + noise
            
            # Truncate or pad to max_frames
            keypoints = self._pad_or_truncate(keypoints, self.max_frames)
            
            return keypoints
            
        except Exception as e:
            logger.error(f"Error loading pose {uid}: {e}")
            return None
    
    def _extract_keypoints(self, pose: Pose) -> np.ndarray:
        """
        Extract keypoints from pose object
        Customize this based on your pose format
        """
        # Example: Extract all body, hand, face keypoints
        # Modify based on your specific requirements
        data = pose.body.data
        data = np.squeeze(data, axis=1)  # Remove singleton dimension
        
        # Extract x, y coordinates (assuming 3D data with x, y, confidence)
        keypoints = data[:, :, :2]  # Shape: (frames, keypoints, 2)
        
        # Flatten to (frames, num_keypoints)
        keypoints = keypoints.reshape(keypoints.shape[0], -1)
        
        return keypoints
    
    def _pad_or_truncate(self, sequence: np.ndarray, max_len: int) -> np.ndarray:
        """Pad or truncate sequence to max_len"""
        seq_len = sequence.shape[0]
        
        if seq_len > max_len:
            return sequence[:max_len]
        elif seq_len < max_len:
            pad_len = max_len - seq_len
            padding = np.zeros((pad_len, sequence.shape[1]))
            return np.vstack([sequence, padding])
        return sequence
    
    def _create_attention_mask(self, sequence: np.ndarray) -> np.ndarray:
        """Create attention mask (1 for real frames, 0 for padding)"""
        # Detect padding (all zeros)
        mask = ~np.all(sequence == 0, axis=1)
        return mask.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys:
                - input_ids: Pose sequence (frames, num_keypoints)
                - attention_mask: Mask for input (frames,)
                - labels: Tokenized text (max_length,)
        """
        row = self.df.iloc[idx]
        uid = row.get('uid', row.get('video_path', str(idx)))
        
        # Load pose
        pose_sequence = self._load_pose(uid)
        
        if pose_sequence is None:
            # Return dummy data if loading fails
            pose_sequence = np.zeros((self.max_frames, self.num_keypoints))
        
        # Create attention mask
        attention_mask = self._create_attention_mask(pose_sequence)
        
        # Get labels
        labels = self.labels[idx]
        
        return {
            'input_ids': torch.FloatTensor(pose_sequence),
            'attention_mask': torch.FloatTensor(attention_mask),
            'labels': torch.LongTensor(labels)
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    pose_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders
    
    Usage:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_path='path/to/train.csv',
            val_path='path/to/val.csv',
            test_path='path/to/test.csv',
            pose_dir='path/to/poses',
            tokenizer=tokenizer,
            batch_size=32,
            max_frames=300,
            step_frames=5
        )
    """
    
    # Create datasets
    train_dataset = SignLanguageDataset(
        data_path=train_path,
        pose_dir=pose_dir,
        tokenizer=tokenizer,
        add_noise=True,  # Augmentation for training
        **dataset_kwargs
    )
    
    val_dataset = SignLanguageDataset(
        data_path=val_path,
        pose_dir=pose_dir,
        tokenizer=tokenizer,
        add_noise=False,
        **dataset_kwargs
    )
    
    test_dataset = SignLanguageDataset(
        data_path=test_path,
        pose_dir=pose_dir,
        tokenizer=tokenizer,
        add_noise=False,
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path="/path/to/train.csv",
        val_path="/path/to/val.csv",
        test_path="/path/to/test.csv",
        pose_dir="/path/to/poses",
        tokenizer=tokenizer,
        batch_size=16,
        max_frames=300,
        step_frames=5
    )
    
    # Test
    for batch in train_loader:
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        break
