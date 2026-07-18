"""Training script for multimodal biometric system."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import Config
from models.ear_model import EarCNN
from models.face_model import FaceCNN
from utils.dataset import MultimodalBiometricDataset, get_transforms


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    model_name: str,
    checkpoint_path: str
) -> nn.Module:
    """
    Train a single model (ear or face).
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of training epochs
        model_name: Name for saving checkpoints
        checkpoint_path: Path to save checkpoints
        
    Returns:
        Trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{num_epochs}')
        for ear_imgs, face_imgs, labels in pbar:
            # Select appropriate input based on model type
            if 'ear' in model_name.lower():
                inputs = ear_imgs.to(device)
            else:
                inputs = face_imgs.to(device)
            
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for ear_imgs, face_imgs, labels in val_loader:
                if 'ear' in model_name.lower():
                    inputs = ear_imgs.to(device)
                else:
                    inputs = face_imgs.to(device)
                
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f'{model_name} - Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 60)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_file = os.path.join(checkpoint_path, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_file)
            print(f'Saved best model to {checkpoint_file}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    checkpoint_file = os.path.join(checkpoint_path, f'{model_name}_best.pth')
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded best model from {checkpoint_file}')
    
    return model


def main():
    """Main training function."""
    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Device configuration
    device = Config.DEVICE
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    os.makedirs(Config.CHECKPOINT_PATH, exist_ok=True)
    
    # Load dataset
    print('Loading dataset...')
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Full dataset
    full_dataset = MultimodalBiometricDataset(
        ear_dir=Config.EAR_DATA_PATH,
        face_dir=Config.FACE_DATA_PATH,
        transform=None,
        ear_size=Config.EAR_IMAGE_SIZE,
        face_size=Config.FACE_IMAGE_SIZE
    )
    
    if len(full_dataset) == 0:
        print('Warning: Dataset is empty. Please add data to data/raw/ear and data/raw/face')
        print('Expected structure:')
        print('  data/raw/ear/person_id_1/image1.jpg')
        print('  data/raw/face/person_id_1/image1.jpg')
        return
    
    # Update num_classes based on dataset
    num_classes = len(full_dataset.class_to_idx)
    print(f'Number of classes: {num_classes}')
    Config.NUM_CLASSES = num_classes
    
    # Split dataset
    train_size = int((1 - Config.VAL_SPLIT - Config.TEST_SPLIT) * len(full_dataset))
    val_size = int(Config.VAL_SPLIT * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.RANDOM_SEED)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, 
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, 
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # Initialize models
    print('\n' + '='*60)
    print('Training Ear Model')
    print('='*60)
    ear_model = EarCNN(
        num_classes=Config.NUM_CLASSES,
        embedding_dim=Config.EMBEDDING_DIM
    ).to(device)
    
    ear_model = train_single_model(
        ear_model, train_loader, val_loader, device,
        Config.NUM_EPOCHS, 'ear_model', Config.CHECKPOINT_PATH
    )
    
    print('\n' + '='*60)
    print('Training Face Model')
    print('='*60)
    face_model = FaceCNN(
        num_classes=Config.NUM_CLASSES,
        embedding_dim=Config.EMBEDDING_DIM
    ).to(device)
    
    face_model = train_single_model(
        face_model, train_loader, val_loader, device,
        Config.NUM_EPOCHS, 'face_model', Config.CHECKPOINT_PATH
    )
    
    print('\n' + '='*60)
    print('Training Complete!')
    print('='*60)
    print(f'Models saved to {Config.CHECKPOINT_PATH}')


if __name__ == '__main__':
    main()
