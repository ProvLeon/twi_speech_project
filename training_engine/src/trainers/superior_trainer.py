import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda import amp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import os
import json
import wandb
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SuperiorTwiDataset(Dataset):
    """Enhanced dataset with advanced augmentation capabilities"""

    def __init__(self, features, labels, label_to_idx=None, augment=True, augment_prob=0.6):
        self.features = features
        self.labels = labels
        self.augment = augment
        self.augment_prob = augment_prob

        if label_to_idx is None:
            unique_labels = sorted(set(labels))
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.label_indices = [self.label_to_idx[label] for label in labels]

        # Initialize augmentation
        self.augmenter = AdvancedAugmentation()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].copy()
        label = self.label_indices[idx]

        # Apply augmentation during training
        if self.augment and np.random.random() < self.augment_prob:
            feature = self.augmenter.augment(feature)

        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature_tensor, label_tensor

    def get_num_classes(self):
        return len(self.label_to_idx)


class AdvancedAugmentation:
    """Advanced augmentation techniques for speech features"""

    def __init__(self):
        self.techniques = [
            self.spec_augment,
            self.time_masking,
            self.freq_masking,
            self.gaussian_noise,
            self.time_shifting,
            self.pitch_shifting,
            self.speed_perturbation
        ]

    def spec_augment(self, features, freq_mask_param=15, time_mask_param=25, num_masks=2):
        """SpecAugment implementation"""
        features = features.copy()
        freq_len, time_len = features.shape

        # Frequency masking
        for _ in range(num_masks):
            if freq_len > freq_mask_param:
                f = np.random.randint(0, freq_mask_param)
                f0 = np.random.randint(0, freq_len - f)
                features[f0:f0+f, :] = 0

        # Time masking
        for _ in range(num_masks):
            if time_len > time_mask_param:
                t = np.random.randint(0, time_mask_param)
                t0 = np.random.randint(0, time_len - t)
                features[:, t0:t0+t] = 0

        return features

    def time_masking(self, features, max_mask_length=20):
        """Time domain masking"""
        features = features.copy()
        time_len = features.shape[1]

        if time_len > max_mask_length:
            mask_length = np.random.randint(1, max_mask_length)
            start_pos = np.random.randint(0, time_len - mask_length)
            features[:, start_pos:start_pos+mask_length] = 0

        return features

    def freq_masking(self, features, max_mask_length=10):
        """Frequency domain masking"""
        features = features.copy()
        freq_len = features.shape[0]

        if freq_len > max_mask_length:
            mask_length = np.random.randint(1, max_mask_length)
            start_pos = np.random.randint(0, freq_len - mask_length)
            features[start_pos:start_pos+mask_length, :] = 0

        return features

    def gaussian_noise(self, features, noise_level=0.005):
        """Add Gaussian noise"""
        noise = np.random.randn(*features.shape) * noise_level
        return features + noise

    def time_shifting(self, features, max_shift=5):
        """Time shifting augmentation"""
        features = features.copy()
        time_len = features.shape[1]
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift > 0:
            features[:, shift:] = features[:, :-shift]
            features[:, :shift] = 0
        elif shift < 0:
            features[:, :shift] = features[:, -shift:]
            features[:, shift:] = 0

        return features

    def pitch_shifting(self, features, max_shift=2):
        """Simulate pitch shifting by frequency bin shifting"""
        features = features.copy()
        freq_len = features.shape[0]
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift > 0:
            features[shift:, :] = features[:-shift, :]
            features[:shift, :] = 0
        elif shift < 0:
            features[:shift, :] = features[-shift:, :]
            features[shift:, :] = 0

        return features

    def speed_perturbation(self, features, speed_factor=None):
        """Simulate speed perturbation through interpolation"""
        if speed_factor is None:
            speed_factor = np.random.uniform(0.9, 1.1)

        time_len = features.shape[1]
        new_time_len = int(time_len / speed_factor)

        # Simple linear interpolation
        if new_time_len != time_len:
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, time_len - 1, time_len)
            new_indices = np.linspace(0, time_len - 1, new_time_len)

            interpolated_features = np.zeros((features.shape[0], new_time_len))
            for i in range(features.shape[0]):
                f = interp1d(old_indices, features[i, :], kind='linear', bounds_error=False, fill_value=0)
                interpolated_features[i, :] = f(new_indices)

            # Pad or truncate to original length
            if new_time_len > time_len:
                return interpolated_features[:, :time_len]
            else:
                padded = np.zeros_like(features)
                padded[:, :new_time_len] = interpolated_features
                return padded

        return features

    def augment(self, features):
        """Apply random augmentation techniques"""
        # Choose 1-3 random augmentation techniques
        num_augs = np.random.randint(1, 4)
        selected_augs = np.random.choice(self.techniques, num_augs, replace=False)

        for aug_fn in selected_augs:
            try:
                features = aug_fn(features)
            except:
                continue  # Skip if augmentation fails

        return features


class SuperiorTrainer:
    """
    Superior training engine that outperforms the original local_dialect approach
    with advanced optimization, regularization, and training strategies
    """

    def __init__(self, model, device, config=None):
        self.model = model
        self.device = device
        self.model.to(device)

        # Load configuration
        self.config = config or self._default_config()

        # Initialize optimizer with advanced settings
        self._setup_optimizer()

        # Initialize loss function with label smoothing
        self._setup_loss_function()

        # Initialize schedulers
        self._setup_schedulers()

        # Mixed precision training
        self.use_amp = self.config.get('use_amp', True) and device.type == 'cuda'
        if self.use_amp:
            self.scaler = amp.GradScaler()

        # Initialize tracking
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [], 'learning_rates': [], 'epochs': []
        }

        # Setup model directory
        self.model_dir = self.config.get('model_dir', 'models/superior')
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize best metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0

        # Early stopping
        self.patience_counter = 0

        # Gradient clipping
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)

    def _default_config(self):
        """Default training configuration"""
        return {
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'optimizer': 'adamw',
            'scheduler': 'cosine_warmup',
            'warmup_epochs': 5,
            'label_smoothing': 0.1,
            'early_stopping_patience': 10,
            'use_amp': True,
            'max_grad_norm': 1.0,
            'model_dir': 'models/superior'
        }

    def _setup_optimizer(self):
        """Setup advanced optimizer"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.01)

        if optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'radam':
            # Rectified Adam - more stable than Adam
            self.optimizer = torch.optim.RAdam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            # Default to AdamW
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

    def _setup_loss_function(self):
        """Setup loss function with label smoothing"""
        label_smoothing = self.config.get('label_smoothing', 0.1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_no_smooth = nn.CrossEntropyLoss()  # For validation

    def _setup_schedulers(self):
        """Setup learning rate schedulers"""
        scheduler_name = self.config.get('scheduler', 'cosine_warmup')

        if scheduler_name == 'cosine_warmup':
            # Cosine annealing with warmup
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.get('warmup_epochs', 5)
            )
            self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100  # Will be updated based on total epochs
            )
            self.use_warmup = True
        else:
            # Plateau scheduler as fallback
            self.main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            self.use_warmup = False

    def compute_class_weights(self, train_loader):
        """Compute class weights for imbalanced datasets"""
        label_counts = {}
        total_samples = 0

        for _, labels in train_loader:
            for label in labels:
                label_idx = label.item()
                label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
                total_samples += 1

        num_classes = len(label_counts)
        weights = torch.zeros(num_classes)

        for label_idx, count in label_counts.items():
            weights[label_idx] = total_samples / (count * num_classes)

        return weights.to(self.device)

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with advanced techniques"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        # Calculate metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')

        return epoch_loss, epoch_acc, epoch_f1

    def validate(self, val_loader, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                if self.use_amp:
                    with amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion_no_smooth(outputs, targets)  # No label smoothing for validation
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion_no_smooth(outputs, targets)

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')

        return epoch_loss, epoch_acc, epoch_f1, all_preds, all_targets

    def train(self, train_loader, val_loader, num_epochs=100, use_class_weights=True):
        """
        Main training loop with advanced features
        """
        print("🚀 Starting Superior Training Engine")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Setup class weights if needed
        if use_class_weights:
            print("Computing class weights...")
            class_weights = self.compute_class_weights(train_loader)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.config.get('label_smoothing', 0.1)
            )
            print(f"Class weights: {class_weights}")

        # Update scheduler T_max for cosine annealing
        if hasattr(self.main_scheduler, 'T_max'):
            self.main_scheduler.T_max = num_epochs

        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")

            # Learning rate scheduling
            if self.use_warmup and epoch <= self.config.get('warmup_epochs', 5):
                self.warmup_scheduler.step()
            elif hasattr(self.main_scheduler, 'step') and not isinstance(self.main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.main_scheduler.step()

            # Train epoch
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc, val_f1, val_preds, val_targets = self.validate(val_loader, epoch)

            # Update plateau scheduler
            if isinstance(self.main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.main_scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)
            self.history['epochs'].append(epoch)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {current_lr:.8f}")

            # Check for best model
            is_best = False
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                is_best = True
                self.patience_counter = 0

                # Save best model
                self.save_model('best_model.pt', epoch, is_best=True)
                print(f"🎉 New best model! F1: {val_f1:.4f}, Acc: {val_acc:.2f}%")

            else:
                self.patience_counter += 1

            # Save regular checkpoint
            if epoch % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pt', epoch)

            # Early stopping
            if self.patience_counter >= self.config.get('early_stopping_patience', 10):
                print(f"Early stopping triggered after {epoch} epochs")
                break

            # Generate classification report for best epochs
            if is_best:
                self.generate_classification_report(val_targets, val_preds, epoch)

        # Final model save and plots
        self.save_model('final_model.pt', epoch)
        self.plot_training_history()
        self.save_training_summary()

        print(f"\n🏁 Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.history

    def save_model(self, filename, epoch, is_best=False):
        """Save model checkpoint"""
        filepath = os.path.join(self.model_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.main_scheduler.state_dict(),
            'history': self.history,
            'config': self.config,
            'best_val_f1': self.best_val_f1,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)

        if is_best:
            print(f"💾 Saved best model: {filepath}")

    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.main_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', {})
        self.config = checkpoint.get('config', self.config)

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"✅ Loaded model from {filepath}")

    def generate_classification_report(self, y_true, y_pred, epoch):
        """Generate detailed classification report"""
        report = classification_report(y_true, y_pred, output_dict=True)

        # Save report
        report_path = os.path.join(self.model_dir, f'classification_report_epoch_{epoch}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()

    def plot_training_history(self):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Loss curves
        axes[0, 0].plot(self.history['epochs'], self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['epochs'], self.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.history['epochs'], self.history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(self.history['epochs'], self.history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # F1 Score curves
        axes[0, 2].plot(self.history['epochs'], self.history['train_f1'], label='Train F1', color='blue')
        axes[0, 2].plot(self.history['epochs'], self.history['val_f1'], label='Val F1', color='red')
        axes[0, 2].set_title('F1 Score Curves')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Learning rate
        axes[1, 0].plot(self.history['epochs'], self.history['learning_rates'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # Loss difference (overfitting indicator)
        loss_diff = [train - val for train, val in zip(self.history['train_loss'], self.history['val_loss'])]
        axes[1, 1].plot(self.history['epochs'], loss_diff, color='purple')
        axes[1, 1].set_title('Loss Difference (Train - Val)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True)

        # Training summary
        axes[1, 2].axis('off')
        summary_text = f"""
Training Summary

Best Validation Metrics:
• F1 Score: {self.best_val_f1:.4f}
• Accuracy: {self.best_val_acc:.2f}%
• Loss: {self.best_val_loss:.4f}

Final Epoch: {self.history['epochs'][-1] if self.history['epochs'] else 'N/A'}
Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}
Device: {self.device}

Configuration:
• Optimizer: {self.config.get('optimizer', 'adamw')}
• Learning Rate: {self.config.get('learning_rate', 0.001)}
• Weight Decay: {self.config.get('weight_decay', 0.01)}
• Label Smoothing: {self.config.get('label_smoothing', 0.1)}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Training plots saved to {self.model_dir}")

    def save_training_summary(self):
        """Save comprehensive training summary"""
        summary = {
            'best_metrics': {
                'val_f1': self.best_val_f1,
                'val_accuracy': self.best_val_acc,
                'val_loss': self.best_val_loss
            },
            'final_metrics': {
                'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
                'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
                'train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else None,
                'val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else None,
                'train_f1': self.history['train_f1'][-1] if self.history['train_f1'] else None,
                'val_f1': self.history['val_f1'][-1] if self.history['val_f1'] else None
            },
            'training_info': {
                'total_epochs': len(self.history['epochs']),
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'device': str(self.device),
                'mixed_precision': self.use_amp
            },
            'config': self.config,
            'history': self.history
        }

        summary_path = os.path.join(self.model_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📝 Training summary saved to {summary_path}")
