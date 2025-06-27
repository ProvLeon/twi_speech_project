import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Tuple, Optional, Union
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class SuperiorTwiDataset(Dataset):
    """
    Advanced dataset class with intelligent augmentation and preprocessing
    """

    def __init__(self,
                 features: List[np.ndarray],
                 labels: List[str],
                 label_to_idx: Optional[Dict[str, int]] = None,
                 augment: bool = True,
                 augment_prob: float = 0.6,
                 max_length: Optional[int] = None,
                 normalize: bool = True):
        """
        Initialize superior dataset

        Args:
            features: List of feature arrays
            labels: List of string labels
            label_to_idx: Mapping from labels to indices
            augment: Whether to apply augmentation
            augment_prob: Probability of applying augmentation
            max_length: Maximum sequence length for padding/truncation
            normalize: Whether to normalize features
        """
        self.features = features
        self.labels = labels
        self.augment = augment
        self.augment_prob = augment_prob
        self.max_length = max_length
        self.normalize = normalize

        # Create label mapping
        if label_to_idx is None:
            unique_labels = sorted(set(labels))
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.label_indices = [self.label_to_idx[label] for label in labels]

        # Standardize features
        self._standardize_features()

        # Initialize augmentation
        if self.augment:
            from .advanced_augmentation import AdvancedAugmentation
            self.augmenter = AdvancedAugmentation()

    def _standardize_features(self):
        """Standardize feature dimensions and apply normalization"""
        if not self.features:
            return

        # Find the common dimensions
        feature_shapes = [f.shape for f in self.features]
        max_freq_dim = max(shape[0] for shape in feature_shapes)

        if self.max_length is None:
            self.max_length = max(shape[1] for shape in feature_shapes)

        # Standardize all features
        standardized_features = []
        for feature in self.features:
            # Pad frequency dimension if needed
            if feature.shape[0] < max_freq_dim:
                padding = np.zeros((max_freq_dim - feature.shape[0], feature.shape[1]))
                feature = np.vstack([feature, padding])

            # Handle time dimension
            if feature.shape[1] > self.max_length:
                # Truncate
                feature = feature[:, :self.max_length]
            elif feature.shape[1] < self.max_length:
                # Pad
                padding = np.zeros((feature.shape[0], self.max_length - feature.shape[1]))
                feature = np.hstack([feature, padding])

            # Normalize if requested
            if self.normalize:
                feature = self._normalize_feature(feature)

            standardized_features.append(feature)

        self.features = standardized_features
        self.feature_shape = (max_freq_dim, self.max_length)

    def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        """Robust feature normalization"""
        # Use robust statistics for normalization
        median = np.median(feature, axis=1, keepdims=True)
        mad = np.median(np.abs(feature - median), axis=1, keepdims=True)

        # Avoid division by zero
        mad = np.where(mad == 0, 1, mad)

        # Normalize using MAD (more robust than std)
        normalized = (feature - median) / (1.4826 * mad)

        # Clip extreme values
        normalized = np.clip(normalized, -5, 5)

        return normalized

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].copy()
        label = self.label_indices[idx]

        # Apply augmentation with probability
        if self.augment and np.random.random() < self.augment_prob:
            feature = self.augmenter.augment(feature)

        # Convert to tensors
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature_tensor, label_tensor

    def get_num_classes(self):
        return len(self.label_to_idx)

    def get_class_distribution(self):
        """Get class distribution for analysis"""
        counter = Counter(self.labels)
        return dict(counter)

    def get_class_weights(self):
        """Compute class weights for balanced training"""
        class_counts = Counter(self.label_indices)
        total_samples = len(self.label_indices)
        num_classes = len(class_counts)

        weights = torch.zeros(num_classes)
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (count * num_classes)

        return weights

    def plot_class_distribution(self, save_path: Optional[str] = None):
        """Plot class distribution"""
        dist = self.get_class_distribution()

        plt.figure(figsize=(12, 6))
        labels = list(dist.keys())
        counts = list(dist.values())

        plt.bar(labels, counts)
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


class DatasetManager:
    """
    Advanced dataset management with intelligent splitting and analysis
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize dataset manager

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()

    def _default_config(self):
        """Default configuration for dataset management"""
        return {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'random_seed': 42,
            'stratify': True,
            'min_samples_per_class': 5,
            'augment_training': True,
            'augment_prob': 0.6,
            'normalize_features': True,
            'max_length': None
        }

    def load_dataset_from_directory(self,
                                  data_dir: str,
                                  metadata_file: Optional[str] = None) -> SuperiorTwiDataset:
        """
        Load dataset from directory structure or metadata file

        Args:
            data_dir: Directory containing processed features
            metadata_file: Optional metadata CSV file

        Returns:
            SuperiorTwiDataset instance
        """
        # Load features and labels
        features_path = os.path.join(data_dir, 'features.npy')
        labels_path = os.path.join(data_dir, 'labels.npy')
        label_map_path = os.path.join(data_dir, 'label_map.json')

        if not all(os.path.exists(p) for p in [features_path, labels_path]):
            raise FileNotFoundError("Required dataset files not found")

        # Load data
        features = np.load(features_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)

        # Load label mapping if available
        label_to_idx = None
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                label_to_idx = json.load(f)

        # Create dataset
        dataset = SuperiorTwiDataset(
            features=features.tolist() if isinstance(features, np.ndarray) else features,
            labels=labels.tolist() if isinstance(labels, np.ndarray) else labels,
            label_to_idx=label_to_idx,
            augment=False,  # Augmentation will be enabled per split
            max_length=self.config.get('max_length'),
            normalize=self.config.get('normalize_features', True)
        )

        return dataset

    def create_stratified_splits(self,
                               dataset: SuperiorTwiDataset) -> Tuple[SuperiorTwiDataset, SuperiorTwiDataset, SuperiorTwiDataset]:
        """
        Create stratified train/validation/test splits

        Args:
            dataset: Input dataset

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Get indices and labels
        indices = list(range(len(dataset)))
        labels = dataset.label_indices

        # Check minimum samples per class
        class_counts = Counter(labels)
        min_count = min(class_counts.values())

        if min_count < self.config.get('min_samples_per_class', 5):
            print(f"Warning: Some classes have fewer than {self.config['min_samples_per_class']} samples")

        # First split: separate test set
        train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
            indices, labels,
            test_size=self.config['test_ratio'],
            stratify=labels if self.config.get('stratify', True) else None,
            random_state=self.config.get('random_seed', 42)
        )

        # Second split: separate train and validation
        val_ratio_adjusted = self.config['val_ratio'] / (self.config['train_ratio'] + self.config['val_ratio'])
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels if self.config.get('stratify', True) else None,
            random_state=self.config.get('random_seed', 42)
        )

        # Create split datasets
        train_dataset = self._create_subset_dataset(dataset, train_indices, augment=True)
        val_dataset = self._create_subset_dataset(dataset, val_indices, augment=False)
        test_dataset = self._create_subset_dataset(dataset, test_indices, augment=False)

        # Print split information
        print(f"Dataset splits created:")
        print(f"  Training: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

        # Print class distribution for each split
        self._print_split_distributions(train_dataset, val_dataset, test_dataset)

        return train_dataset, val_dataset, test_dataset

    def _create_subset_dataset(self,
                             original_dataset: SuperiorTwiDataset,
                             indices: List[int],
                             augment: bool = False) -> SuperiorTwiDataset:
        """Create a subset dataset from indices"""
        subset_features = [original_dataset.features[i] for i in indices]
        subset_labels = [original_dataset.labels[i] for i in indices]

        return SuperiorTwiDataset(
            features=subset_features,
            labels=subset_labels,
            label_to_idx=original_dataset.label_to_idx,
            augment=augment,
            augment_prob=self.config.get('augment_prob', 0.6),
            max_length=original_dataset.max_length,
            normalize=False  # Already normalized in original dataset
        )

    def _print_split_distributions(self, train_ds, val_ds, test_ds):
        """Print class distributions for each split"""
        print("\nClass distributions:")

        for name, dataset in [("Train", train_ds), ("Validation", val_ds), ("Test", test_ds)]:
            dist = dataset.get_class_distribution()
            print(f"\n{name}:")
            for label, count in sorted(dist.items()):
                percentage = (count / len(dataset)) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")

    def create_cross_validation_splits(self,
                                     dataset: SuperiorTwiDataset,
                                     n_folds: int = 5) -> List[Tuple[SuperiorTwiDataset, SuperiorTwiDataset]]:
        """
        Create cross-validation splits

        Args:
            dataset: Input dataset
            n_folds: Number of CV folds

        Returns:
            List of (train_dataset, val_dataset) tuples
        """
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.config.get('random_seed', 42)
        )

        indices = list(range(len(dataset)))
        labels = dataset.label_indices

        cv_splits = []

        for fold, (train_indices, val_indices) in enumerate(skf.split(indices, labels)):
            train_dataset = self._create_subset_dataset(dataset, train_indices.tolist(), augment=True)
            val_dataset = self._create_subset_dataset(dataset, val_indices.tolist(), augment=False)

            cv_splits.append((train_dataset, val_dataset))

            print(f"Fold {fold + 1}: Train={len(train_dataset)}, Val={len(val_dataset)}")

        return cv_splits

    def create_dataloaders(self,
                          train_dataset: SuperiorTwiDataset,
                          val_dataset: SuperiorTwiDataset,
                          test_dataset: Optional[SuperiorTwiDataset] = None,
                          batch_size: int = 32,
                          num_workers: int = 4,
                          use_weighted_sampling: bool = True) -> Dict[str, DataLoader]:
        """
        Create optimized data loaders

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Optional test dataset
            batch_size: Batch size
            num_workers: Number of worker processes
            use_weighted_sampling: Whether to use weighted sampling for training

        Returns:
            Dictionary of data loaders
        """
        # Create weighted sampler for training if requested
        train_sampler = None
        if use_weighted_sampling:
            class_weights = train_dataset.get_class_weights()
            sample_weights = [class_weights[label_idx] for label_idx in train_dataset.label_indices]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

        # Create data loaders
        dataloaders = {}

        # Training loader
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,  # Drop last incomplete batch for consistent training
            persistent_workers=num_workers > 0
        )

        # Validation loader
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )

        # Test loader if provided
        if test_dataset is not None:
            dataloaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0
            )

        print(f"Created data loaders:")
        print(f"  Training batches: {len(dataloaders['train'])}")
        print(f"  Validation batches: {len(dataloaders['val'])}")
        if 'test' in dataloaders:
            print(f"  Test batches: {len(dataloaders['test'])}")

        return dataloaders

    def analyze_dataset(self, dataset: SuperiorTwiDataset, save_dir: Optional[str] = None):
        """
        Comprehensive dataset analysis

        Args:
            dataset: Dataset to analyze
            save_dir: Directory to save analysis plots
        """
        print("=== Dataset Analysis ===")
        print(f"Total samples: {len(dataset)}")
        print(f"Number of classes: {dataset.get_num_classes()}")
        print(f"Feature shape: {dataset.feature_shape}")

        # Class distribution
        class_dist = dataset.get_class_distribution()
        print(f"\nClass distribution:")
        for label, count in sorted(class_dist.items()):
            percentage = (count / len(dataset)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")

        # Calculate class balance metrics
        counts = list(class_dist.values())
        balance_ratio = min(counts) / max(counts)
        print(f"\nBalance ratio (min/max): {balance_ratio:.3f}")

        if balance_ratio < 0.5:
            print("⚠️  Dataset is imbalanced - consider using weighted sampling")

        # Feature statistics
        all_features = np.array([dataset.features[i] for i in range(min(1000, len(dataset)))])
        print(f"\nFeature statistics (sample of {len(all_features)} features):")
        print(f"  Mean: {all_features.mean():.4f}")
        print(f"  Std: {all_features.std():.4f}")
        print(f"  Min: {all_features.min():.4f}")
        print(f"  Max: {all_features.max():.4f}")

        # Save visualizations if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # Class distribution plot
            dataset.plot_class_distribution(
                save_path=os.path.join(save_dir, 'class_distribution.png')
            )

            # Feature distribution plot
            self._plot_feature_statistics(all_features, save_dir)

        print(f"\n✅ Dataset analysis complete")
        if save_dir:
            print(f"📊 Plots saved to {save_dir}")

    def _plot_feature_statistics(self, features: np.ndarray, save_dir: str):
        """Plot feature statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Feature value distribution
        axes[0, 0].hist(features.flatten(), bins=50, alpha=0.7)
        axes[0, 0].set_title('Feature Value Distribution')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')

        # Feature mean across samples
        feature_means = features.mean(axis=(0, 2))  # Mean across batch and time
        axes[0, 1].plot(feature_means)
        axes[0, 1].set_title('Mean Feature Values Across Frequency Bins')
        axes[0, 1].set_xlabel('Frequency Bin')
        axes[0, 1].set_ylabel('Mean Value')

        # Feature variance across samples
        feature_vars = features.var(axis=(0, 2))
        axes[1, 0].plot(feature_vars)
        axes[1, 0].set_title('Feature Variance Across Frequency Bins')
        axes[1, 0].set_xlabel('Frequency Bin')
        axes[1, 0].set_ylabel('Variance')

        # Sample feature map
        if len(features) > 0:
            sample_idx = np.random.randint(0, len(features))
            im = axes[1, 1].imshow(features[sample_idx], aspect='auto', origin='lower')
            axes[1, 1].set_title(f'Sample Feature Map (Sample {sample_idx})')
            axes[1, 1].set_xlabel('Time Frames')
            axes[1, 1].set_ylabel('Frequency Bins')
            plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_dataset_info(self,
                         dataset: SuperiorTwiDataset,
                         filepath: str,
                         additional_info: Optional[Dict] = None):
        """
        Save comprehensive dataset information

        Args:
            dataset: Dataset to save info for
            filepath: Path to save JSON file
            additional_info: Additional information to include
        """
        info = {
            'dataset_info': {
                'total_samples': len(dataset),
                'num_classes': dataset.get_num_classes(),
                'feature_shape': dataset.feature_shape,
                'class_names': list(dataset.label_to_idx.keys()),
                'class_distribution': dataset.get_class_distribution()
            },
            'config': self.config,
            'label_mapping': dataset.label_to_idx
        }

        if additional_info:
            info.update(additional_info)

        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"📝 Dataset info saved to {filepath}")
