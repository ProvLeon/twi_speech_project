#!/usr/bin/env python3
"""
Superior Twi Speech Training Engine
==================================

This is the main training script for our superior speech recognition model
that outperforms the local_dialect_speech_model approach.

Key Improvements:
- Advanced multi-scale feature extraction
- Superior model architecture with attention mechanisms
- Enhanced training strategies with mixed precision
- Comprehensive data augmentation
- Intelligent learning rate scheduling
- Advanced regularization techniques
"""

import os
import sys
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import our superior components
from src.models.advanced_speech_model import SuperiorTwiSpeechModel
from src.features.advanced_feature_extractor import SuperiorFeatureExtractor
from src.features.dataset_utils import DatasetManager
from src.trainers.superior_trainer import SuperiorTrainer
from src.data.audio_recorder import interactive_recording_session


def setup_device():
    """Setup the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")

    return device


def load_or_extract_features(recordings_dir, output_dir, force_reextract=False):
    """
    Load existing features or extract new ones from recordings

    Args:
        recordings_dir: Directory containing audio recordings
        output_dir: Directory to save/load processed features
        force_reextract: Whether to force re-extraction even if features exist

    Returns:
        Dictionary containing dataset information
    """
    features_path = Path(output_dir) / 'features.npy'

    if features_path.exists() and not force_reextract:
        print(f"📁 Loading existing features from {output_dir}")
        try:
            # Load existing dataset info
            info_path = Path(output_dir) / 'dataset_info.json'
            if info_path.exists():
                with open(info_path, 'r') as f:
                    dataset_info = json.load(f)
                print("✅ Loaded existing dataset info")
                return dataset_info
        except Exception as e:
            print(f"⚠️  Error loading existing features: {e}")
            print("Proceeding with feature extraction...")

    print(f"🔄 Extracting features from {recordings_dir}")

    # Create feature extractor
    feature_extractor = SuperiorFeatureExtractor(
        output_dir=output_dir,
        max_length=200  # Standardize to 200 time frames
    )

    # Extract features from recordings
    dataset_info = feature_extractor.extract_features_from_recordings(
        recordings_dir=recordings_dir
    )

    # Save dataset info
    info_path = Path(output_dir) / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    return dataset_info


def create_model(feature_info, num_classes, config):
    """
    Create the superior model with optimal configuration

    Args:
        feature_info: Information about feature dimensions
        num_classes: Number of output classes
        config: Model configuration

    Returns:
        Superior model instance
    """
    input_dim = feature_info['total']

    model = SuperiorTwiSpeechModel(
        input_dim=input_dim,
        hidden_dim=config.get('hidden_dim', 256),
        num_classes=num_classes,
        num_conv_layers=config.get('num_conv_layers', 4),
        num_attention_layers=config.get('num_attention_layers', 3),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1)
    )

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🧠 Model Architecture:")
    print(f"   Input dimension: {input_dim}")
    print(f"   Hidden dimension: {config.get('hidden_dim', 256)}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    return model


def create_superior_config():
    """Create superior training configuration"""
    return {
        # Model architecture
        'hidden_dim': 256,
        'num_conv_layers': 4,
        'num_attention_layers': 3,
        'num_heads': 8,
        'dropout': 0.1,

        # Training parameters
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'cosine_warmup',
        'warmup_epochs': 5,
        'label_smoothing': 0.1,
        'early_stopping_patience': 15,
        'use_amp': True,
        'max_grad_norm': 1.0,

        # Dataset parameters
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'batch_size': 32,
        'num_workers': 4,
        'use_weighted_sampling': True,
        'augment_training': True,
        'augment_prob': 0.6,

        # Directories
        'model_dir': 'models/superior',
        'output_dir': 'data/processed_superior',
        'analysis_dir': 'analysis/superior'
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Superior Twi Speech Training Engine')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'record'],
                        help='Operation mode: train or record')
    parser.add_argument('--recordings_dir', type=str,
                        help='Directory containing audio recordings')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--force_reextract', action='store_true',
                        help='Force re-extraction of features')
    parser.add_argument('--config_file', type=str,
                        help='Path to JSON configuration file')
    parser.add_argument('--model_dir', type=str, default='models/superior',
                        help='Directory to save models')

    args = parser.parse_args()

    if args.mode == 'record':
        interactive_recording_session()
        return

    if not args.recordings_dir:
        parser.error("--recordings_dir is required for training mode")

    # Setup
    print("=" * 60)
    print("🎯 SUPERIOR TWI SPEECH TRAINING ENGINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        print(f"📋 Loaded configuration from {args.config_file}")
    else:
        config = create_superior_config()
        print("📋 Using default superior configuration")

    # Override config with command line arguments
    if args.batch_size != 32:
        config['batch_size'] = args.batch_size
    if args.learning_rate != 0.001:
        config['learning_rate'] = args.learning_rate
    if args.model_dir != 'models/superior':
        config['model_dir'] = args.model_dir

    # Create output directories
    for dir_key in ['model_dir', 'output_dir', 'analysis_dir']:
        os.makedirs(config[dir_key], exist_ok=True)

    # Setup device
    device = setup_device()

    # Step 1: Feature Extraction
    print("\n" + "="*50)
    print("📊 STEP 1: FEATURE EXTRACTION")
    print("="*50)

    dataset_info = load_or_extract_features(
        recordings_dir=args.recordings_dir,
        output_dir=config['output_dir'],
        force_reextract=args.force_reextract
    )

    if not dataset_info['features']:
        print("❌ No features extracted. Please check your recordings directory.")
        return

    print(f"✅ Extracted {len(dataset_info['features'])} feature samples")
    print(f"   Feature dimensions: {dataset_info['feature_info']['total']}")

    # Step 2: Dataset Preparation
    print("\n" + "="*50)
    print("📁 STEP 2: DATASET PREPARATION")
    print("="*50)

    # Create dataset manager
    dataset_manager = DatasetManager(config)

    # Load full dataset
    full_dataset = dataset_manager.load_dataset_from_directory(config['output_dir'])

    # Analyze dataset
    print("\n📊 Dataset Analysis:")
    dataset_manager.analyze_dataset(full_dataset, config['analysis_dir'])

    # Create stratified splits
    train_dataset, val_dataset, test_dataset = dataset_manager.create_stratified_splits(full_dataset)

    # Create data loaders
    dataloaders = dataset_manager.create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_weighted_sampling=config['use_weighted_sampling']
    )

    # Step 3: Model Creation
    print("\n" + "="*50)
    print("🧠 STEP 3: MODEL CREATION")
    print("="*50)

    model = create_model(
        feature_info=dataset_info['feature_info'],
        num_classes=full_dataset.get_num_classes(),
        config=config
    )

    # Step 4: Training Setup
    print("\n" + "="*50)
    print("🏋️  STEP 4: TRAINING SETUP")
    print("="*50)

    trainer = SuperiorTrainer(
        model=model,
        device=device,
        config=config
    )

    # Save configuration
    config_path = os.path.join(config['model_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Save dataset info to model directory
    dataset_info_path = os.path.join(config['model_dir'], 'dataset_info.json')
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"💾 Configuration saved to {config_path}")
    print(f"📊 Dataset info saved to {dataset_info_path}")

    # Step 5: Training
    print("\n" + "="*50)
    print("🚀 STEP 5: SUPERIOR TRAINING")
    print("="*50)

    try:
        history = trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=args.epochs,
            use_class_weights=config.get('use_class_weights', True)
        )

        print("\n🎉 Training completed successfully!")

        # Step 6: Final Evaluation
        print("\n" + "="*50)
        print("📈 STEP 6: FINAL EVALUATION")
        print("="*50)

        if 'test' in dataloaders:
            # Load best model for final evaluation
            best_model_path = os.path.join(config['model_dir'], 'best_model.pt')
            if os.path.exists(best_model_path):
                trainer.load_model(best_model_path)
                print("✅ Loaded best model for final evaluation")

                # Evaluate on test set
                test_loss, test_acc, test_f1, test_preds, test_targets = trainer.validate(
                    dataloaders['test'], epoch=0
                )

                print(f"🏆 Final Test Results:")
                print(f"   Test Loss: {test_loss:.4f}")
                print(f"   Test Accuracy: {test_acc:.2f}%")
                print(f"   Test F1-Score: {test_f1:.4f}")

                # Generate final classification report
                trainer.generate_classification_report(test_targets, test_preds, epoch='final')

        # Step 7: Model Comparison
        print("\n" + "="*50)
        print("🔄 STEP 7: PERFORMANCE COMPARISON")
        print("="*50)

        print("📊 Superior Engine Performance Summary:")
        print(f"   Best Validation F1-Score: {trainer.best_val_f1:.4f}")
        print(f"   Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
        print(f"   Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Training Device: {device}")

        # Save final summary
        final_summary = {
            'training_completed': True,
            'completion_time': datetime.now().isoformat(),
            'best_metrics': {
                'val_f1': trainer.best_val_f1,
                'val_accuracy': trainer.best_val_acc,
                'val_loss': trainer.best_val_loss
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'input_dim': dataset_info['feature_info']['total'],
                'num_classes': full_dataset.get_num_classes()
            },
            'dataset_info': {
                'total_samples': len(full_dataset),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset) if test_dataset else 0
            }
        }

        summary_path = os.path.join(config['model_dir'], 'final_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2)

        print(f"\n💾 Final summary saved to {summary_path}")
        print(f"📁 All outputs saved to {config['model_dir']}")

        print("\n" + "="*60)
        print("🎯 SUPERIOR TRAINING ENGINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Your superior model is ready at: {config['model_dir']}")
        print("This model should outperform the local_dialect_speech_model approach.")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💾 Saving current progress...")
        trainer.save_model('interrupted_model.pt', epoch=len(trainer.history.get('epochs', [0])))

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💾 Saving debug information...")

        # Save debug info
        debug_info = {
            'error': str(e),
            'error_type': type(e).__name__,
            'config': config,
            'history': trainer.history if 'trainer' in locals() else {}
        }

        debug_path = os.path.join(config['model_dir'], 'debug_info.json')
        with open(debug_path, 'w') as f:
            json.dump(debug_info, f, indent=2)

        raise


if __name__ == "__main__":
    main()
