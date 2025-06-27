#!/usr/bin/env python3
"""
Comprehensive Training Scenarios Example for Twi Speech Recognition Engine

This script demonstrates all available training scenarios and how to use them:
1. Speech-to-Text (Twi Audio → Twi Text)
2. Speech Translation (Twi Audio → English Text)
3. Multilingual (Twi Audio → Both Twi/English)
4. Cross-lingual (Twi Audio + Text → English)

Each scenario shows:
- When to use it
- How the data flows
- Expected inputs and outputs
- Training configuration
- Example usage

Run this script to see practical examples of each training approach.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("🎯 Twi Speech Training Scenarios Comprehensive Guide")
print("=" * 70)

def setup_environment():
    """Setup the environment and load necessary modules"""
    print("\n1. Setting up environment...")
    print("-" * 40)

    try:
        from src.config.config_loader import load_config
        from src.utils.env_loader import load_env
        from src.data.data_manager import TwiDataManager
        from src.data.training_datasets import create_dataset

        # Load environment variables
        load_env()
        print("✓ Environment variables loaded")

        # Load configuration
        config_path = current_dir / "configs" / "default_config.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            config = {
                'audio': {'sample_rate': 16000},
                'text': {'language': 'tw'},
                'mongodb': {},
                'r2_storage': {},
                'cache_dir': 'data/cache',
                'recordings_dir': 'data/recordings',
                'script_file': 'script_actual.ts'
            }
        print("✓ Configuration loaded")

        return config, TwiDataManager

    except Exception as e:
        print(f"✗ Setup error: {e}")
        return None, None

def demonstrate_scenario_1_speech_to_text(manager):
    """Scenario 1: Standard Speech Recognition (Twi Audio → Twi Text)"""
    print("\n" + "="*60)
    print("📝 SCENARIO 1: Speech-to-Text (Standard ASR)")
    print("="*60)

    print("\n🎯 Use Case:")
    print("   Convert Twi speech directly to Twi text")
    print("   Perfect for: Transcription services, voice notes, accessibility")

    print("\n📊 Data Flow:")
    print("   Input:  Twi Audio (WAV/M4A files)")
    print("   Target: prompt_text (Twi transcription)")
    print("   Output: Twi text transcription")

    print("\n📋 Example Data:")
    print("   Audio: 'Bue adwadie app no' (spoken)")
    print("   Target: 'Bue adwadie app no' (text)")
    print("   Model learns: Audio → Same text in Twi")

    print("\n⚙️ Configuration:")
    print("   training_type = 'speech_to_text'")
    print("   Uses: sample.prompt_text or sample.transcription")
    print("   Best models: Wav2Vec2, Whisper")

    try:
        print("\n🔄 Running Speech-to-Text pipeline...")
        training_data = manager.prepare_training_data(
            samples=manager.load_local_data() or manager._create_dummy_samples(),
            feature_type="wav2vec2",
            training_type="speech_to_text",
            apply_augmentation=True,
            augmentation_factor=2,
            max_workers=2
        )

        print("✅ Speech-to-Text preparation completed!")
        print(f"   Samples: {training_data['total_samples']}")
        print(f"   Target: Twi text transcriptions")

        # Show scenario info
        scenario = training_data['metadata']['training_scenarios']
        print(f"   Description: {scenario['description']}")

    except Exception as e:
        print(f"✗ Error in scenario 1: {e}")

def demonstrate_scenario_2_translation(manager):
    """Scenario 2: Speech Translation (Twi Audio → English Text)"""
    print("\n" + "="*60)
    print("🌍 SCENARIO 2: Speech Translation")
    print("="*60)

    print("\n🎯 Use Case:")
    print("   Translate Twi speech directly to English text")
    print("   Perfect for: Real-time translation, international communication")

    print("\n📊 Data Flow:")
    print("   Input:  Twi Audio (WAV/M4A files)")
    print("   Target: meaning (English translation)")
    print("   Output: English text translation")

    print("\n📋 Example Data:")
    print("   Audio: 'Bue adwadie app no' (spoken)")
    print("   Target: 'Open the shopping app' (English)")
    print("   Model learns: Twi Audio → English meaning")

    print("\n⚙️ Configuration:")
    print("   training_type = 'translation'")
    print("   Uses: sample.meaning (English)")
    print("   Best models: Whisper (multilingual), SeamlessM4T")

    try:
        print("\n🔄 Running Speech Translation pipeline...")
        training_data = manager.prepare_training_data(
            samples=manager.load_local_data() or manager._create_dummy_samples(),
            feature_type="wav2vec2",
            training_type="translation",
            apply_augmentation=True,
            augmentation_factor=2,
            max_workers=2
        )

        print("✅ Speech Translation preparation completed!")
        print(f"   Samples: {training_data['total_samples']}")
        print(f"   Target: English translations")

        scenario = training_data['metadata']['training_scenarios']
        print(f"   Description: {scenario['description']}")

    except Exception as e:
        print(f"✗ Error in scenario 2: {e}")

def demonstrate_scenario_3_multilingual(manager):
    """Scenario 3: Multilingual Training (Twi Audio → Both Twi/English)"""
    print("\n" + "="*60)
    print("🔄 SCENARIO 3: Multilingual Model")
    print("="*60)

    print("\n🎯 Use Case:")
    print("   One model that can do both transcription AND translation")
    print("   Perfect for: Unified systems, task-switching applications")

    print("\n📊 Data Flow:")
    print("   Input:  Twi Audio + Task Token")
    print("   Target: Either Twi text OR English text (task-dependent)")
    print("   Output: Task-specific text")

    print("\n📋 Example Data:")
    print("   Audio: 'Bue adwadie app no' (spoken)")
    print("   Task 1: '<transcribe_twi>' → 'Bue adwadie app no'")
    print("   Task 2: '<translate_to_english>' → 'Open the shopping app'")
    print("   Model learns: One model, two tasks")

    print("\n⚙️ Configuration:")
    print("   training_type = 'multilingual'")
    print("   Uses: Both sample.prompt_text AND sample.meaning")
    print("   task_mixing_ratio: 0.5 = 50% each task")
    print("   Best models: Whisper, SeamlessM4T")

    try:
        print("\n🔄 Running Multilingual pipeline...")
        training_data = manager.prepare_training_data(
            samples=manager.load_local_data() or manager._create_dummy_samples(),
            feature_type="wav2vec2",
            training_type="multilingual",
            apply_augmentation=True,
            augmentation_factor=2,
            max_workers=2,
            task_mixing_ratio=0.6  # 60% English, 40% Twi
        )

        print("✅ Multilingual preparation completed!")
        print(f"   Samples: {training_data['total_samples']}")
        print(f"   Target: Mixed Twi/English (60% English)")

        scenario = training_data['metadata']['training_scenarios']
        print(f"   Description: {scenario['description']}")

    except Exception as e:
        print(f"✗ Error in scenario 3: {e}")

def demonstrate_scenario_4_cross_lingual(manager):
    """Scenario 4: Cross-lingual with Text Input (Twi Audio + Text → English)"""
    print("\n" + "="*60)
    print("🔗 SCENARIO 4: Cross-lingual Multimodal")
    print("="*60)

    print("\n🎯 Use Case:")
    print("   Use both audio AND text for better translation")
    print("   Perfect for: High-accuracy translation, noisy audio scenarios")

    print("\n📊 Data Flow:")
    print("   Input:  Twi Audio + Twi Text (two modalities)")
    print("   Target: English translation")
    print("   Output: English text (enhanced by dual input)")

    print("\n📋 Example Data:")
    print("   Audio: 'Bue adwadie app no' (spoken)")
    print("   Text:  'Bue adwadie app no' (written)")
    print("   Target: 'Open the shopping app' (English)")
    print("   Model learns: Audio + Text → Better English")

    print("\n⚙️ Configuration:")
    print("   training_type = 'cross_lingual'")
    print("   Uses: sample.audio + sample.prompt_text → sample.meaning")
    print("   Best models: Custom multimodal architectures")

    try:
        print("\n🔄 Running Cross-lingual pipeline...")
        training_data = manager.prepare_training_data(
            samples=manager.load_local_data() or manager._create_dummy_samples(),
            feature_type="wav2vec2",
            training_type="cross_lingual",
            apply_augmentation=False,  # More conservative for multimodal
            augmentation_factor=1,
            max_workers=2
        )

        print("✅ Cross-lingual preparation completed!")
        print(f"   Samples: {training_data['total_samples']}")
        print(f"   Target: English (with dual input)")

        scenario = training_data['metadata']['training_scenarios']
        print(f"   Description: {scenario['description']}")

    except Exception as e:
        print(f"✗ Error in scenario 4: {e}")

def demonstrate_dataset_creation():
    """Show how to create PyTorch datasets for each scenario"""
    print("\n" + "="*60)
    print("🏗️ DATASET CREATION EXAMPLES")
    print("="*60)

    print("\n📚 PyTorch Dataset Creation:")
    print("After preparing training data, create PyTorch datasets:")

    print("\n# Example 1: Speech-to-Text Dataset")
    print("from transformers import Wav2Vec2Processor")
    print("from src.data.training_datasets import create_dataset")
    print("")
    print("processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')")
    print("dataset = create_dataset(")
    print("    samples=samples,")
    print("    dataset_type='speech_to_text',")
    print("    processor=processor,")
    print("    audio_processor=audio_proc,")
    print("    text_processor=text_proc,")
    print("    is_training=True")
    print(")")

    print("\n# Example 2: Translation Dataset")
    print("dataset = create_dataset(")
    print("    samples=samples,")
    print("    dataset_type='translation',")
    print("    processor=whisper_processor,")
    print("    audio_processor=audio_proc,")
    print("    text_processor=text_proc")
    print(")")

    print("\n# Example 3: Multilingual Dataset")
    print("dataset = create_dataset(")
    print("    samples=samples,")
    print("    dataset_type='multilingual',")
    print("    processor=multilingual_processor,")
    print("    audio_processor=audio_proc,")
    print("    text_processor=text_proc,")
    print("    task_mixing_ratio=0.7  # 70% English")
    print(")")

def show_data_structure_examples():
    """Show what the actual data looks like in each scenario"""
    print("\n" + "="*60)
    print("🔍 DATA STRUCTURE EXAMPLES")
    print("="*60)

    print("\n📋 Sample AudioSample Structure:")
    print("AudioSample {")
    print("    audio_path: 'data/recordings/TWI_Speaker_001/ScriptD_15_...m4a'")
    print("    prompt_text: 'Me werɛ aho pa ara'  # Twi text")
    print("    meaning: 'I am very worried'      # English translation")
    print("    transcription: 'Me werɛ aho pa ara'  # Same as prompt_text")
    print("    speaker_id: 'TWI_Speaker_001'")
    print("    dialect: 'Asante'")
    print("    duration: 2.5")
    print("}")

    print("\n🎯 Training Targets by Scenario:")
    print("")
    print("Scenario 1 (Speech-to-Text):")
    print("  Input:  audio_path → load audio")
    print("  Target: prompt_text → 'Me werɛ aho pa ara'")
    print("")
    print("Scenario 2 (Translation):")
    print("  Input:  audio_path → load audio")
    print("  Target: meaning → 'I am very worried'")
    print("")
    print("Scenario 3 (Multilingual):")
    print("  Input:  audio_path → load audio")
    print("  Target: prompt_text OR meaning (task-dependent)")
    print("         '<transcribe_twi> Me werɛ aho pa ara'")
    print("         '<translate_to_english> I am very worried'")
    print("")
    print("Scenario 4 (Cross-lingual):")
    print("  Input:  audio_path + prompt_text → dual modality")
    print("  Target: meaning → 'I am very worried'")

def show_model_recommendations():
    """Show recommended models for each scenario"""
    print("\n" + "="*60)
    print("🤖 MODEL RECOMMENDATIONS")
    print("="*60)

    scenarios = {
        "Speech-to-Text (Twi → Twi)": {
            "models": ["Wav2Vec2", "Whisper (small)", "Custom transformer"],
            "pros": ["Fast inference", "Good for low-resource languages", "Lightweight"],
            "cons": ["Requires Twi tokenizer", "Limited by training data"],
            "best_for": "Real-time transcription, mobile apps"
        },
        "Speech Translation (Twi → English)": {
            "models": ["Whisper (medium/large)", "SeamlessM4T", "Custom seq2seq"],
            "pros": ["Pre-trained multilingual", "Good translation quality", "Robust"],
            "cons": ["Larger model size", "Higher latency"],
            "best_for": "High-quality translation, research"
        },
        "Multilingual (Twi → Both)": {
            "models": ["Whisper (large)", "SeamlessM4T", "mT5 + speech encoder"],
            "pros": ["Single model", "Task flexibility", "Resource efficient"],
            "cons": ["Complex training", "Task interference possible"],
            "best_for": "Unified systems, production deployments"
        },
        "Cross-lingual (Audio+Text → English)": {
            "models": ["Custom multimodal", "CLIP + T5", "Dual-encoder architecture"],
            "pros": ["Highest accuracy", "Robust to noise", "Rich representations"],
            "cons": ["Complex architecture", "More data needed", "Research-stage"],
            "best_for": "Research, high-accuracy applications"
        }
    }

    for scenario, info in scenarios.items():
        print(f"\n📊 {scenario}:")
        print(f"   Models: {', '.join(info['models'])}")
        print(f"   Pros: {', '.join(info['pros'])}")
        print(f"   Cons: {', '.join(info['cons'])}")
        print(f"   Best for: {info['best_for']}")

def show_practical_usage():
    """Show how to actually use this in practice"""
    print("\n" + "="*60)
    print("🚀 PRACTICAL USAGE GUIDE")
    print("="*60)

    print("\n🔧 Step-by-Step Usage:")
    print("")
    print("1. Choose Your Scenario:")
    print("   - Need Twi transcription? → Use 'speech_to_text'")
    print("   - Need English translation? → Use 'translation'")
    print("   - Need both? → Use 'multilingual'")
    print("   - Have text + audio? → Use 'cross_lingual'")
    print("")
    print("2. Run the Pipeline:")
    print("   python main_engine.py")
    print("   → Select option 5 (Complete Processing Pipeline)")
    print("   → Choose your training type")
    print("   → Configure augmentation and features")
    print("")
    print("3. Get Training Data:")
    print("   data/processed/")
    print("   ├── train_YYYYMMDD_HHMMSS.pkl")
    print("   ├── validation_YYYYMMDD_HHMMSS.pkl")
    print("   ├── test_YYYYMMDD_HHMMSS.pkl")
    print("   └── metadata_YYYYMMDD_HHMMSS.json")
    print("")
    print("4. Train Your Model:")
    print("   → Use prepared data with your favorite framework")
    print("   → PyTorch, HuggingFace Transformers, etc.")

def main():
    """Main function to run all demonstrations"""
    try:
        # Setup
        config, TwiDataManager = setup_environment()
        if not TwiDataManager:
            print("❌ Failed to setup environment")
            return

        manager = TwiDataManager(config)
        print("✓ TwiDataManager initialized")

        # Run all demonstrations
        demonstrate_scenario_1_speech_to_text(manager)
        demonstrate_scenario_2_translation(manager)
        demonstrate_scenario_3_multilingual(manager)
        demonstrate_scenario_4_cross_lingual(manager)

        # Show additional info
        demonstrate_dataset_creation()
        show_data_structure_examples()
        show_model_recommendations()
        show_practical_usage()

        # Summary
        print("\n" + "="*70)
        print("📋 SUMMARY")
        print("="*70)

        print("\n✅ All Training Scenarios Demonstrated:")
        print("1. ✓ Speech-to-Text: Twi Audio → Twi Text")
        print("2. ✓ Translation: Twi Audio → English Text")
        print("3. ✓ Multilingual: Twi Audio → Both")
        print("4. ✓ Cross-lingual: Twi Audio + Text → English")

        print("\n🎯 Choose Based on Your Use Case:")
        print("• Transcription service → Speech-to-Text")
        print("• Translation service → Translation")
        print("• Unified system → Multilingual")
        print("• Research/accuracy → Cross-lingual")

        print("\n🚀 Ready to Train!")
        print("Use: python main_engine.py → option 5")
        print("Then: python train.py with your prepared data")

        print("\n📚 Data Structure:")
        print("prompt_text (Twi) + meaning (English) = Complete training data")

    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
