"""
Training Dataset Classes for Twi Speech Recognition Engine

This module provides different dataset classes for various training scenarios:
1. Speech-to-Text (Twi Audio → Twi Text)
2. Speech Translation (Twi Audio → English Text)
3. Multilingual (Twi Audio → Both Twi and English)
4. Cross-lingual (Twi Audio + Text → English)

Each dataset class handles the specific data format needed for different model architectures.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from transformers import Wav2Vec2Processor, WhisperProcessor

from .structures import AudioSample, TrainingExample

logger = logging.getLogger(__name__)


class TwiSpeechToTextDataset(Dataset):
    """
    Dataset for Twi Speech-to-Text (Audio → Twi Text)

    Use case: Standard ASR where we want to transcribe Twi speech to Twi text
    Target: prompt_text (Twi)
    """

    def __init__(self,
                 samples: List[AudioSample],
                 processor: Union[Wav2Vec2Processor, WhisperProcessor],
                 audio_processor,  # TwiAudioProcessor - avoiding circular import
                 text_processor,   # TwiTextProcessor - avoiding circular import
                 is_training: bool = True,
                 max_audio_length: int = 320000,  # 20 seconds at 16kHz
                 max_text_length: int = 512):

        self.samples = samples
        self.processor = processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.is_training = is_training
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length

        # Filter samples that have Twi text
        self.valid_samples = [s for s in samples if s.prompt_text or s.transcription]

        logger.info(f"TwiSpeechToTextDataset: {len(self.valid_samples)}/{len(samples)} valid samples")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.valid_samples[idx]

        try:
            # Load and preprocess audio
            audio, sr = self.audio_processor.load_audio(sample.audio_path)
            audio = self.audio_processor.preprocess_audio(
                audio, sr, apply_augmentation=self.is_training
            )

            # Use Twi text as target
            twi_text = sample.prompt_text or sample.transcription
            normalized_text = self.text_processor.normalize_text(twi_text)

            # Process audio
            audio_inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_audio_length
            )

            # Process text labels
            if hasattr(self.processor, 'tokenizer'):
                # For Wav2Vec2 with tokenizer
                text_inputs = self.processor.tokenizer(
                    normalized_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length
                )
                labels = text_inputs.input_ids.squeeze()
            else:
                # For other processors (like Whisper)
                labels = self.processor.tokenizer.encode(normalized_text, return_tensors="pt").squeeze()

            return {
                'input_values': audio_inputs.input_values.squeeze(),
                'attention_mask': audio_inputs.attention_mask.squeeze() if hasattr(audio_inputs, 'attention_mask') else None,
                'labels': labels,
                'twi_text': normalized_text,
                'speaker_id': sample.speaker_id,
                'dialect': sample.dialect,
                'sample_id': sample.prompt_id
            }

        except Exception as e:
            logger.error(f"Error processing sample {idx} for speech-to-text: {e}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample for error cases"""
        dummy_audio = np.zeros(16000)  # 1 second of silence
        dummy_text = "dummy"

        audio_inputs = self.processor(
            dummy_audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        if hasattr(self.processor, 'tokenizer'):
            labels = self.processor.tokenizer.encode(dummy_text, return_tensors="pt").squeeze()
        else:
            labels = torch.tensor([0])  # Fallback

        return {
            'input_values': audio_inputs.input_values.squeeze(),
            'attention_mask': audio_inputs.attention_mask.squeeze() if hasattr(audio_inputs, 'attention_mask') else torch.ones_like(audio_inputs.input_values.squeeze()),
            'labels': labels,
            'twi_text': dummy_text,
            'speaker_id': "dummy",
            'dialect': "dummy",
            'sample_id': "dummy"
        }


class TwiSpeechTranslationDataset(Dataset):
    """
    Dataset for Twi Speech Translation (Audio → English Text)

    Use case: Direct speech translation where we want Twi audio → English text
    Target: meaning (English)
    """

    def __init__(self,
                 samples: List[AudioSample],
                 processor: Union[Wav2Vec2Processor, WhisperProcessor],
                 audio_processor,  # TwiAudioProcessor - avoiding circular import
                 text_processor,   # TwiTextProcessor - avoiding circular import
                 is_training: bool = True,
                 max_audio_length: int = 320000,
                 max_text_length: int = 512):

        self.samples = samples
        self.processor = processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.is_training = is_training
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length

        # Filter samples that have English meaning
        self.valid_samples = [s for s in samples if s.meaning]

        logger.info(f"TwiSpeechTranslationDataset: {len(self.valid_samples)}/{len(samples)} valid samples")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.valid_samples[idx]

        try:
            # Load and preprocess audio
            audio, sr = self.audio_processor.load_audio(sample.audio_path)
            audio = self.audio_processor.preprocess_audio(
                audio, sr, apply_augmentation=self.is_training
            )

            # Use English meaning as target
            english_text = sample.meaning
            # Note: We might want different normalization for English
            normalized_text = english_text.strip()

            # Process audio
            audio_inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_audio_length
            )

            # Process English text labels
            if hasattr(self.processor, 'tokenizer'):
                text_inputs = self.processor.tokenizer(
                    normalized_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length
                )
                labels = text_inputs.input_ids.squeeze()
            else:
                labels = self.processor.tokenizer.encode(normalized_text, return_tensors="pt").squeeze()

            return {
                'input_values': audio_inputs.input_values.squeeze(),
                'attention_mask': audio_inputs.attention_mask.squeeze() if hasattr(audio_inputs, 'attention_mask') else None,
                'labels': labels,
                'english_text': normalized_text,
                'twi_text': sample.prompt_text or sample.transcription,  # Keep for reference
                'speaker_id': sample.speaker_id,
                'dialect': sample.dialect,
                'sample_id': sample.prompt_id
            }

        except Exception as e:
            logger.error(f"Error processing sample {idx} for speech translation: {e}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample for error cases"""
        dummy_audio = np.zeros(16000)
        dummy_text = "dummy translation"

        audio_inputs = self.processor(
            dummy_audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        if hasattr(self.processor, 'tokenizer'):
            labels = self.processor.tokenizer.encode(dummy_text, return_tensors="pt").squeeze()
        else:
            labels = torch.tensor([0])

        return {
            'input_values': audio_inputs.input_values.squeeze(),
            'attention_mask': audio_inputs.attention_mask.squeeze() if hasattr(audio_inputs, 'attention_mask') else torch.ones_like(audio_inputs.input_values.squeeze()),
            'labels': labels,
            'english_text': dummy_text,
            'twi_text': "dummy twi",
            'speaker_id': "dummy",
            'dialect': "dummy",
            'sample_id': "dummy"
        }


class TwiMultilingualDataset(Dataset):
    """
    Dataset for Multilingual Training (Audio → Both Twi and English)

    Use case: Joint training where model learns both transcription and translation
    Targets: Both prompt_text (Twi) and meaning (English)
    """

    def __init__(self,
                 samples: List[AudioSample],
                 processor: Union[Wav2Vec2Processor, WhisperProcessor],
                 audio_processor,  # TwiAudioProcessor - avoiding circular import
                 text_processor,   # TwiTextProcessor - avoiding circular import
                 is_training: bool = True,
                 max_audio_length: int = 320000,
                 max_text_length: int = 512,
                 task_mixing_ratio: float = 0.5):  # 0.5 = equal Twi/English, 1.0 = all English

        self.samples = samples
        self.processor = processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.is_training = is_training
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.task_mixing_ratio = task_mixing_ratio

        # Filter samples that have both Twi and English
        self.valid_samples = [s for s in samples if (s.prompt_text or s.transcription) and s.meaning]

        logger.info(f"TwiMultilingualDataset: {len(self.valid_samples)}/{len(samples)} valid samples")
        logger.info(f"Task mixing ratio: {task_mixing_ratio} (0.0=all Twi, 1.0=all English)")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.valid_samples[idx]

        try:
            # Load and preprocess audio
            audio, sr = self.audio_processor.load_audio(sample.audio_path)
            audio = self.audio_processor.preprocess_audio(
                audio, sr, apply_augmentation=self.is_training
            )

            # Decide task based on mixing ratio
            use_english = np.random.random() < self.task_mixing_ratio if self.is_training else False

            if use_english:
                target_text = sample.meaning
                task_type = "translation"
                # Add task prefix for multilingual models
                target_text = f"<translate_to_english> {target_text}"
            else:
                target_text = sample.prompt_text or sample.transcription
                task_type = "transcription"
                target_text = f"<transcribe_twi> {target_text}"

            normalized_text = self.text_processor.normalize_text(target_text) if not use_english else target_text.strip()

            # Process audio
            audio_inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_audio_length
            )

            # Process text labels
            if hasattr(self.processor, 'tokenizer'):
                text_inputs = self.processor.tokenizer(
                    normalized_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length
                )
                labels = text_inputs.input_ids.squeeze()
            else:
                labels = self.processor.tokenizer.encode(normalized_text, return_tensors="pt").squeeze()

            return {
                'input_values': audio_inputs.input_values.squeeze(),
                'attention_mask': audio_inputs.attention_mask.squeeze() if hasattr(audio_inputs, 'attention_mask') else None,
                'labels': labels,
                'target_text': normalized_text,
                'task_type': task_type,
                'twi_text': sample.prompt_text or sample.transcription,
                'english_text': sample.meaning,
                'speaker_id': sample.speaker_id,
                'dialect': sample.dialect,
                'sample_id': sample.prompt_id
            }

        except Exception as e:
            logger.error(f"Error processing sample {idx} for multilingual: {e}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample for error cases"""
        dummy_audio = np.zeros(16000)
        dummy_text = "<transcribe_twi> dummy"

        audio_inputs = self.processor(
            dummy_audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        if hasattr(self.processor, 'tokenizer'):
            labels = self.processor.tokenizer.encode(dummy_text, return_tensors="pt").squeeze()
        else:
            labels = torch.tensor([0])

        return {
            'input_values': audio_inputs.input_values.squeeze(),
            'attention_mask': audio_inputs.attention_mask.squeeze() if hasattr(audio_inputs, 'attention_mask') else torch.ones_like(audio_inputs.input_values.squeeze()),
            'labels': labels,
            'target_text': dummy_text,
            'task_type': "transcription",
            'twi_text': "dummy twi",
            'english_text': "dummy english",
            'speaker_id': "dummy",
            'dialect': "dummy",
            'sample_id': "dummy"
        }


class TwiCrossLingualDataset(Dataset):
    """
    Dataset for Cross-lingual Training with Auxiliary Input

    Use case: Audio + Twi text → English text (using both audio and text inputs)
    This can help with better translation by providing both modalities
    """

    def __init__(self,
                 samples: List[AudioSample],
                 processor: Union[Wav2Vec2Processor, WhisperProcessor],
                 audio_processor,  # TwiAudioProcessor - avoiding circular import
                 text_processor,   # TwiTextProcessor - avoiding circular import
                 is_training: bool = True,
                 max_audio_length: int = 320000,
                 max_text_length: int = 512):

        self.samples = samples
        self.processor = processor
        self.audio_processor = audio_processor
        self.text_processor = text_processor
        self.is_training = is_training
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length

        # Filter samples that have all three: audio, Twi text, and English text
        self.valid_samples = [s for s in samples if (s.prompt_text or s.transcription) and s.meaning]

        logger.info(f"TwiCrossLingualDataset: {len(self.valid_samples)}/{len(samples)} valid samples")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.valid_samples[idx]

        try:
            # Load and preprocess audio
            audio, sr = self.audio_processor.load_audio(sample.audio_path)
            audio = self.audio_processor.preprocess_audio(
                audio, sr, apply_augmentation=self.is_training
            )

            # Prepare texts
            twi_text = sample.prompt_text or sample.transcription
            english_text = sample.meaning

            # Process audio
            audio_inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_audio_length
            )

            # Process Twi text (auxiliary input)
            normalized_twi = self.text_processor.normalize_text(twi_text)
            if hasattr(self.processor, 'tokenizer'):
                twi_inputs = self.processor.tokenizer(
                    normalized_twi,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length
                )
                twi_input_ids = twi_inputs.input_ids.squeeze()
            else:
                twi_input_ids = self.processor.tokenizer.encode(normalized_twi, return_tensors="pt").squeeze()

            # Process English text (target)
            normalized_english = english_text.strip()
            if hasattr(self.processor, 'tokenizer'):
                english_inputs = self.processor.tokenizer(
                    normalized_english,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length
                )
                labels = english_inputs.input_ids.squeeze()
            else:
                labels = self.processor.tokenizer.encode(normalized_english, return_tensors="pt").squeeze()

            return {
                'input_values': audio_inputs.input_values.squeeze(),
                'attention_mask': audio_inputs.attention_mask.squeeze() if hasattr(audio_inputs, 'attention_mask') else None,
                'twi_input_ids': twi_input_ids,  # Auxiliary Twi text input
                'labels': labels,  # English target
                'twi_text': normalized_twi,
                'english_text': normalized_english,
                'speaker_id': sample.speaker_id,
                'dialect': sample.dialect,
                'sample_id': sample.prompt_id
            }

        except Exception as e:
            logger.error(f"Error processing sample {idx} for cross-lingual: {e}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample for error cases"""
        dummy_audio = np.zeros(16000)
        dummy_twi = "dummy twi"
        dummy_english = "dummy english"

        audio_inputs = self.processor(
            dummy_audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        if hasattr(self.processor, 'tokenizer'):
            twi_ids = self.processor.tokenizer.encode(dummy_twi, return_tensors="pt").squeeze()
            labels = self.processor.tokenizer.encode(dummy_english, return_tensors="pt").squeeze()
        else:
            twi_ids = torch.tensor([0])
            labels = torch.tensor([0])

        return {
            'input_values': audio_inputs.input_values.squeeze(),
            'attention_mask': audio_inputs.attention_mask.squeeze() if hasattr(audio_inputs, 'attention_mask') else torch.ones_like(audio_inputs.input_values.squeeze()),
            'twi_input_ids': twi_ids,
            'labels': labels,
            'twi_text': dummy_twi,
            'english_text': dummy_english,
            'speaker_id': "dummy",
            'dialect': "dummy",
            'sample_id': "dummy"
        }


def create_dataset(samples: List[AudioSample],
                  dataset_type: str,
                  processor: Union[Wav2Vec2Processor, WhisperProcessor],
                  audio_processor,  # TwiAudioProcessor - avoiding circular import
                  text_processor,   # TwiTextProcessor - avoiding circular import
                  is_training: bool = True,
                  **kwargs) -> Dataset:
    """
    Factory function to create the appropriate dataset based on training type

    Args:
        samples: List of AudioSample objects
        dataset_type: One of ['speech_to_text', 'translation', 'multilingual', 'cross_lingual']
        processor: Wav2Vec2 or Whisper processor
        audio_processor: Audio preprocessing
        text_processor: Text preprocessing
        is_training: Whether this is for training (affects augmentation)
        **kwargs: Additional arguments for specific dataset types

    Returns:
        Appropriate dataset instance
    """

    dataset_classes = {
        'speech_to_text': TwiSpeechToTextDataset,
        'translation': TwiSpeechTranslationDataset,
        'multilingual': TwiMultilingualDataset,
        'cross_lingual': TwiCrossLingualDataset
    }

    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from {list(dataset_classes.keys())}")

    dataset_class = dataset_classes[dataset_type]

    return dataset_class(
        samples=samples,
        processor=processor,
        audio_processor=audio_processor,
        text_processor=text_processor,
        is_training=is_training,
        **kwargs
    )


# Usage example and documentation
"""
USAGE EXAMPLES:

1. Standard Twi Speech Recognition (Audio → Twi Text):
   dataset = create_dataset(
       samples=samples,
       dataset_type='speech_to_text',
       processor=wav2vec2_processor,
       audio_processor=audio_proc,
       text_processor=text_proc
   )

2. Twi to English Translation (Audio → English Text):
   dataset = create_dataset(
       samples=samples,
       dataset_type='translation',
       processor=whisper_processor,
       audio_processor=audio_proc,
       text_processor=text_proc
   )

3. Multilingual Training (Audio → Either Twi or English):
   dataset = create_dataset(
       samples=samples,
       dataset_type='multilingual',
       processor=multilingual_processor,
       audio_processor=audio_proc,
       text_processor=text_proc,
       task_mixing_ratio=0.7  # 70% English, 30% Twi
   )

4. Cross-lingual with Text Input (Audio + Twi Text → English):
   dataset = create_dataset(
       samples=samples,
       dataset_type='cross_lingual',
       processor=cross_processor,
       audio_processor=audio_proc,
       text_processor=text_proc
   )

TRAINING SCENARIOS:

Scenario 1: Pure ASR (Speech Recognition)
- Input: Twi audio
- Output: Twi text (prompt_text)
- Use case: Transcribing Twi speech to text

Scenario 2: Speech Translation
- Input: Twi audio
- Output: English text (meaning)
- Use case: Direct translation from Twi speech to English

Scenario 3: Multilingual Model
- Input: Twi audio
- Output: Either Twi or English text (task-dependent)
- Use case: One model that can do both transcription and translation

Scenario 4: Multimodal Cross-lingual
- Input: Twi audio + Twi text
- Output: English text
- Use case: Better translation using both audio and text context
"""
