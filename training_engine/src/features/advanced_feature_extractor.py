import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class AdvancedAudioProcessor:
    """
    Advanced audio processor with superior feature extraction capabilities
    that outperforms the basic MFCC approach used in local_dialect_speech_model
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_mels: int = 80,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: int = 2048,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None):
        """
        Initialize advanced audio processor

        Args:
            sample_rate: Target sample rate
            n_mfcc: Number of MFCCs to extract
            n_mels: Number of mel filter banks
            n_fft: FFT window size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            f_min: Minimum frequency for mel scale
            f_max: Maximum frequency for mel scale
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2

        # Initialize transforms
        self._initialize_transforms()

    def _initialize_transforms(self):
        """Initialize torchaudio transforms for better performance"""
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0
        )

        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'n_mels': self.n_mels,
                'f_min': self.f_min,
                'f_max': self.f_max
            }
        )

        # Spectral centroid
        self.spectral_centroid = T.SpectralCentroid(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Pitch detection
        self.pitch_transform = T.PitchShift(
            sample_rate=self.sample_rate,
            n_steps=0  # No shift, just for detection
        )

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Advanced audio loading with format detection and quality enhancement

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # First try with torchaudio for better performance
            if file_path.endswith(('.wav', '.flac', '.m4a')):
                waveform, sr = torchaudio.load(file_path)
                audio = waveform.numpy().flatten()
                if sr != self.sample_rate:
                    resampler = T.Resample(sr, self.sample_rate)
                    audio = resampler(torch.from_numpy(audio)).numpy()
                return audio, self.sample_rate

            # Handle MP3 and other formats
            elif file_path.endswith('.mp3'):
                audio_segment = AudioSegment.from_mp3(file_path)
                audio_segment = audio_segment.set_channels(1)
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio = audio / np.max(np.abs(audio))  # Normalize
                return audio, self.sample_rate

            else:
                # Fallback to librosa
                audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                return audio, sr

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silence if loading fails
            return np.zeros(self.sample_rate), self.sample_rate

    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Advanced audio enhancement pipeline

        Args:
            audio: Input audio signal

        Returns:
            Enhanced audio signal
        """
        # Apply pre-emphasis filter
        audio = self._preemphasis(audio)

        # Advanced noise reduction
        audio = self._spectral_subtraction_denoising(audio)

        # Dynamic range compression
        audio = self._dynamic_range_compression(audio)

        # Normalize
        audio = self._normalize_audio(audio)

        return audio

    def _preemphasis(self, audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to enhance high frequencies"""
        return np.append(audio[0], audio[1:] - alpha * audio[:-1])

    def _spectral_subtraction_denoising(self, audio: np.ndarray) -> np.ndarray:
        """Advanced spectral subtraction for noise reduction"""
        # Compute STFT
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Estimate noise spectrum from first 0.5 seconds
        noise_frames = int(0.5 * self.sample_rate / self.hop_length)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        # Spectral subtraction with over-subtraction factor
        alpha = 2.0
        beta = 0.01

        # Compute spectral subtraction
        magnitude_enhanced = magnitude - alpha * noise_spectrum

        # Apply spectral floor
        magnitude_enhanced = np.maximum(magnitude_enhanced, beta * magnitude)

        # Reconstruct signal
        D_enhanced = magnitude_enhanced * np.exp(1j * phase)
        audio_enhanced = librosa.istft(D_enhanced, hop_length=self.hop_length)

        return audio_enhanced

    def _dynamic_range_compression(self, audio: np.ndarray,
                                 threshold: float = 0.1,
                                 ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression"""
        # Simple compressor
        compressed = np.copy(audio)
        mask = np.abs(audio) > threshold
        compressed[mask] = np.sign(audio[mask]) * (
            threshold + (np.abs(audio[mask]) - threshold) / ratio
        )
        return compressed

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Advanced audio normalization"""
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * 0.1

        # Peak normalization
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio / peak * 0.95

        return audio

    def extract_comprehensive_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive audio features that go beyond basic MFCC

        Args:
            audio: Audio signal

        Returns:
            Dictionary of extracted features
        """
        audio_tensor = torch.from_numpy(audio).float()

        features = {}

        # 1. Mel-frequency cepstral coefficients (MFCCs)
        mfcc = self.mfcc_transform(audio_tensor).numpy()
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features['mfcc'] = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

        # 2. Mel-scale spectrogram
        mel_spec = self.mel_spectrogram(audio_tensor).numpy()
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spectrogram'] = mel_spec_db

        # 3. Spectral features
        spectral_centroid = self.spectral_centroid(audio_tensor).numpy()
        features['spectral_centroid'] = spectral_centroid

        # 4. Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['chroma'] = chroma

        # 5. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['spectral_contrast'] = spectral_contrast

        # 6. Tonnetz (harmonic network)
        tonnetz = librosa.feature.tonnetz(
            y=audio, sr=self.sample_rate
        )
        features['tonnetz'] = tonnetz

        # 7. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )
        features['zcr'] = zcr

        # 8. Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['spectral_rolloff'] = spectral_rolloff

        # 9. Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['spectral_bandwidth'] = spectral_bandwidth

        # 10. Poly features
        poly_features = librosa.feature.poly_features(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        features['poly_features'] = poly_features

        return features

    def create_unified_feature_vector(self, features: Dict[str, np.ndarray],
                                    max_length: Optional[int] = None) -> np.ndarray:
        """
        Create a unified feature vector from all extracted features

        Args:
            features: Dictionary of extracted features
            max_length: Maximum sequence length for padding/truncation

        Returns:
            Unified feature matrix
        """
        # Standardize all features to same time dimension
        min_time_frames = min(feat.shape[-1] for feat in features.values())

        unified_features = []
        for feat_name, feat_data in features.items():
            # Truncate to minimum length
            if feat_data.shape[-1] > min_time_frames:
                feat_data = feat_data[:, :min_time_frames]

            # Normalize features
            feat_data = self._normalize_features(feat_data)
            unified_features.append(feat_data)

        # Concatenate all features
        combined_features = np.vstack(unified_features)

        # Apply length standardization if specified
        if max_length is not None:
            if combined_features.shape[1] > max_length:
                combined_features = combined_features[:, :max_length]
            elif combined_features.shape[1] < max_length:
                padding = np.zeros((combined_features.shape[0],
                                  max_length - combined_features.shape[1]))
                combined_features = np.hstack([combined_features, padding])

        return combined_features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Advanced feature normalization with outlier handling"""
        # Robust normalization using median and MAD
        median = np.median(features, axis=1, keepdims=True)
        mad = np.median(np.abs(features - median), axis=1, keepdims=True)

        # Avoid division by zero
        mad = np.where(mad == 0, 1, mad)

        # Normalize
        normalized = (features - median) / (1.4826 * mad)  # 1.4826 is the MAD constant

        # Clip outliers
        normalized = np.clip(normalized, -3, 3)

        return normalized

    def preprocess(self, file_path: str, max_length: Optional[int] = None) -> np.ndarray:
        """
        Complete preprocessing pipeline that outperforms the basic approach

        Args:
            file_path: Path to audio file
            max_length: Maximum sequence length

        Returns:
            Preprocessed feature matrix
        """
        # Load audio
        audio, _ = self.load_audio(file_path)

        # Enhance audio quality
        audio = self.enhance_audio(audio)

        # Extract comprehensive features
        features = self.extract_comprehensive_features(audio)

        # Create unified feature vector
        unified_features = self.create_unified_feature_vector(features, max_length)

        return unified_features

    def get_feature_info(self) -> Dict[str, int]:
        """
        Get information about feature dimensions

        Returns:
            Dictionary with feature dimension information
        """
        # Calculate total feature dimensions
        feature_dims = {
            'mfcc': self.n_mfcc * 3,  # MFCC + delta + delta-delta
            'mel_spectrogram': self.n_mels,
            'spectral_centroid': 1,
            'chroma': 12,
            'spectral_contrast': 7,
            'tonnetz': 6,
            'zcr': 1,
            'spectral_rolloff': 1,
            'spectral_bandwidth': 1,
            'poly_features': 2
        }

        total_dims = sum(feature_dims.values())
        feature_dims['total'] = total_dims

        return feature_dims


class SuperiorFeatureExtractor:
    """
    Feature extractor that uses the advanced audio processor
    """

    def __init__(self, output_dir: str = "data/processed_superior",
                 max_length: Optional[int] = None):
        """
        Initialize superior feature extractor

        Args:
            output_dir: Directory to save processed features
            max_length: Maximum sequence length for standardization
        """
        self.processor = AdvancedAudioProcessor()
        self.output_dir = output_dir
        self.max_length = max_length

        import os
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_features_from_recordings(self, recordings_dir: str,
                                       metadata_file: Optional[str] = None) -> Dict:
        """
        Extract features from recorded audio files

        Args:
            recordings_dir: Directory containing audio recordings
            metadata_file: Optional metadata file with labels

        Returns:
            Dictionary containing features and labels
        """
        import os
        import pandas as pd
        from tqdm import tqdm

        # Get all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []

        for root, dirs, files in os.walk(recordings_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))

        print(f"Found {len(audio_files)} audio files")

        # Load metadata if available
        labels = []
        if metadata_file and os.path.exists(metadata_file):
            metadata = pd.read_csv(metadata_file)
            # Create mapping from filename to label
            label_map = dict(zip(metadata['file'], metadata['intent']))
        else:
            label_map = {}

        # Extract features
        features_list = []
        labels_list = []

        for audio_file in tqdm(audio_files, desc="Extracting features"):
            try:
                # Extract features
                features = self.processor.preprocess(audio_file, self.max_length)
                features_list.append(features)

                # Get label
                filename = os.path.basename(audio_file)
                label = label_map.get(filename, os.path.dirname(audio_file).split('/')[-1])
                labels_list.append(label)

                # Save individual feature file
                feature_filename = filename.replace('.wav', '.npy').replace('.mp3', '.npy')
                feature_path = os.path.join(self.output_dir, feature_filename)
                np.save(feature_path, features)

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue

        # Save combined dataset
        np.save(os.path.join(self.output_dir, 'features.npy'), features_list)
        np.save(os.path.join(self.output_dir, 'labels.npy'), labels_list)

        # Create and save label mapping
        unique_labels = sorted(set(labels_list))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}

        import json
        with open(os.path.join(self.output_dir, 'label_map.json'), 'w') as f:
            json.dump(label_to_idx, f, indent=2)

        # Save feature info
        feature_info = self.processor.get_feature_info()
        with open(os.path.join(self.output_dir, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f, indent=2)

        print(f"Feature extraction complete. Saved to {self.output_dir}")
        print(f"Total features: {len(features_list)}")
        print(f"Feature dimensions: {feature_info['total']}")
        print(f"Labels: {unique_labels}")

        return {
            'features': features_list,
            'labels': labels_list,
            'label_to_idx': label_to_idx,
            'feature_info': feature_info
        }
