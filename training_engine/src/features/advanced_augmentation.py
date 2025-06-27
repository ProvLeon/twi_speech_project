import numpy as np
import torch
import random
from typing import List, Callable, Optional
import scipy.signal
from scipy.interpolate import interp1d


class AdvancedAugmentation:
    """
    Advanced augmentation techniques for speech features that go beyond
    the basic approach used in local_dialect_speech_model
    """

    def __init__(self, augmentation_prob: float = 0.6):
        """
        Initialize advanced augmentation

        Args:
            augmentation_prob: Probability of applying augmentation
        """
        self.augmentation_prob = augmentation_prob

        # Define all augmentation techniques
        self.techniques = [
            self.spec_augment,
            self.time_masking,
            self.freq_masking,
            self.gaussian_noise,
            self.time_shifting,
            self.pitch_shifting,
            self.speed_perturbation,
            self.dynamic_range_compression,
            self.spectral_subtraction,
            self.formant_shifting,
            self.vocal_tract_length_perturbation,
            self.adaptive_noise_addition,
            self.frequency_warping,
            self.time_stretching,
            self.mixup_features
        ]

    def spec_augment(self, features: np.ndarray,
                    freq_mask_param: int = 15,
                    time_mask_param: int = 25,
                    num_freq_masks: int = 2,
                    num_time_masks: int = 2) -> np.ndarray:
        """
        SpecAugment implementation with configurable parameters

        Args:
            features: Input feature matrix (freq_bins, time_frames)
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply

        Returns:
            Augmented feature matrix
        """
        features = features.copy()
        freq_len, time_len = features.shape

        # Apply frequency masking
        for _ in range(num_freq_masks):
            if freq_len > freq_mask_param:
                f_mask_size = np.random.randint(1, min(freq_mask_param, freq_len // 3))
                f_start = np.random.randint(0, freq_len - f_mask_size)
                features[f_start:f_start + f_mask_size, :] = 0

        # Apply time masking
        for _ in range(num_time_masks):
            if time_len > time_mask_param:
                t_mask_size = np.random.randint(1, min(time_mask_param, time_len // 3))
                t_start = np.random.randint(0, time_len - t_mask_size)
                features[:, t_start:t_start + t_mask_size] = 0

        return features

    def time_masking(self, features: np.ndarray, max_mask_length: int = 20) -> np.ndarray:
        """
        Advanced time domain masking with variable mask patterns

        Args:
            features: Input feature matrix
            max_mask_length: Maximum length of time mask

        Returns:
            Time-masked features
        """
        features = features.copy()
        time_len = features.shape[1]

        if time_len > max_mask_length:
            # Random number of masks (1-3)
            num_masks = np.random.randint(1, 4)

            for _ in range(num_masks):
                mask_length = np.random.randint(1, min(max_mask_length, time_len // 4))
                start_pos = np.random.randint(0, time_len - mask_length)

                # Apply different mask patterns
                mask_type = np.random.choice(['zero', 'noise', 'interpolate'])

                if mask_type == 'zero':
                    features[:, start_pos:start_pos + mask_length] = 0
                elif mask_type == 'noise':
                    noise_level = np.std(features) * 0.1
                    features[:, start_pos:start_pos + mask_length] = np.random.randn(
                        features.shape[0], mask_length) * noise_level
                elif mask_type == 'interpolate':
                    # Linear interpolation between boundaries
                    if start_pos > 0 and start_pos + mask_length < time_len:
                        start_vals = features[:, start_pos - 1:start_pos]
                        end_vals = features[:, start_pos + mask_length:start_pos + mask_length + 1]
                        for i in range(features.shape[0]):
                            features[i, start_pos:start_pos + mask_length] = np.linspace(
                                start_vals[i, 0], end_vals[i, 0], mask_length)

        return features

    def freq_masking(self, features: np.ndarray, max_mask_length: int = 10) -> np.ndarray:
        """
        Advanced frequency domain masking

        Args:
            features: Input feature matrix
            max_mask_length: Maximum length of frequency mask

        Returns:
            Frequency-masked features
        """
        features = features.copy()
        freq_len = features.shape[0]

        if freq_len > max_mask_length:
            num_masks = np.random.randint(1, 3)

            for _ in range(num_masks):
                mask_length = np.random.randint(1, min(max_mask_length, freq_len // 4))
                start_pos = np.random.randint(0, freq_len - mask_length)

                # Apply frequency-specific masking
                features[start_pos:start_pos + mask_length, :] = 0

        return features

    def gaussian_noise(self, features: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """
        Add adaptive Gaussian noise based on signal characteristics

        Args:
            features: Input feature matrix
            noise_level: Base noise level

        Returns:
            Noise-augmented features
        """
        # Adaptive noise level based on signal energy
        signal_energy = np.mean(features ** 2)
        adaptive_noise_level = noise_level * (1 + signal_energy)

        noise = np.random.randn(*features.shape) * adaptive_noise_level
        return features + noise

    def time_shifting(self, features: np.ndarray, max_shift: int = 8) -> np.ndarray:
        """
        Advanced time shifting with boundary handling

        Args:
            features: Input feature matrix
            max_shift: Maximum shift in time frames

        Returns:
            Time-shifted features
        """
        features = features.copy()
        time_len = features.shape[1]
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift > 0:
            # Shift right
            features[:, shift:] = features[:, :-shift]
            # Fill beginning with edge values
            features[:, :shift] = features[:, shift:shift+1]
        elif shift < 0:
            # Shift left
            features[:, :shift] = features[:, -shift:]
            # Fill end with edge values
            features[:, shift:] = features[:, shift-1:shift]

        return features

    def pitch_shifting(self, features: np.ndarray, max_shift: int = 3) -> np.ndarray:
        """
        Simulate pitch shifting by frequency bin manipulation

        Args:
            features: Input feature matrix
            max_shift: Maximum frequency bin shift

        Returns:
            Pitch-shifted features
        """
        features = features.copy()
        freq_len = features.shape[0]
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift > 0:
            # Shift up in frequency
            features[shift:, :] = features[:-shift, :]
            features[:shift, :] = 0
        elif shift < 0:
            # Shift down in frequency
            features[:shift, :] = features[-shift:, :]
            features[shift:, :] = 0

        return features

    def speed_perturbation(self, features: np.ndarray,
                          speed_range: tuple = (0.85, 1.15)) -> np.ndarray:
        """
        Advanced speed perturbation using interpolation

        Args:
            features: Input feature matrix
            speed_range: Range of speed factors

        Returns:
            Speed-perturbed features
        """
        speed_factor = np.random.uniform(*speed_range)
        time_len = features.shape[1]
        new_time_len = int(time_len / speed_factor)

        # Ensure new length is valid
        if new_time_len < 1 or new_time_len > time_len * 2:
            return features

        # Interpolate each frequency bin
        new_features = np.zeros((features.shape[0], time_len))
        old_indices = np.linspace(0, time_len - 1, time_len)
        new_indices = np.linspace(0, time_len - 1, new_time_len)

        for i in range(features.shape[0]):
            if new_time_len != time_len:
                f = interp1d(old_indices, features[i, :], kind='linear',
                           bounds_error=False, fill_value=0)
                interpolated = f(new_indices)

                # Pad or truncate to original length
                if new_time_len > time_len:
                    new_features[i, :] = interpolated[:time_len]
                else:
                    new_features[i, :new_time_len] = interpolated
            else:
                new_features[i, :] = features[i, :]

        return new_features

    def dynamic_range_compression(self, features: np.ndarray,
                                 threshold: float = 0.1,
                                 ratio: float = 3.0) -> np.ndarray:
        """
        Apply dynamic range compression to features

        Args:
            features: Input feature matrix
            threshold: Compression threshold
            ratio: Compression ratio

        Returns:
            Compressed features
        """
        features = features.copy()

        # Apply compression per frequency bin
        for i in range(features.shape[0]):
            signal = features[i, :]
            abs_signal = np.abs(signal)

            # Find samples above threshold
            mask = abs_signal > threshold

            # Apply compression
            compressed_signal = signal.copy()
            compressed_signal[mask] = np.sign(signal[mask]) * (
                threshold + (abs_signal[mask] - threshold) / ratio
            )

            features[i, :] = compressed_signal

        return features

    def spectral_subtraction(self, features: np.ndarray,
                           alpha: float = 2.0,
                           beta: float = 0.01) -> np.ndarray:
        """
        Simulate spectral subtraction for noise robustness

        Args:
            features: Input feature matrix
            alpha: Over-subtraction factor
            beta: Spectral floor factor

        Returns:
            Processed features
        """
        features = features.copy()

        # Estimate noise spectrum from first few frames
        noise_frames = min(5, features.shape[1] // 4)
        if noise_frames > 0:
            noise_spectrum = np.mean(np.abs(features[:, :noise_frames]), axis=1, keepdims=True)

            # Apply spectral subtraction
            magnitude = np.abs(features)
            phase = np.sign(features)

            # Subtract noise
            cleaned_magnitude = magnitude - alpha * noise_spectrum

            # Apply spectral floor
            cleaned_magnitude = np.maximum(cleaned_magnitude, beta * magnitude)

            features = cleaned_magnitude * phase

        return features

    def formant_shifting(self, features: np.ndarray, shift_factor: float = 0.1) -> np.ndarray:
        """
        Simulate formant shifting by frequency warping

        Args:
            features: Input feature matrix
            shift_factor: Amount of formant shift

        Returns:
            Formant-shifted features
        """
        features = features.copy()
        freq_len = features.shape[0]

        # Create warping function
        shift = np.random.uniform(-shift_factor, shift_factor)
        warp_factor = 1 + shift

        # Warp frequency axis
        old_freq_indices = np.arange(freq_len)
        new_freq_indices = old_freq_indices * warp_factor

        # Clip to valid range
        new_freq_indices = np.clip(new_freq_indices, 0, freq_len - 1)

        # Interpolate warped features
        warped_features = np.zeros_like(features)
        for t in range(features.shape[1]):
            f = interp1d(old_freq_indices, features[:, t], kind='linear',
                        bounds_error=False, fill_value=0)
            warped_features[:, t] = f(new_freq_indices)

        return warped_features

    def vocal_tract_length_perturbation(self, features: np.ndarray,
                                       vtl_factor: float = 0.1) -> np.ndarray:
        """
        Simulate vocal tract length perturbation

        Args:
            features: Input feature matrix
            vtl_factor: VTL perturbation factor

        Returns:
            VTL-perturbed features
        """
        # This is similar to formant shifting but with different scaling
        vtl_shift = np.random.uniform(-vtl_factor, vtl_factor)
        return self.formant_shifting(features, vtl_shift)

    def adaptive_noise_addition(self, features: np.ndarray) -> np.ndarray:
        """
        Add noise adaptively based on local signal characteristics

        Args:
            features: Input feature matrix

        Returns:
            Noise-augmented features
        """
        features = features.copy()

        # Calculate local signal energy
        window_size = min(10, features.shape[1] // 4)
        if window_size > 0:
            for i in range(0, features.shape[1], window_size):
                end_idx = min(i + window_size, features.shape[1])
                window = features[:, i:end_idx]

                # Adaptive noise level based on local energy
                local_energy = np.mean(window ** 2)
                noise_level = 0.01 * (1 + local_energy)

                noise = np.random.randn(*window.shape) * noise_level
                features[:, i:end_idx] += noise

        return features

    def frequency_warping(self, features: np.ndarray, warp_factor: float = 0.2) -> np.ndarray:
        """
        Apply frequency warping for robustness

        Args:
            features: Input feature matrix
            warp_factor: Maximum warping factor

        Returns:
            Frequency-warped features
        """
        features = features.copy()
        freq_len = features.shape[0]

        # Generate random warping function
        warp = np.random.uniform(-warp_factor, warp_factor)

        # Create non-linear frequency mapping
        old_indices = np.arange(freq_len)
        new_indices = old_indices + warp * np.sin(2 * np.pi * old_indices / freq_len)
        new_indices = np.clip(new_indices, 0, freq_len - 1)

        # Apply warping
        warped_features = np.zeros_like(features)
        for t in range(features.shape[1]):
            f = interp1d(old_indices, features[:, t], kind='linear',
                        bounds_error=False, fill_value=0)
            warped_features[:, t] = f(new_indices)

        return warped_features

    def time_stretching(self, features: np.ndarray, stretch_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """
        Time stretching while preserving spectral characteristics

        Args:
            features: Input feature matrix
            stretch_range: Range of stretch factors

        Returns:
            Time-stretched features
        """
        stretch_factor = np.random.uniform(*stretch_range)
        time_len = features.shape[1]
        new_time_len = int(time_len * stretch_factor)

        if new_time_len <= 0 or new_time_len > time_len * 2:
            return features

        # Stretch using phase vocoder-like approach
        stretched_features = np.zeros((features.shape[0], time_len))

        if new_time_len != time_len:
            old_indices = np.linspace(0, time_len - 1, new_time_len)
            new_indices = np.arange(time_len)

            for i in range(features.shape[0]):
                f = interp1d(old_indices, features[i, :new_time_len] if new_time_len <= time_len else
                           np.pad(features[i, :], (0, new_time_len - time_len), mode='edge'),
                           kind='cubic', bounds_error=False, fill_value=0)
                stretched_features[i, :] = f(new_indices)
        else:
            stretched_features = features

        return stretched_features

    def mixup_features(self, features: np.ndarray, other_features: Optional[np.ndarray] = None,
                      alpha: float = 0.2) -> np.ndarray:
        """
        Apply mixup augmentation to features

        Args:
            features: Primary feature matrix
            other_features: Optional second feature matrix for mixing
            alpha: Mixup parameter

        Returns:
            Mixed features
        """
        if other_features is None:
            # Self-mixup with shifted version
            shift = np.random.randint(1, features.shape[1])
            other_features = np.roll(features, shift, axis=1)

        # Generate mixup coefficient
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5

        # Mix features
        mixed_features = lam * features + (1 - lam) * other_features

        return mixed_features

    def augment(self, features: np.ndarray, num_augmentations: Optional[int] = None) -> np.ndarray:
        """
        Apply random combination of augmentation techniques

        Args:
            features: Input feature matrix
            num_augmentations: Number of augmentations to apply (random if None)

        Returns:
            Augmented features
        """
        if np.random.random() > self.augmentation_prob:
            return features

        # Choose number of augmentations
        if num_augmentations is None:
            num_augmentations = np.random.randint(1, 4)

        # Select random techniques
        selected_techniques = np.random.choice(
            self.techniques,
            size=min(num_augmentations, len(self.techniques)),
            replace=False
        )

        # Apply selected augmentations
        augmented_features = features.copy()

        for technique in selected_techniques:
            try:
                # Special handling for mixup
                if technique.__name__ == 'mixup_features':
                    # Use mixup with lower probability
                    if np.random.random() < 0.3:
                        augmented_features = technique(augmented_features)
                else:
                    augmented_features = technique(augmented_features)
            except Exception as e:
                # Skip failed augmentations
                continue

        return augmented_features

    def create_augmentation_pipeline(self, techniques: List[str]) -> Callable:
        """
        Create a custom augmentation pipeline

        Args:
            techniques: List of technique names to include

        Returns:
            Augmentation function
        """
        technique_map = {func.__name__: func for func in self.techniques}
        selected_functions = [technique_map[name] for name in techniques if name in technique_map]

        def pipeline(features: np.ndarray) -> np.ndarray:
            augmented = features.copy()
            for func in selected_functions:
                try:
                    augmented = func(augmented)
                except:
                    continue
            return augmented

        return pipeline
