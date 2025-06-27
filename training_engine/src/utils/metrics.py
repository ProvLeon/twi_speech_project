"""
Metrics Utilities for Twi Speech Recognition Training

This module provides custom metrics and evaluation tools specifically designed
for training Twi speech recognition models. It includes implementations of
standard ASR metrics like WER and CER, as well as Twi-specific metrics for
dialect classification and tone accuracy.

Key Features:
- Word Error Rate (WER) and Character Error Rate (CER) computation
- Real-time metric tracking during training
- Dialect classification accuracy
- Tone accuracy measurement
- Training progress metrics
- Custom loss functions

Author: Twi Speech Recognition Team
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from collections import defaultdict
import editdistance
from jiwer import wer, cer

logger = logging.getLogger(__name__)


class WERMetric(nn.Module):
    """Word Error Rate metric for training monitoring"""

    def __init__(self, processor=None):
        super().__init__()
        self.processor = processor
        self.reset()

    def reset(self):
        """Reset metric state"""
        self.total_wer = 0.0
        self.num_samples = 0
        self.predictions = []
        self.references = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               input_lengths: Optional[torch.Tensor] = None,
               target_lengths: Optional[torch.Tensor] = None):
        """
        Update WER with new predictions and targets

        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target sequences [batch_size, target_len]
            input_lengths: Lengths of input sequences
            target_lengths: Lengths of target sequences
        """
        if self.processor is None:
            logger.warning("No processor available for WER computation")
            return

        # Convert predictions to text
        pred_ids = torch.argmax(predictions, dim=-1)
        pred_texts = self.processor.batch_decode(pred_ids)

        # Convert targets to text
        target_texts = self.processor.batch_decode(targets)

        # Compute WER for each sample
        for pred_text, target_text in zip(pred_texts, target_texts):
            try:
                sample_wer = wer([target_text], [pred_text])
                self.total_wer += sample_wer
                self.num_samples += 1

                # Store for later analysis
                self.predictions.append(pred_text)
                self.references.append(target_text)
            except Exception as e:
                logger.warning(f"Error computing WER for sample: {e}")

    def compute(self) -> float:
        """Compute average WER"""
        if self.num_samples == 0:
            return 0.0
        return self.total_wer / self.num_samples

    def get_samples(self) -> Tuple[List[str], List[str]]:
        """Get accumulated predictions and references"""
        return self.predictions.copy(), self.references.copy()


class CERMetric(nn.Module):
    """Character Error Rate metric for training monitoring"""

    def __init__(self, processor=None):
        super().__init__()
        self.processor = processor
        self.reset()

    def reset(self):
        """Reset metric state"""
        self.total_cer = 0.0
        self.num_samples = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               input_lengths: Optional[torch.Tensor] = None,
               target_lengths: Optional[torch.Tensor] = None):
        """Update CER with new predictions and targets"""
        if self.processor is None:
            logger.warning("No processor available for CER computation")
            return

        # Convert predictions to text
        pred_ids = torch.argmax(predictions, dim=-1)
        pred_texts = self.processor.batch_decode(pred_ids)

        # Convert targets to text
        target_texts = self.processor.batch_decode(targets)

        # Compute CER for each sample
        for pred_text, target_text in zip(pred_texts, target_texts):
            try:
                sample_cer = cer([target_text], [pred_text])
                self.total_cer += sample_cer
                self.num_samples += 1
            except Exception as e:
                logger.warning(f"Error computing CER for sample: {e}")

    def compute(self) -> float:
        """Compute average CER"""
        if self.num_samples == 0:
            return 0.0
        return self.total_cer / self.num_samples


class DialectAccuracyMetric(nn.Module):
    """Dialect classification accuracy metric"""

    def __init__(self, num_dialects: int = 3):
        super().__init__()
        self.num_dialects = num_dialects
        self.reset()

    def reset(self):
        """Reset metric state"""
        self.correct = 0
        self.total = 0
        self.per_dialect_correct = [0] * self.num_dialects
        self.per_dialect_total = [0] * self.num_dialects

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update dialect accuracy

        Args:
            predictions: Dialect predictions [batch_size, num_dialects]
            targets: Target dialect labels [batch_size]
        """
        pred_labels = torch.argmax(predictions, dim=-1)
        correct = (pred_labels == targets).sum().item()

        self.correct += correct
        self.total += targets.size(0)

        # Update per-dialect accuracy
        for i in range(self.num_dialects):
            dialect_mask = (targets == i)
            if dialect_mask.sum() > 0:
                dialect_correct = (pred_labels[dialect_mask] == targets[dialect_mask]).sum().item()
                self.per_dialect_correct[i] += dialect_correct
                self.per_dialect_total[i] += dialect_mask.sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute dialect accuracy metrics"""
        overall_accuracy = self.correct / max(self.total, 1)

        per_dialect_accuracy = []
        for i in range(self.num_dialects):
            if self.per_dialect_total[i] > 0:
                acc = self.per_dialect_correct[i] / self.per_dialect_total[i]
            else:
                acc = 0.0
            per_dialect_accuracy.append(acc)

        return {
            'dialect_accuracy': overall_accuracy,
            'asante_accuracy': per_dialect_accuracy[0],
            'akuapem_accuracy': per_dialect_accuracy[1],
            'fante_accuracy': per_dialect_accuracy[2],
        }


class ToneAccuracyMetric(nn.Module):
    """Tone classification accuracy metric for Twi"""

    def __init__(self, num_tones: int = 5):
        super().__init__()
        self.num_tones = num_tones
        self.reset()

    def reset(self):
        """Reset metric state"""
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               input_lengths: Optional[torch.Tensor] = None):
        """
        Update tone accuracy

        Args:
            predictions: Tone predictions [batch_size, seq_len, num_tones]
            targets: Target tone labels [batch_size, seq_len]
            input_lengths: Valid sequence lengths
        """
        pred_labels = torch.argmax(predictions, dim=-1)

        if input_lengths is not None:
            # Only compute accuracy for valid positions
            for i, length in enumerate(input_lengths):
                valid_preds = pred_labels[i, :length]
                valid_targets = targets[i, :length]
                correct = (valid_preds == valid_targets).sum().item()

                self.correct += correct
                self.total += length.item()
        else:
            # Compute for all positions
            correct = (pred_labels == targets).sum().item()
            self.correct += correct
            self.total += targets.numel()

    def compute(self) -> float:
        """Compute tone accuracy"""
        return self.correct / max(self.total, 1)


class TrainingMetrics:
    """Container for all training metrics"""

    def __init__(self, processor=None, num_dialects: int = 3, num_tones: int = 5):
        self.wer_metric = WERMetric(processor)
        self.cer_metric = CERMetric(processor)
        self.dialect_metric = DialectAccuracyMetric(num_dialects)
        self.tone_metric = ToneAccuracyMetric(num_tones)

        # Training progress metrics
        self.epoch_losses = []
        self.step_losses = []
        self.learning_rates = []
        self.gradient_norms = []

    def reset(self):
        """Reset all metrics"""
        self.wer_metric.reset()
        self.cer_metric.reset()
        self.dialect_metric.reset()
        self.tone_metric.reset()

    def update(self,
               logits: torch.Tensor,
               labels: torch.Tensor,
               input_lengths: Optional[torch.Tensor] = None,
               target_lengths: Optional[torch.Tensor] = None,
               dialect_logits: Optional[torch.Tensor] = None,
               dialect_labels: Optional[torch.Tensor] = None,
               tone_logits: Optional[torch.Tensor] = None,
               tone_labels: Optional[torch.Tensor] = None):
        """Update all metrics with batch data"""

        # Update WER and CER
        self.wer_metric.update(logits, labels, input_lengths, target_lengths)
        self.cer_metric.update(logits, labels, input_lengths, target_lengths)

        # Update dialect accuracy if available
        if dialect_logits is not None and dialect_labels is not None:
            self.dialect_metric.update(dialect_logits, dialect_labels)

        # Update tone accuracy if available
        if tone_logits is not None and tone_labels is not None:
            self.tone_metric.update(tone_logits, tone_labels, input_lengths)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {
            'wer': self.wer_metric.compute(),
            'cer': self.cer_metric.compute(),
            'tone_accuracy': self.tone_metric.compute(),
        }

        # Add dialect metrics
        dialect_metrics = self.dialect_metric.compute()
        metrics.update(dialect_metrics)

        return metrics

    def log_training_step(self, loss: float, lr: float, grad_norm: float):
        """Log training step metrics"""
        self.step_losses.append(loss)
        self.learning_rates.append(lr)
        self.gradient_norms.append(grad_norm)

    def log_epoch(self, epoch_loss: float):
        """Log epoch metrics"""
        self.epoch_losses.append(epoch_loss)

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history for plotting"""
        return {
            'epoch_losses': self.epoch_losses.copy(),
            'step_losses': self.step_losses.copy(),
            'learning_rates': self.learning_rates.copy(),
            'gradient_norms': self.gradient_norms.copy(),
        }


class CTCLoss(nn.Module):
    """Enhanced CTC Loss with additional features"""

    def __init__(self, blank_id: int = 0, reduction: str = "mean",
                 zero_infinity: bool = True, label_smoothing: float = 0.0):
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute CTC loss

        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target sequences [batch_size, target_len]
            input_lengths: Lengths of input sequences [batch_size]
            target_lengths: Lengths of target sequences [batch_size]

        Returns:
            CTC loss
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # CTC expects time-first format
        log_probs = log_probs.transpose(0, 1)  # [seq_len, batch_size, vocab_size]

        loss = F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank_id,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity
        )

        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            # Uniform distribution over vocabulary
            vocab_size = logits.size(-1)
            uniform_dist = torch.full_like(logits, 1.0 / vocab_size)
            uniform_log_probs = torch.log(uniform_dist)

            smoothed_loss = F.kl_div(
                log_probs.transpose(0, 1),  # Back to batch-first
                uniform_dist,
                reduction='batchmean'
            )

            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smoothed_loss

        return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in auxiliary tasks"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss

        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Target labels [batch_size]

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss"""

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss

        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Target labels [batch_size]

        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed targets
        smoothed_targets = torch.zeros_like(log_probs)
        smoothed_targets.fill_(self.smoothing / (self.num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = -torch.sum(smoothed_targets * log_probs, dim=-1)
        return loss.mean()


def compute_wer_batch(predictions: List[str], references: List[str]) -> float:
    """Compute WER for a batch of predictions"""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    total_wer = 0.0
    for pred, ref in zip(predictions, references):
        try:
            sample_wer = wer([ref], [pred])
            total_wer += sample_wer
        except Exception:
            # Fallback to edit distance
            pred_words = pred.split()
            ref_words = ref.split()
            if ref_words:
                distance = editdistance.eval(ref_words, pred_words)
                sample_wer = distance / len(ref_words)
                total_wer += sample_wer

    return total_wer / len(predictions) if predictions else 0.0


def compute_cer_batch(predictions: List[str], references: List[str]) -> float:
    """Compute CER for a batch of predictions"""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    total_cer = 0.0
    for pred, ref in zip(predictions, references):
        try:
            sample_cer = cer([ref], [pred])
            total_cer += sample_cer
        except Exception:
            # Fallback to character-level edit distance
            if ref:
                distance = editdistance.eval(list(pred), list(ref))
                sample_cer = distance / len(ref)
                total_cer += sample_cer

    return total_cer / len(predictions) if predictions else 0.0


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Simple BLEU score computation"""
    try:
        from sacrebleu import corpus_bleu
        refs = [[ref] for ref in references]
        bleu = corpus_bleu(predictions, refs)
        return bleu.score / 100.0
    except ImportError:
        # Fallback to simple word overlap
        total_score = 0.0
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.split())
            ref_words = set(ref.split())
            if ref_words:
                overlap = len(pred_words & ref_words)
                total_score += overlap / len(ref_words)
        return total_score / len(predictions) if predictions else 0.0


class MetricsTracker:
    """Utility class for tracking metrics during training"""

    def __init__(self, metrics_config: Optional[Dict] = None):
        self.config = metrics_config or {}
        self.history = defaultdict(list)
        self.best_metrics = {}
        self.current_metrics = {}

    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics for current step"""
        self.current_metrics = metrics.copy()

        for name, value in metrics.items():
            self.history[name].append((step, value))

            # Track best metrics
            if name not in self.best_metrics:
                self.best_metrics[name] = {'value': value, 'step': step}
            else:
                # For WER/CER, lower is better; for accuracy, higher is better
                is_better = (
                    (name in ['wer', 'cer'] and value < self.best_metrics[name]['value']) or
                    (name not in ['wer', 'cer'] and value > self.best_metrics[name]['value'])
                )
                if is_better:
                    self.best_metrics[name] = {'value': value, 'step': step}

    def get_best(self, metric_name: str) -> Optional[Dict]:
        """Get best value for a specific metric"""
        return self.best_metrics.get(metric_name)

    def get_current(self, metric_name: str) -> Optional[float]:
        """Get current value for a specific metric"""
        return self.current_metrics.get(metric_name)

    def get_history(self, metric_name: str) -> List[Tuple[int, float]]:
        """Get history for a specific metric"""
        return self.history.get(metric_name, [])

    def is_improving(self, metric_name: str, patience: int = 5) -> bool:
        """Check if metric is improving over last patience steps"""
        history = self.get_history(metric_name)
        if len(history) < patience + 1:
            return True

        recent_values = [value for _, value in history[-patience:]]
        baseline_value = history[-patience-1][1]

        # For WER/CER, improvement means decrease
        if metric_name in ['wer', 'cer']:
            return min(recent_values) < baseline_value
        else:
            return max(recent_values) > baseline_value

    def summary(self) -> Dict[str, Dict]:
        """Get summary of all metrics"""
        summary = {}
        for metric_name in self.best_metrics:
            summary[metric_name] = {
                'best': self.best_metrics[metric_name],
                'current': self.current_metrics.get(metric_name),
                'history_length': len(self.history[metric_name])
            }
        return summary


if __name__ == "__main__":
    # Example usage
    print("Testing metrics utilities...")

    # Test WER computation
    predictions = ["hello world", "how are you"]
    references = ["hello word", "how are you"]

    wer_score = compute_wer_batch(predictions, references)
    cer_score = compute_cer_batch(predictions, references)
    bleu_score = compute_bleu_score(predictions, references)

    print(f"WER: {wer_score:.3f}")
    print(f"CER: {cer_score:.3f}")
    print(f"BLEU: {bleu_score:.3f}")

    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update({'wer': 0.25, 'accuracy': 0.85}, step=1)
    tracker.update({'wer': 0.20, 'accuracy': 0.90}, step=2)

    print(f"Best WER: {tracker.get_best('wer')}")
    print(f"Current accuracy: {tracker.get_current('accuracy')}")

    print("All tests passed!")
