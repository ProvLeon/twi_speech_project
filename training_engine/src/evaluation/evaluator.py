"""
Comprehensive Evaluation Module for Twi Speech Recognition

This module provides evaluation metrics and tools specifically designed for
assessing Twi speech recognition systems. It includes standard ASR metrics
like WER and CER, as well as Twi-specific evaluation criteria such as
dialect classification accuracy and tone error analysis.

Key Features:
- Word Error Rate (WER) and Character Error Rate (CER) calculation
- Dialect-aware evaluation metrics
- Tone accuracy assessment
- Code-switching detection evaluation
- Detailed error analysis and visualization
- Statistical significance testing
- Performance benchmarking tools

Author: Twi Speech Recognition Team
"""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import statistics

import torch
import numpy as np
from jiwer import wer, cer
try:
    from jiwer import compute_measures
except ImportError:
    # Fallback for older jiwer versions
    compute_measures = None
import editdistance
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    wer: float
    cer: float
    bleu: float
    dialect_accuracy: Optional[float] = None
    tone_accuracy: Optional[float] = None
    code_switch_accuracy: Optional[float] = None

    # Detailed metrics
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0
    hits: int = 0

    # Per-dialect metrics
    dialect_wer: Optional[Dict[str, float]] = None
    dialect_cer: Optional[Dict[str, float]] = None

    # Error analysis
    common_errors: Optional[Dict[str, int]] = None
    tone_errors: Optional[Dict[str, int]] = None


class TwiTextNormalizer:
    """Text normalization utilities for Twi language"""

    def __init__(self):
        # Twi-specific character mappings
        self.char_mappings = {
            'ɔ': 'ɔ',  # Ensure consistent encoding
            'ɛ': 'ɛ',
            'ŋ': 'ŋ',
        }

        # Tone marks
        self.tone_marks = ['́', '̀', '̂', '̃', '̄', '̆', '̇']

        # Common contractions and variations
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
        }

    def normalize(self, text: str, remove_tones: bool = False) -> str:
        """
        Normalize Twi text for evaluation

        Args:
            text: Input text
            remove_tones: Whether to remove tone marks

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize('NFC', text)

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Handle contractions (for code-switching scenarios)
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # Standardize Twi characters
        for old_char, new_char in self.char_mappings.items():
            text = text.replace(old_char, new_char)

        # Remove tone marks if requested
        if remove_tones:
            for tone in self.tone_marks:
                text = text.replace(tone, '')

        # Remove punctuation for WER calculation
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def extract_tones(self, text: str) -> List[str]:
        """
        Extract tone marks from text

        Args:
            text: Input text with tone marks

        Returns:
            List of tone marks in order
        """
        tones = []
        for char in text:
            if char in self.tone_marks:
                tones.append(char)
        return tones


class TwiEvaluator:
    """Main evaluator class for Twi speech recognition"""

    def __init__(self, processor=None, config: Optional[Dict] = None):
        """
        Initialize evaluator

        Args:
            processor: Model processor for decoding
            config: Evaluation configuration
        """
        self.processor = processor
        self.config = config or {}
        self.normalizer = TwiTextNormalizer()

        # Twi dialects
        self.dialects = ['Asante', 'Akuapem', 'Fante']
        self.dialect_map = {dialect: i for i, dialect in enumerate(self.dialects)}

        # Language codes for code-switching
        self.languages = ['tw', 'en']  # Twi and English

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        dialect_predictions: Optional[List[int]] = None,
        dialect_references: Optional[List[int]] = None,
        tone_predictions: Optional[List[str]] = None,
        tone_references: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics

        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            dialect_predictions: Predicted dialect labels
            dialect_references: Reference dialect labels
            tone_predictions: Predicted text with tones
            tone_references: Reference text with tones

        Returns:
            Dictionary of evaluation metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        # Normalize texts
        norm_predictions = [self.normalizer.normalize(pred) for pred in predictions]
        norm_references = [self.normalizer.normalize(ref) for ref in references]

        # Core ASR metrics
        wer_score = self._compute_wer(norm_predictions, norm_references)
        cer_score = self._compute_cer(norm_predictions, norm_references)
        bleu_score = self._compute_bleu(norm_predictions, norm_references)

        # Detailed error analysis
        error_metrics = self._compute_detailed_errors(norm_predictions, norm_references)

        metrics = {
            'wer': wer_score,
            'cer': cer_score,
            'bleu': bleu_score,
            **error_metrics
        }

        # Dialect classification metrics
        if dialect_predictions is not None and dialect_references is not None:
            dialect_metrics = self._compute_dialect_metrics(
                dialect_predictions, dialect_references
            )
            metrics.update(dialect_metrics)

        # Tone accuracy metrics
        if tone_predictions is not None and tone_references is not None:
            tone_metrics = self._compute_tone_metrics(
                tone_predictions, tone_references
            )
            metrics.update(tone_metrics)

        # Per-dialect WER/CER if dialect info available
        if dialect_references is not None:
            per_dialect_metrics = self._compute_per_dialect_metrics(
                norm_predictions, norm_references, dialect_references
            )
            metrics.update(per_dialect_metrics)

        return metrics

    def _compute_wer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Word Error Rate"""
        try:
            return wer(references, predictions)
        except Exception as e:
            logger.warning(f"Error computing WER: {e}")
            return float('inf')

    def _compute_cer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Character Error Rate"""
        try:
            return cer(references, predictions)
        except Exception as e:
            logger.warning(f"Error computing CER: {e}")
            return float('inf')

    def _compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score"""
        try:
            from sacrebleu import corpus_bleu
            # Convert to format expected by sacrebleu
            refs = [[ref] for ref in references]
            bleu = corpus_bleu(predictions, refs)
            return bleu.score / 100.0  # Convert to 0-1 range
        except ImportError:
            logger.warning("sacrebleu not available, using simple BLEU approximation")
            return self._simple_bleu(predictions, references)
        except Exception as e:
            logger.warning(f"Error computing BLEU: {e}")
            return 0.0

    def _simple_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Simple BLEU approximation"""
        total_score = 0.0
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.split())
            ref_words = set(ref.split())
            if ref_words:
                overlap = len(pred_words & ref_words)
                total_score += overlap / len(ref_words)
        return total_score / len(predictions) if predictions else 0.0

    def _compute_detailed_errors(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Union[int, float]]:
        """Compute detailed error analysis"""
        total_insertions = 0
        total_deletions = 0
        total_substitutions = 0
        total_hits = 0

        for pred, ref in zip(predictions, references):
            try:
                measures = compute_measures(ref, pred)
                total_insertions += measures['insertions']
                total_deletions += measures['deletions']
                total_substitutions += measures['substitutions']
                total_hits += measures['hits']
            except Exception:
                # Fallback to edit distance
                pred_words = pred.split()
                ref_words = ref.split()
                distance = editdistance.eval(ref_words, pred_words)
                total_substitutions += distance
                total_hits += max(0, len(ref_words) - distance)

        total_words = total_hits + total_substitutions + total_deletions

        return {
            'insertions': total_insertions,
            'deletions': total_deletions,
            'substitutions': total_substitutions,
            'hits': total_hits,
            'insertion_rate': total_insertions / max(total_words, 1),
            'deletion_rate': total_deletions / max(total_words, 1),
            'substitution_rate': total_substitutions / max(total_words, 1),
        }

    def _compute_dialect_metrics(
        self, predictions: List[int], references: List[int]
    ) -> Dict[str, float]:
        """Compute dialect classification metrics"""
        try:
            accuracy = accuracy_score(references, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                references, predictions, average='weighted', zero_division=0
            )

            return {
                'dialect_accuracy': accuracy,
                'dialect_precision': precision,
                'dialect_recall': recall,
                'dialect_f1': f1,
            }
        except Exception as e:
            logger.warning(f"Error computing dialect metrics: {e}")
            return {
                'dialect_accuracy': 0.0,
                'dialect_precision': 0.0,
                'dialect_recall': 0.0,
                'dialect_f1': 0.0,
            }

    def _compute_tone_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute tone accuracy metrics"""
        try:
            total_tones = 0
            correct_tones = 0

            for pred, ref in zip(predictions, references):
                pred_tones = self.normalizer.extract_tones(pred)
                ref_tones = self.normalizer.extract_tones(ref)

                # Align tones (simple matching)
                min_len = min(len(pred_tones), len(ref_tones))
                total_tones += len(ref_tones)
                correct_tones += sum(1 for i in range(min_len) if pred_tones[i] == ref_tones[i])

            tone_accuracy = correct_tones / max(total_tones, 1)

            return {
                'tone_accuracy': tone_accuracy,
                'tone_coverage': min_len / max(len(ref_tones), 1) if 'ref_tones' in locals() else 0.0,
            }
        except Exception as e:
            logger.warning(f"Error computing tone metrics: {e}")
            return {'tone_accuracy': 0.0, 'tone_coverage': 0.0}

    def _compute_per_dialect_metrics(
        self, predictions: List[str], references: List[str], dialects: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """Compute WER/CER per dialect"""
        try:
            dialect_metrics = {}

            for dialect_idx, dialect_name in enumerate(self.dialects):
                # Filter samples for this dialect
                dialect_preds = [pred for pred, d in zip(predictions, dialects) if d == dialect_idx]
                dialect_refs = [ref for ref, d in zip(references, dialects) if d == dialect_idx]

                if dialect_refs:  # Only compute if we have samples
                    dialect_wer = self._compute_wer(dialect_preds, dialect_refs)
                    dialect_cer = self._compute_cer(dialect_preds, dialect_refs)

                    dialect_metrics[f'{dialect_name.lower()}_wer'] = dialect_wer
                    dialect_metrics[f'{dialect_name.lower()}_cer'] = dialect_cer

            return dialect_metrics
        except Exception as e:
            logger.warning(f"Error computing per-dialect metrics: {e}")
            return {}

    def analyze_errors(
        self, predictions: List[str], references: List[str], top_k: int = 10
    ) -> Dict[str, List[Tuple[str, str, int]]]:
        """
        Analyze common errors in predictions

        Args:
            predictions: Predicted transcriptions
            references: Reference transcriptions
            top_k: Number of top errors to return

        Returns:
            Dictionary with error analysis
        """
        substitution_errors = Counter()
        insertion_errors = Counter()
        deletion_errors = Counter()

        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()

            # Simple alignment for error analysis
            pred_idx, ref_idx = 0, 0

            while pred_idx < len(pred_words) and ref_idx < len(ref_words):
                if pred_words[pred_idx] == ref_words[ref_idx]:
                    pred_idx += 1
                    ref_idx += 1
                else:
                    # Check for insertion
                    if pred_idx + 1 < len(pred_words) and pred_words[pred_idx + 1] == ref_words[ref_idx]:
                        insertion_errors[pred_words[pred_idx]] += 1
                        pred_idx += 1
                    # Check for deletion
                    elif ref_idx + 1 < len(ref_words) and pred_words[pred_idx] == ref_words[ref_idx + 1]:
                        deletion_errors[ref_words[ref_idx]] += 1
                        ref_idx += 1
                    # Substitution
                    else:
                        substitution_errors[(ref_words[ref_idx], pred_words[pred_idx])] += 1
                        pred_idx += 1
                        ref_idx += 1

        return {
            'top_substitutions': substitution_errors.most_common(top_k),
            'top_insertions': insertion_errors.most_common(top_k),
            'top_deletions': deletion_errors.most_common(top_k),
        }

    def compute_confidence_intervals(
        self, predictions: List[str], references: List[str], confidence: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals for metrics using bootstrap sampling

        Args:
            predictions: Predicted transcriptions
            references: Reference transcriptions
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary with confidence intervals
        """
        try:
            import scipy.stats as stats

            n_bootstrap = 1000
            n_samples = len(predictions)

            wer_samples = []
            cer_samples = []

            for _ in range(n_bootstrap):
                # Bootstrap sampling
                indices = np.random.choice(n_samples, n_samples, replace=True)
                boot_preds = [predictions[i] for i in indices]
                boot_refs = [references[i] for i in indices]

                # Compute metrics
                boot_wer = self._compute_wer(boot_preds, boot_refs)
                boot_cer = self._compute_cer(boot_preds, boot_refs)

                wer_samples.append(boot_wer)
                cer_samples.append(boot_cer)

            # Compute confidence intervals
            alpha = 1 - confidence
            wer_ci = np.percentile(wer_samples, [100 * alpha/2, 100 * (1 - alpha/2)])
            cer_ci = np.percentile(cer_samples, [100 * alpha/2, 100 * (1 - alpha/2)])

            return {
                'wer_ci': (float(wer_ci[0]), float(wer_ci[1])),
                'cer_ci': (float(cer_ci[0]), float(cer_ci[1])),
            }
        except ImportError:
            logger.warning("scipy not available for confidence intervals")
            return {}
        except Exception as e:
            logger.warning(f"Error computing confidence intervals: {e}")
            return {}

    def generate_report(
        self,
        predictions: List[str],
        references: List[str],
        dialect_predictions: Optional[List[int]] = None,
        dialect_references: Optional[List[int]] = None,
        model_name: str = "Twi ASR Model"
    ) -> str:
        """
        Generate a comprehensive evaluation report

        Args:
            predictions: Predicted transcriptions
            references: Reference transcriptions
            dialect_predictions: Predicted dialect labels
            dialect_references: Reference dialect labels
            model_name: Name of the model being evaluated

        Returns:
            Formatted evaluation report
        """
        # Compute all metrics
        metrics = self.compute_metrics(
            predictions, references, dialect_predictions, dialect_references
        )

        # Error analysis
        errors = self.analyze_errors(predictions, references)

        # Confidence intervals
        ci = self.compute_confidence_intervals(predictions, references)

        # Generate report
        report = f"""
# Twi Speech Recognition Evaluation Report

## Model: {model_name}
## Dataset Size: {len(predictions)} samples

### Core ASR Metrics
- Word Error Rate (WER): {metrics['wer']:.3f}
- Character Error Rate (CER): {metrics['cer']:.3f}
- BLEU Score: {metrics['bleu']:.3f}

### Detailed Error Analysis
- Insertions: {metrics.get('insertions', 0)} ({metrics.get('insertion_rate', 0):.1%})
- Deletions: {metrics.get('deletions', 0)} ({metrics.get('deletion_rate', 0):.1%})
- Substitutions: {metrics.get('substitutions', 0)} ({metrics.get('substitution_rate', 0):.1%})
- Hits: {metrics.get('hits', 0)}

### Confidence Intervals (95%)"""

        if ci:
            report += f"""
- WER: [{ci['wer_ci'][0]:.3f}, {ci['wer_ci'][1]:.3f}]
- CER: [{ci['cer_ci'][0]:.3f}, {ci['cer_ci'][1]:.3f}]"""

        # Dialect metrics
        if 'dialect_accuracy' in metrics:
            report += f"""

### Dialect Classification
- Accuracy: {metrics['dialect_accuracy']:.3f}
- Precision: {metrics['dialect_precision']:.3f}
- Recall: {metrics['dialect_recall']:.3f}
- F1-Score: {metrics['dialect_f1']:.3f}"""

        # Per-dialect metrics
        dialect_wer_found = any(key.endswith('_wer') for key in metrics.keys() if '_' in key)
        if dialect_wer_found:
            report += "\n\n### Per-Dialect Performance"
            for dialect in self.dialects:
                dialect_key = dialect.lower()
                if f'{dialect_key}_wer' in metrics:
                    report += f"\n- {dialect}: WER {metrics[f'{dialect_key}_wer']:.3f}, CER {metrics[f'{dialect_key}_cer']:.3f}"

        # Common errors
        if errors['top_substitutions']:
            report += "\n\n### Top Substitution Errors"
            for (ref, pred), count in errors['top_substitutions'][:5]:
                report += f"\n- '{ref}' → '{pred}' ({count} times)"

        # Target achievement
        target_wer = 0.30  # 30% target
        if metrics['wer'] <= target_wer:
            report += f"\n\n✅ **TARGET ACHIEVED**: WER {metrics['wer']:.1%} is below target of {target_wer:.1%}"
        else:
            improvement_needed = metrics['wer'] - target_wer
            report += f"\n\n❌ **Target not met**: WER {metrics['wer']:.1%} exceeds target of {target_wer:.1%} by {improvement_needed:.1%}"

        return report

    def benchmark_against_baseline(
        self,
        predictions: List[str],
        references: List[str],
        baseline_predictions: List[str],
        model_name: str = "Current Model",
        baseline_name: str = "Baseline Model"
    ) -> Dict[str, float]:
        """
        Benchmark current model against a baseline

        Args:
            predictions: Current model predictions
            references: Reference transcriptions
            baseline_predictions: Baseline model predictions
            model_name: Name of current model
            baseline_name: Name of baseline model

        Returns:
            Comparison metrics
        """
        # Compute metrics for both models
        current_metrics = self.compute_metrics(predictions, references)
        baseline_metrics = self.compute_metrics(baseline_predictions, references)

        # Compute improvements
        improvements = {}
        for metric in ['wer', 'cer', 'bleu']:
            if metric in current_metrics and metric in baseline_metrics:
                if metric == 'bleu':  # Higher is better for BLEU
                    improvement = current_metrics[metric] - baseline_metrics[metric]
                else:  # Lower is better for WER/CER
                    improvement = baseline_metrics[metric] - current_metrics[metric]
                improvements[f'{metric}_improvement'] = improvement
                improvements[f'{metric}_relative_improvement'] = improvement / max(baseline_metrics[metric], 1e-8)

        return {
            f'{model_name}_metrics': current_metrics,
            f'{baseline_name}_metrics': baseline_metrics,
            'improvements': improvements
        }


# Utility functions
def load_evaluation_data(file_path: str) -> Tuple[List[str], List[str]]:
    """Load evaluation data from file"""
    predictions, references = [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    pred, ref = line.strip().split('\t', 1)
                    predictions.append(pred)
                    references.append(ref)
    except Exception as e:
        logger.error(f"Error loading evaluation data: {e}")
        raise

    return predictions, references


def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to file"""
    try:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")


if __name__ == "__main__":
    # Example usage
    evaluator = TwiEvaluator()

    # Test data
    predictions = [
        "me din de kofi",
        "wo ho te sɛn",
        "mepɛ aduan bi"
    ]

    references = [
        "me din de kwame",
        "wo ho te sɛn",
        "mepɛ aduan kakra"
    ]

    # Compute metrics
    metrics = evaluator.compute_metrics(predictions, references)
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Generate report
    report = evaluator.generate_report(predictions, references)
    print("\n" + report)
