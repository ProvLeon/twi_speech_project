"""
Evaluation module for the Twi Speech Training Engine
"""

from .evaluator import (
    SpeechEvaluator,
    evaluate_model,
    compute_metrics,
    generate_evaluation_report
)

__all__ = [
    'SpeechEvaluator',
    'evaluate_model',
    'compute_metrics',
    'generate_evaluation_report'
]
