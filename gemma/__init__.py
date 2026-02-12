"""
VisualCite Attribution Benchmark

Benchmark Qwen3-VL models on cell attribution task for tabular data.
"""

__version__ = "1.0.0"

from .config import BenchmarkConfig, DataRepresentation, PromptStrategy, ModelSize
from .data_loader import VisualCiteDataset, VisualCiteSample
from .metrics import evaluate_single_prediction, aggregate_metrics, CellMetrics, AggregatedMetrics
from .model_handler import Qwen3VLModel, ModelManager
from .prompt_builder import PromptBuilder
from .checkpoint_manager import CheckpointManager
from .result_logger import ResultLogger

__all__ = [
    "BenchmarkConfig",
    "DataRepresentation", 
    "PromptStrategy",
    "ModelSize",
    "VisualCiteDataset",
    "VisualCiteSample",
    "evaluate_single_prediction",
    "aggregate_metrics",
    "CellMetrics",
    "AggregatedMetrics",
    "Qwen3VLModel",
    "ModelManager",
    "PromptBuilder",
    "CheckpointManager",
    "ResultLogger"
]