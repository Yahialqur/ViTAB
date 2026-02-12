# Qwen3-VL VisualCite Attribution Benchmark

This directory contains the implementation for benchmarking Qwen3-VL vision-language models on the VisualCite cell attribution task. The system evaluates models across multiple dimensions: model sizes (2B-32B parameters), data representations (text and images), prompting strategies, and confidence measurement frameworks.

## Directory Overview

The codebase is organized into several functional modules that work together to orchestrate comprehensive benchmarking experiments with checkpoint/resume capabilities and multi-framework confidence measurement.

---

## Core Files

### `__init__.py`
Package initialization file that exports the main public API. Provides centralized access to key classes and functions used throughout the benchmark system.

### `config.py`
Central configuration hub containing all benchmark settings, model specifications, and prompt templates.

**Key Components:**
- **Enums**: `ModelSize` (2B, 4B, 8B, 32B), `DataRepresentation` (JSON, Markdown, 5 image styles), `PromptStrategy` (zero-shot, few-shot, chain-of-thought)
- **BenchmarkConfig**: Dataclass with all configurable parameters (dataset paths, model settings, confidence extraction options, hardware settings, checkpoint intervals)
- **PROMPT_TEMPLATES**: Dictionary mapping each prompt strategy to its template string
- **COT_EXTRACTION_SUFFIX**: Template suffix used in two-step chain-of-thought inference to extract final answers from reasoning

---

## Data & Model Infrastructure

### `data_loader.py`
Handles loading, parsing, and preprocessing of the VisualCite dataset.

**Key Functions:**
- **VisualCiteSample**: Dataclass representing a single benchmark sample (question, answer, table in multiple formats, ground truth cells)
- **VisualCiteDataset**: Dataset loader that reads from JSONL, filters by split, and provides iteration
- **decode_image()**: Decodes base64-encoded table images to PIL Images
- **parse_model_output()**: Parses model predictions into standardized cell coordinate format
- **extract_cells_from_formulas()**: Extracts individual cell references from Excel-style formulas

### `model_handler.py`
Manages Qwen3-VL model lifecycle, inference, and multi-image message handling.

**Key Classes:**
- **Qwen3VLModel**: Wraps a Qwen3-VL model and processor with generation methods
  - `generate_text()`: For text-only prompts
  - `generate_with_image()`: For single or multi-image prompts with proper message content structure
  - `generate_with_logits()`: Returns generated text, full logit tensor [seq_len, vocab_size], and token IDs for confidence extraction
- **ModelManager**: Factory for loading/unloading models with memory management
- **InferenceResult**: Dataclass storing outputs, timing, and confidence extraction data (logits, token_probabilities, generated_token_ids)

**Features:**
- Multi-image support via message content arrays: `[{type:"image"}, {type:"text"}, {type:"image"}, {type:"text"}]`
- `<IMAGE>` marker replacement for explicit image positioning
- Comprehensive logging of image dimensions and message structure
- Configurable temperature, sampling, and token limits

### `prompt_builder.py`
Constructs prompts for different strategies and handles few-shot example selection.

**Key Methods:**
- **build_prompt()**: Main entry point that returns `(prompt_text, example_image)` tuple
  - Selects appropriate template based on strategy
  - For few-shot: loads real examples from validation split with matching representation style
  - For images: uses `<IMAGE>` markers instead of embedding table text
  - For text (JSON/Markdown): embeds actual table text in prompt
- **build_cot_extraction_prompt()**: Creates second-stage prompt for extracting final answer from chain-of-thought reasoning
- **_load_few_shot_examples()**: Loads validation split examples for few-shot learning

**Representation Handling:**
- Text representations: Directly embed table as JSON or Markdown string
- Image representations: Use `<IMAGE>` markers, load matching style images (arial, times_new_roman, red, blue, green)
- Few-shot examples automatically match the target representation format

---

## Benchmark Execution

### `benchmark_runner.py`
Main orchestration script for running comprehensive attribution benchmarks.

**Workflow:**
1. Load dataset and configuration
2. For each (model, representation, strategy) combination:
   - Check for existing checkpoint
   - Initialize model
   - Load few-shot examples (for few-shot strategy)
   - Run inference on each sample (with progress tracking)
   - Extract internal confidence if enabled
   - Query verbalized certainty if enabled
   - Save results incrementally via checkpoints
3. Export results to JSON and CSV
4. Calculate aggregate metrics

**Features:**
- Checkpoint/resume capability (saves every N samples)
- Multi-modal inference (handles both text and image tables)
- Logs verbalized certainty queries to separate JSONL files
- Tracks inference time per sample
- Comprehensive logging with run identifiers

### `confidence_benchmark_runner.py`
Specialized benchmark runner focused on confidence-probability alignment evaluation following "Confidence Under the Hood" (ACL 2024) methodology.

**Pipeline:**
1. Present cell attribution as multiple-choice question
2. Extract internal confidence from token probabilities during generation
3. Query verbalized certainty using Confidence Querying Prompt (CQP)
4. Compute alignment metrics (Spearman's ρ, calibration curves)
5. Analyze confidence-correctness relationship

**Differences from standard benchmark:**
- Forces multiple-choice format for probability extraction
- Focuses exclusively on confidence measurement
- Computes specialized alignment metrics
- Generates confidence calibration plots

---

## Confidence Measurement Framework

### Internal Confidence Extraction Methodology

The system implements a robust joint probability approach for extracting confidence from token-level probabilities, specifically designed for multi-token cell attribution tasks:

**Algorithm Overview:**
1. **Token Identification**: Locate answer tokens within the full generated sequence
   - Searches for multiple variations (with leading spaces, newlines, equals signs)
   - Selects the match appearing latest in the sequence (the final answer)
   - Handles Excel-style coordinates with "=" prefix (e.g., "=A12")

2. **Probability Computation**: Calculate geometric mean of token probabilities
   - Formula: `confidence = exp((1/N) × Σ log(p_i))` where N is the number of answer tokens
   - Converts logits to log-probabilities via log-softmax for numerical stability
   - More robust than max probability for multi-token answers ("A12" has 3 tokens: "A", "1", "2")

3. **Cell-Level Confidence**: Compute confidence for each predicted cell independently
   - Each cell coordinate may span multiple tokens
   - Cells not found in output receive 0.0 confidence

4. **Aggregation**: Combine cell confidences (default: arithmetic mean)
   - Options: mean, max, min, or product across all predicted cells

**Why Geometric Mean?**
- **Joint probability interpretation**: Represents the probability of generating the entire token sequence
- **Length normalization**: Comparable across different cell coordinates ("A1" vs "AB12")
- **Numerical stability**: Uses log-space computation to avoid underflow
- **Superior to max**: Max only considers the most confident token, ignoring evidence from others

### `confidence_types.py`
Defines data structures for confidence measurements and analysis.

**Key Classes:**
- **CertaintyLevel**: Enum mapping 6-point Likert scale (a-f) to numerical values (0.0-1.0)
- **ConfidenceResult**: Stores internal confidence, verbalized certainty, and alignment metrics for a single sample
- **CellConfidenceResult**: Per-cell confidence scores with alignment metrics
- **AggregatedConfidenceMetrics**: Aggregated statistics across all samples (mean/median confidence, Spearman correlation, calibration data)

### `confidence_extractor.py`
Core confidence measurement implementation with two extraction methods.

**Key Classes:**
- **InternalConfidenceExtractor**: Extracts confidence from token probability distributions using joint probability methodology
  - **Joint Probability Approach**: Computes geometric mean of token probabilities (exp(mean(log(p_i)))) instead of max probability
  - Handles multi-token cell coordinates (e.g., "A12", "AB5") correctly by combining evidence across all tokens
  - **Token Alignment**: Strictly aligns tokens with the final answer, ignoring reasoning steps in chain-of-thought
  - **Robust Token Matching**: Handles leading spaces, newlines, and equals signs ("=A1") via multiple search strategies
  - **Cell-Level Confidence**: Computes confidence per predicted cell, then aggregates (mean/max/min/product)
  - Based on Algorithm 1 from "Confidence Under the Hood" (ACL 2024), adapted for multi-token generation
- **VerbalizedCertaintyExtractor**: Queries model for explicit confidence statements
  - Uses Confidence Querying Prompt (CQP)
  - Parses natural language certainty into 0.0-1.0 scale via CertaintyLevel enum
  - Separate logging to `verbalized_certainty_logs/` for analysis

### `verbalized_certainty.py`
Implements verbalized certainty extraction via Confidence Querying Prompts.

**Key Components:**
- **CQP_TEMPLATE_CELL_ATTRIBUTION**: Prompt template asking model to self-assess confidence
  - **Third-Person Perspective (TPP)**: Presents Q&A as from another model to reduce bias
  - **Option Contextualization (OC)**: Provides all table cells for comparative evaluation
  - **Likert Scale Utilization (LSU)**: Uses qualitative certainty scale for nuanced assessment
- **query_confidence()**: Sends CQP to model and retrieves response
- **extract_certainty_from_response()**: Parses response to identify certainty level (a-f) and converts to numerical score
- **format_cells_by_row()**: Formats all valid table cells grouped by row for prompt clarity

**Framework:**
- 6-point Likert scale: (a) Completely confident → (f) Not confident at all
- Maps to [1.0, 0.8, 0.6, 0.4, 0.2, 0.0] numerical scale
- Robust parsing with fallback to None on unclear responses

**Key Functions:**
- **find_answer_token_indices()**: Locates answer tokens within generated sequence using robust multi-strategy search
- **compute_geometric_mean_confidence()**: Computes geometric mean of token probabilities using log-space arithmetic
- **extract_cell_confidences()**: Extracts confidence for each predicted cell by finding and scoring its tokens
- **compute_aggregate_confidence()**: Aggregates cell-level confidences (mean/max/min/product)
- **compute_all_cell_probabilities()**: Computes probabilities for all valid table cells (for conformal prediction)

### `uncertainty_quantification.py`
Implements conformal prediction for uncertainty quantification and prediction set construction.

**Key Functions:**
- **split_conformal_prediction()**: Splits dataset into calibration and test sets
- **construct_prediction_sets()**: Builds prediction sets at specified coverage level
- **calibrate_confidence_scores()**: Calibrates raw confidence scores using calibration set
- **compute_coverage_metrics()**: Evaluates empirical coverage and prediction set sizes
- **run_split_conformal()**: End-to-end conformal prediction pipeline with coverage analysis

**Methodology:**
- Split conformal prediction for distribution-free uncertainty quantification
- Constructs prediction sets with theoretical coverage guarantees
- Useful for safety-critical applications requiring valid uncertainty estimates

### `alignment_metrics.py`
Computes metrics for analyzing confidence-probability alignment and cell-level attribution quality.

**Metrics:**
- **Spearman's rank correlation** between internal confidence and verbalized certainty
- **Calibration curves** (expected vs observed correctness by confidence bin)
- **Brier score** for probabilistic forecast quality
- **Expected Calibration Error (ECE)** for measuring calibration
- **Cell-level alignment**: Correlates per-cell confidence with per-cell correctness (F1, precision, recall)
- **compute_cell_alignment_metrics()**: Comprehensive alignment analysis with stratification by confidence quartiles
- Stratified analysis by confidence level and correctness

---

## Evaluation & Utilities

### `metrics.py`
Cell attribution evaluation metrics implementation.

**Key Functions:**
- **evaluate_single_prediction()**: Compares predicted cells against ground truth
  - Returns: `CellMetrics` (F1, precision, recall, IoU, exact match)
  - Handles various formats: single cells, multiple cells, ranges, formulas
- **aggregate_metrics()**: Aggregates metrics across multiple samples
  - Returns: `AggregatedMetrics` (means, medians, standard deviations)
- Cell coordinate normalization and comparison utilities

### `table_utils.py`
Helper functions for table coordinate manipulation.

**Functions:**
- **parse_cell_reference()**: Converts Excel-style coordinates ("B5") to (row, col) tuples
- **format_cell_reference()**: Converts (row, col) to Excel notation
- **expand_cell_range()**: Expands ranges ("A1:C3") to list of individual cells
- **normalize_formula()**: Standardizes formula formats for comparison

### `result_logger.py`
Handles result persistence and export to multiple formats.

**Key Classes:**
- **ResultLogger**: Manages result files and exports
  - `log_result()`: Appends result to JSONL file
  - `log_vc_instance()`: Logs verbalized certainty queries separately
  - `export_csv()`: Exports results to CSV with flattened metrics
  - `export_vc_instances_csv()`: Exports VC data to dedicated CSV

**File Structure:**
- `benchmark_results/`: Main results directory
  - `results_<timestamp>.jsonl`: Main benchmark results
  - `results_<timestamp>.csv`: CSV export for analysis
  - `verbalized_certainty_logs/`: Separate VC query logs
    - `vc_<timestamp>.jsonl`: VC query inputs/outputs
    - `vc_<timestamp>.csv`: VC data in tabular format

---

## Supporting Files

### `checkpoint_manager.py`
Manages checkpointing for long-running benchmarks with resume capability.

**Key Functions:**
- **save_checkpoint()**: Persists progress (completed sample IDs, intermediate results)
- **load_checkpoint()**: Restores progress from disk
- **get_run_key()**: Generates unique identifier for (model, representation, strategy) combination
- **should_skip_sample()**: Checks if sample already processed in checkpoint

**Features:**
- Incremental saving (every N samples)
- Per-configuration checkpoints (each model×representation×strategy tracked separately)
- Prevents duplicate processing on resume
- Includes metadata (timestamps, configuration hash)

### `test_benchmark.py`
Unit and integration tests for benchmark components.

**Test Coverage:**
- Data loader functionality
- Model inference pipeline
- Metric calculations
- Prompt building
- Checkpoint save/load
- End-to-end smoke tests

### `requirements.txt`
Python package dependencies for the benchmark system.

**Key Dependencies:**
- `torch`: Deep learning framework
- `transformers`: Qwen3-VL model and processor
- `Pillow`: Image processing
- `tqdm`: Progress bars
- `scipy`: Statistical functions (Spearman correlation)
- `numpy`, `pandas`: Data manipulation
- `pytest`: Testing framework

---

## Execution Flow Summary

### Standard Attribution Benchmark (`benchmark_runner.py`)
```
Load Config → Load Dataset → For each (model, representation, strategy):
  ├─ Load/Resume Checkpoint
  ├─ Initialize Model
  ├─ Build Prompt (with few-shot examples if needed)
  ├─ Run Inference (text or image)
  ├─ Extract Internal Confidence (optional)
  ├─ Query Verbalized Certainty (optional)
  ├─ Evaluate Against Ground Truth
  ├─ Save Checkpoint (every N samples)
  └─ Export Results (JSONL + CSV)
→ Aggregate Metrics → Generate Summary Report
```

### Confidence Benchmark (`confidence_benchmark_runner.py`)
```
Load Config → Load Dataset → For each (model, representation, strategy):
  ├─ Generate Answer with Logits (full token-level probability tracking)
  ├─ Extract Internal Confidence:
  │   ├─ Find answer tokens in generated sequence
  │   ├─ Compute geometric mean of token probabilities per cell
  │   └─ Aggregate cell confidences
  ├─ Query Verbalized Certainty (via CQP with TPP/OC/LSU)
  ├─ Compute Cell-Level Alignment Metrics:
  │   ├─ Per-cell F1, precision, recall
  │   ├─ Confidence-correctness correlation
  │   └─ Stratification by confidence quartiles
  ├─ Compute Aggregate Alignment (Spearman's ρ, ECE, Brier)
  └─ Run Conformal Prediction (optional, for coverage analysis)
→ Aggregate Alignment Statistics → Export Calibration Plots and Cell-Level Reports
```

---

## Key Design Patterns

1. **Representation Polymorphism**: Same interface handles text (JSON/Markdown) and image representations
2. **Strategy Pattern**: Prompt strategies (zero-shot, few-shot, CoT) use common interface with different templates
3. **Checkpoint/Resume**: All long-running operations support incremental progress saving
4. **Multi-Framework Confidence**: Three complementary approaches (internal, verbalized, conformal) for comprehensive confidence measurement
5. **Logging Separation**: Standard results, verbalized certainty queries, and debug logs kept in separate streams
6. **Lazy Loading**: Models loaded on-demand and unloaded after use to manage GPU memory
