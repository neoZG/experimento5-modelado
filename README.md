# Language Model Perplexity Evaluation

This project provides tools for evaluating language model performance through perplexity metrics on standard benchmarks like WikiText-2 and C4 datasets.

## Features

- Supports evaluation of any HuggingFace Transformers-based language model
- Compatible with multiple datasets (WikiText-2 and C4)
- Configurable context length and stride for sliding window evaluation
- Automatic handling of different model precisions (FP16, BF16) based on architecture
- Batch processing for efficient GPU utilization
- Built-in support for specific model architectures (BitNet, Phi-2, Gemma)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/experimento5-modelado.git
cd experimento5-modelado
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the evaluation script with your desired configuration:

```bash
python main.py --dataset wikitext2 --models microsoft/phi-2 google/gemma-2b --context_length 512 --batch_size 8
```

### Command Line Arguments

- `--dataset`: Choose between "wikitext2" or "c4" [required]
- `--models`: Space-separated list of HuggingFace model names to evaluate
- `--context_length`: Context window length in tokens (default: 512)
- `--stride`: Stride for sliding window (default: equal to context_length)
- `--batch_size`: Batch size for parallel inference (default: 8)
- `--max_examples`: Number of examples to sample from C4 (default: 1000)
- `--max_tokens_per_doc`: Maximum tokens per document (default: 1024)
- `--device`: Computation device ("cuda" or "cpu", default: "cuda")
- `--seed`: Random seed for reproducibility (default: 42)

## Project Structure

```
.
├── main.py                 # Main script for running evaluations
├── perplexity_eval.py     # Core perplexity computation logic
├── requirements.txt       # Project dependencies
├── results/              # Evaluation results directory
│   └── eval_results.json # JSON file with evaluation results
└── README.md             # Project documentation
```

## Example Results

Sample evaluation results on WikiText-2:

```json
{
    "microsoft/phi-2": {
        "perplexity": 7.245,
        "tokens_evaluated": 245673
    },
    "google/gemma-2b": {
        "perplexity": 8.123,
        "tokens_evaluated": 245673
    }
}
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lm_perplexity_eval,
    title = {Language Model Perplexity Evaluation},
    author = {Your Name},
    year = {2024},
    url = {https://github.com/yourusername/experimento5-modelado}
}
```