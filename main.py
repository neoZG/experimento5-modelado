import argparse
import json
import numpy as np, random, torch
import time
from datetime import datetime
from datasets import load_dataset
from perplexity_eval import compute_perplexity
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of LLMs on WikiText-2 or C4.")
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], required=True,
                        help="Dataset to evaluate on: 'wikitext2' (WikiText-2 test set) or 'c4' (subset of C4 validation).")
    parser.add_argument("--models", nargs="+", default=[
                        "microsoft/bitnet-b1.58-2B-4T-bf16",
                        "EleutherAI/gpt-neo-2.7B",
                        "microsoft/phi-2",
                        "google/gemma-2b"],
                        help="HuggingFace model names or paths to evaluate. Space-separated list.")
    parser.add_argument("--context_length", type=int, default=512,
                        help="Context window length (tokens) for evaluation chunks.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for sliding window. Default (None) = no overlap (equal to context_length).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (number of sequences) for parallel inference on GPU.")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="For C4: number of examples to sample from validation set. If not set, defaults to 1000.")
    parser.add_argument("--max_tokens_per_doc", type=int, default=1024,
                        help="Truncate each document to at most this many tokens for evaluation (None = no limit).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device: 'cuda' for GPU (default), 'cpu' for CPU.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data shuffling (relevant for C4 sampling).")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save evaluation results.")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU instead.")
        args.device = "cpu"
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load and prepare dataset
    if args.dataset == "wikitext2":
        # Load WikiText-2 raw test split
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Concatenate all text lines into one string, separated by double newlines (preserves article boundaries)
        text_data = "\n\n".join(ds["text"])
    elif args.dataset == "c4":
        # Load C4 English validation set
        ds = load_dataset("c4", "en", split="validation", streaming=False)
        total_examples = len(ds)
        # Determine number of examples to use
        num_examples = args.max_examples if args.max_examples is not None else 1000
        num_examples = min(num_examples, total_examples)
        # Shuffle and select subset
        ds = ds.shuffle(seed=args.seed).select(range(num_examples))
        # Extract text field from each example
        text_data = [entry["text"] for entry in ds]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Prepare results dictionary
    results = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
        "configuration": {
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "max_tokens_per_doc": args.max_tokens_per_doc,
            "device": args.device
        },
        f"{args.dataset}_results": {}
    }
    
    # Run perplexity evaluation for each model
    for model_name in args.models:
        print(f"Evaluating model {model_name} on {args.dataset}...")
        start_time = time.time()
        ppl, token_count = compute_perplexity(model_name, text_data,
                                             context_length=args.context_length,
                                             stride=args.stride,
                                             batch_size=args.batch_size,
                                             max_tokens_per_doc=(None if args.max_tokens_per_doc == 0 else args.max_tokens_per_doc),
                                             device=args.device)
        eval_time = time.time() - start_time
        
        # Store results
        results[f"{args.dataset}_results"][model_name] = {
            "perplexity": float(ppl),
            "tokens_evaluated": token_count,
            "evaluation_time_seconds": round(eval_time, 2)
        }
        if args.dataset == "c4":
            results[f"{args.dataset}_results"][model_name]["num_examples"] = num_examples
        
        print(f"Perplexity of {model_name} on {args.dataset}: {ppl:.4f} "
              f"(evaluated on {token_count} tokens in {eval_time:.2f} seconds)")
    
    # Save results to JSON
    output_file = output_dir / "eval_results.json"
    # If file exists, merge with existing results
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
        # Update only the current dataset's results
        existing_results[f"{args.dataset}_results"] = results[f"{args.dataset}_results"]
        results = existing_results
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n=== Summary of results ===")
    for model_name, result in results[f"{args.dataset}_results"].items():
        print(f"{model_name}: PPL = {result['perplexity']:.4f} "
              f"(evaluated in {result['evaluation_time_seconds']:.2f}s)")

if __name__ == "__main__":
    main()
