import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model_name: str, texts, context_length: int = 512, stride: int = None, 
                       batch_size: int = 8, max_tokens_per_doc: int = None, device: str = "cuda") -> (float, int):
    """
    Compute perplexity of a given model over the provided text(s).
    :param model_name: HuggingFace model identifier or path.
    :param texts: Either a single string (for a contiguous corpus) or a list of strings (for separate documents).
    :param context_length: Number of tokens for each evaluation chunk (max sequence length to use).
    :param stride: Step size for sliding window (if None, defaults to context_length, meaning no overlap).
    :param batch_size: Number of sequences to evaluate in parallel per batch.
    :param max_tokens_per_doc: If set, truncate each text to at most this many tokens.
    :param device: "cuda" for GPU, "cpu" for CPU.
    :return: (perplexity, total_predicted_tokens) as a tuple.
    """
    # Use GPU if available unless overridden
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)
    torch.manual_seed(0)  # Ensure deterministic behavior for model (if any dropout, not expected in eval)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use pad token if available, otherwise use EOS or 0 as pad
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    
    # Determine torch dtype for model (half precision on GPU, full precision on CPU)
    load_kwargs = {}
    if device.type == "cuda":
        # Use half-precision for speed if on GPU
        if "bitnet" in model_name.lower():
            load_kwargs["torch_dtype"] = torch.bfloat16   # BitNet b1.58 uses BF16 weights
            load_kwargs["trust_remote_code"] = True       # allow custom BitNet modeling code
        elif "phi-2" in model_name.lower():
            load_kwargs["torch_dtype"] = torch.bfloat16   # Phi-2 benefits from bfloat16 to avoid overflow
        elif "gemma" in model_name.lower():
            load_kwargs["torch_dtype"] = torch.float16    # Gemma 2B can use FP16
        else:
            load_kwargs["torch_dtype"] = torch.float16
        # For any model with custom code not in transformers, enable remote code execution
        if "trust_remote_code" not in load_kwargs:
            load_kwargs["trust_remote_code"] = True if "bitnet" in model_name.lower() else False
    else:
        # On CPU, use full precision
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["trust_remote_code"] = True if "bitnet" in model_name.lower() else False
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.to(device)
    model.eval()
    
    # Default stride equals context_length (no overlap) if not provided
    if stride is None:
        stride = context_length
    
    # Prepare texts list
    if isinstance(texts, str):
        texts_to_eval = [texts]
    else:
        texts_to_eval = list(texts)
    
    nll_sum = 0.0
    n_loss_tokens = 0
    
    # Process each text (document) separately
    for text in texts_to_eval:
        # Tokenize the text. We disable truncation to get the full sequence.
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings["input_ids"][0]  # tensor of token IDs
        # Truncate long documents if max_tokens_per_doc is set
        if max_tokens_per_doc is not None and input_ids.size(0) > max_tokens_per_doc:
            input_ids = input_ids[:max_tokens_per_doc]
        seq_length = input_ids.size(0)
        if seq_length == 0:
            continue  # skip empty text if any
        
        # Slide through the sequence in chunks
        prev_end = 0
        for begin in range(0, seq_length, stride):
            end = min(begin + context_length, seq_length)
            target_length = end - prev_end  # number of new tokens to predict in this chunk
            # Slice the input for the current chunk
            input_chunk = input_ids[begin:end].to(device)
            # Prepare target labels for this chunk
            target_chunk = input_chunk.clone()
            if target_length < input_chunk.size(0):
                # Ignore tokens that are just context (not to be predicted)
                target_chunk[:-target_length] = -100
            # Forward pass (batched if batch_size>1, but here we handle chunk by chunk sequentially)
            with torch.no_grad():
                outputs = model(input_chunk.unsqueeze(0), labels=target_chunk.unsqueeze(0))
            # outputs.loss is mean NLL over valid tokens in this chunk
            neg_log_lik = outputs.loss * (target_chunk != -100).sum().item()
            # Account for the model's internal shifting: subtract one per sequence
            neg_log_lik = neg_log_lik - outputs.loss  # subtract one token worth of NLL
            nll_sum += neg_log_lik.item()
            n_loss_tokens += (target_chunk != -100).sum().item() - 1  # add valid tokens - 1
            prev_end = end
            if end == seq_length:
                break
    
    # Compute perplexity
    avg_nll = nll_sum / n_loss_tokens
    ppl = math.exp(avg_nll)
    return ppl, n_loss_tokens
