"""
MicroGuard — Lightweight RAG Faithfulness Classifier

Usage:
    from microguard import MicroGuard
    guard = MicroGuard(model="gemma-270m")
    result = guard.check(context="...", question="...", answer="...")
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Model registry — maps user-friendly names to HuggingFace paths
MODEL_REGISTRY = {
    "gemma-270m": {
        "base": "google/gemma-3-270m-it",
        "adapter": "MicroGuard/gemma-270m-faithfulness",  # HF Hub path (update after upload)
    },
    "qwen-0.5b": {
        "base": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter": "MicroGuard/qwen-0.5b-faithfulness",
    },
    "gemma-1b": {
        "base": "google/gemma-3-1b-it",
        "adapter": "MicroGuard/gemma-1b-faithfulness",
    },
}

SYSTEM_PROMPT = "You are a faithfulness evaluator for RAG systems. You must respond with exactly one word."
USER_TEMPLATE = """Context: {context}
Question: {query}
Answer: {answer}

Is every claim in the answer fully supported by the context? Respond with exactly one word: FAITHFUL or UNFAITHFUL."""


class MicroGuard:
    """Lightweight RAG faithfulness classifier.

    Args:
        model: Model name ("gemma-270m", "qwen-0.5b", "gemma-1b")
               or path to a local adapter directory.
        base_model: Optional base model override (for local adapters).
        device: Device to use ("auto", "cuda", "mps", "cpu").
        max_length: Maximum sequence length for tokenization.

    Example:
        >>> guard = MicroGuard(model="gemma-270m")
        >>> result = guard.check(
        ...     context="Paris is the capital of France.",
        ...     question="What is the capital of France?",
        ...     answer="The capital of France is Paris."
        ... )
        >>> print(result['verdict'])
        'FAITHFUL'
    """

    def __init__(self, model="gemma-270m", base_model=None, device="auto", max_length=512):
        self.max_length = max_length

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load model
        if model in MODEL_REGISTRY:
            config = MODEL_REGISTRY[model]
            base_name = config["base"]
            adapter_path = config["adapter"]
        else:
            # Assume it's a local path
            if base_model is None:
                raise ValueError(
                    f"Unknown model '{model}'. Use one of {list(MODEL_REGISTRY.keys())} "
                    f"or provide base_model= for a local adapter path."
                )
            base_name = base_model
            adapter_path = model

        self.tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype, trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Cache token IDs for constrained decoding
        self.faithful_ids = self.tokenizer.encode("FAITHFUL", add_special_tokens=False)
        self.unfaithful_ids = self.tokenizer.encode("UNFAITHFUL", add_special_tokens=False)

        self.model_name = model

    def check(self, context, answer, question=""):
        """Check if an answer is faithful to the context.

        Args:
            context: The retrieved document/passage.
            answer: The generated answer to evaluate.
            question: The user's original question (optional).

        Returns:
            dict with keys:
                - verdict: "FAITHFUL" or "UNFAITHFUL"
                - confidence: float (0-100)
                - faithful_score: raw logit for FAITHFUL
                - unfaithful_score: raw logit for UNFAITHFUL
                - latency_ms: inference time in milliseconds
        """
        # Truncate inputs
        context = context[:900]
        question = (question or "N/A")[:200]
        answer = answer[:400]

        # Format prompt
        msg = USER_TEMPLATE.format(context=context, query=question, answer=answer)
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + msg}]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"<|im_start|>user\n{SYSTEM_PROMPT}\n\n{msg}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Constrained decoding
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            f_score = logits[0, self.faithful_ids[0]].item()
            u_score = logits[0, self.unfaithful_ids[0]].item()
        latency_ms = (time.time() - start_time) * 1000

        # Compute confidence via softmax
        scores = torch.tensor([f_score, u_score])
        probs = torch.nn.functional.softmax(scores, dim=0)
        confidence = probs.max().item() * 100

        verdict = "FAITHFUL" if f_score > u_score else "UNFAITHFUL"

        return {
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "faithful_score": round(f_score, 4),
            "unfaithful_score": round(u_score, 4),
            "latency_ms": round(latency_ms, 1),
        }

    def check_batch(self, examples, show_progress=True):
        """Check faithfulness for a batch of examples.

        Args:
            examples: List of dicts with keys 'context', 'answer', and optionally 'question'.
            show_progress: Print progress updates.

        Returns:
            List of result dicts (same format as check()).
        """
        results = []
        for i, ex in enumerate(examples):
            result = self.check(
                context=ex.get("context", ""),
                answer=ex.get("answer", ex.get("response", "")),
                question=ex.get("question", ex.get("query", "")),
            )
            results.append(result)
            if show_progress and (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(examples)}...")
        return results

    def __repr__(self):
        return f"MicroGuard(model='{self.model_name}', device='{self.device}')"
