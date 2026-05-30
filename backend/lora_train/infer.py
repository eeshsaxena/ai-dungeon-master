"""
LoRA inference wrapper (Phase 8).

Lazy-loads the fine-tuned model + LoRA adapter on first use and keeps it in
memory for subsequent calls. Used by narrator.py when LLM_PROVIDER=lora.

The pipeline runs in a thread-pool executor (same pattern as image_gen.py
for SD) so the FastAPI event loop stays free during inference.
"""
import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

_LORA_BASE_MODEL   = os.getenv("LORA_BASE_MODEL",   "microsoft/phi-3-mini-4k-instruct")
_LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", str(Path(__file__).parent / "adapter"))

_pipe: Any = None
_pipe_lock = threading.Lock()
_pipe_error: Optional[str] = None


def _load_pipeline() -> None:
    """Load model + adapter. Called once from a background thread at startup."""
    global _pipe, _pipe_error
    adapter = Path(_LORA_ADAPTER_PATH)
    if not adapter.exists():
        _pipe_error = f"Adapter not found at {adapter}. Run lora_train/train.py first."
        print(f"[LoRA] {_pipe_error}")
        return
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"[LoRA] Loading base model: {_LORA_BASE_MODEL} on {device}…")
        tokenizer = AutoTokenizer.from_pretrained(_LORA_ADAPTER_PATH, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            _LORA_BASE_MODEL,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        print(f"[LoRA] Applying adapter: {adapter}")
        model = PeftModel.from_pretrained(base, str(adapter))
        model.eval()

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=350,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
        with _pipe_lock:
            _pipe = pipe
        if device == "cuda":
            import torch
            print(f"[LoRA] Ready on {torch.cuda.get_device_name(0)}")
        else:
            print("[LoRA] Ready on CPU (inference will be slow)")
    except Exception as exc:
        with _pipe_lock:
            _pipe_error = str(exc)
        print(f"[LoRA] Failed to load: {exc}")


def warmup_lora() -> None:
    """Start model loading in a background thread (called at server startup)."""
    if os.getenv("LLM_PROVIDER", "ollama") != "lora":
        return
    t = threading.Thread(target=_load_pipeline, daemon=True)
    t.start()


def _get_pipeline():
    """Block until the pipeline is ready (max 5 min), then return it."""
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        if _pipe_error is not None:
            return None
    import time
    for _ in range(300):
        time.sleep(1)
        with _pipe_lock:
            if _pipe is not None:
                return _pipe
            if _pipe_error is not None:
                return None
    return None


async def generate(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Run LoRA inference async (thread-pool). messages is a list of
    {'role': ..., 'content': ...} dicts (system + user + optional history).
    Returns the assistant reply text, or None on failure.
    """
    def _run() -> Optional[str]:
        pipe = _get_pipeline()
        if pipe is None:
            return None
        try:
            out = pipe(messages)
            # transformers pipeline returns list of {generated_text: [...]}
            generated = out[0]["generated_text"]
            # The pipeline appends the new assistant turn to the list
            if isinstance(generated, list):
                last = generated[-1]
                if isinstance(last, dict):
                    return last.get("content", "")
            return str(generated)
        except Exception as exc:
            print(f"[LoRA] Inference error: {exc}")
            return None

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


def stats() -> Dict[str, Any]:
    return {
        "provider": "lora",
        "base_model": _LORA_BASE_MODEL,
        "adapter_path": _LORA_ADAPTER_PATH,
        "adapter_exists": Path(_LORA_ADAPTER_PATH).exists(),
        "ready": _pipe is not None,
        "loading": _pipe is None and _pipe_error is None,
        "error": _pipe_error,
    }
