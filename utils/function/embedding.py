# embedding.py
import os
from typing import Iterable, List, Union

from dotenv import load_dotenv

# Embeddings
from sentence_transformers import SentenceTransformer

# Chat LLM (streaming)
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

load_dotenv()

# ---------- Config ----------
HF_TOKEN = os.getenv("HF_TOKEN") or None
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHAT_MODEL_NAME = os.getenv(
    "HF_CHAT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"  # small, fast default
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

_embedding_model: SentenceTransformer = None
_tokenizer: AutoTokenizer = None
_chat_model: AutoModelForCausalLM = None

def _ensure_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

def _ensure_chat_model():
    global _tokenizer, _chat_model
    if _tokenizer is None or _chat_model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            CHAT_MODEL_NAME, use_auth_token=HF_TOKEN
        )
        _chat_model = AutoModelForCausalLM.from_pretrained(
            CHAT_MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto" if DEVICE == "cuda" else None,
            use_auth_token=HF_TOKEN,
        )
        if DEVICE == "cpu":
            _chat_model.to(DEVICE)

def get_embedding(texts: Union[str, List[str]]) -> List[List[float]]:
    """
    Returns embeddings for a single string or list of strings.
    Compatible shape with your previous function: List[List[float]].
    """
    _ensure_embedding_model()
    if isinstance(texts, str):
        texts = [texts]
    # Normalize for cosine similarity by default (common for ST models)
    vecs = _embedding_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return [v.tolist() for v in vecs]

def _build_prompt(system_message: str, user_message: str) -> dict:
    """
    Builds a chat prompt using the tokenizer's chat template if available.
    Fallbacks to a simple prompt string for models without templates.
    """
    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": user_message}]
    # If the tokenizer has a chat template, return tokenized inputs for generation.
    try:
        _ensure_chat_model()
        chat_text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"input_ids": _tokenizer(chat_text, return_tensors="pt").input_ids.to(_chat_model.device)}
    except Exception:
        # Fallback basic prompt
        prompt = f"<<SYSTEM>>\n{system_message}\n\n<<USER>>\n{user_message}\n\n<<ASSISTANT>>"
        _ensure_chat_model()
        return {"input_ids": _tokenizer(prompt, return_tensors="pt").input_ids.to(_chat_model.device)}


def get_model_stream(system_message: str, user_message: str) -> Iterable[str]:
    """
    Streams generated text chunks from a local HF model.
    Usage:
        for chunk in get_model_stream(sys_msg, usr_msg):
            print(chunk, end="", flush=True)
    """
    _ensure_chat_model()
    inputs = _build_prompt(system_message, user_message)

    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        eos_token_id=_tokenizer.eos_token_id,
    )

    # Generate in a background thread so we can iterate tokens
    import threading
    thread = threading.Thread(target=_chat_model.generate, kwargs=gen_kwargs)
    thread.start()

    # Yield tokens as they arrive
    for token in streamer:
        yield token
