import os
import re
import gc
import threading
import torch
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Initialize the FastAPI app
app = FastAPI(title="Crux", description="Text Summarization using T5", version="1.0")

# Device definition (Apple Silicon MPS -> CUDA -> CPU fallback)
if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    # Limit worker threads on CPU to avoid thread contention on cloud containers
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))

# Model & generation settings configurable via environment variables
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH", "100"))
MIN_SUMMARY_LENGTH = int(os.getenv("MIN_SUMMARY_LENGTH", "10"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "1"))  # 1 = greedy decoding for fast CPU inference

# Model source selection: local directory if weights exist, otherwise Hugging Face model hub
LOCAL_MODEL_DIR = os.getenv("MODEL_PATH", "./saved_summary_model")
has_local_weights = os.path.exists(LOCAL_MODEL_DIR) and (
    os.path.exists(os.path.join(LOCAL_MODEL_DIR, "model.safetensors"))
    or os.path.exists(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin"))
)
# Uses local fine-tuned weights if present, otherwise downloads from our Hugging Face model repo
HF_MODEL_REPO = "fireside4137/Text-Summarizer"
MODEL_SOURCE = LOCAL_MODEL_DIR if has_local_weights else HF_MODEL_REPO

# Lazy-loaded model references to ensure instant boot and low memory footprint (<50MB on startup)
_model = None
_tokenizer = None
_model_lock = threading.Lock()

def get_model_and_tokenizer():
    """
    Thread-safe lazy loader for model and tokenizer.
    Loads using bfloat16 to halve memory consumption (~120MB instead of 242MB),
    preventing Out-of-Memory crashes on 512MB RAM cloud tiers.
    """
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    with _model_lock:
        if _model is None:
            gc.collect()
            tok = T5Tokenizer.from_pretrained(MODEL_SOURCE)

            # Try loading in bfloat16 to halve memory; fallback to standard float32 if unsupported
            try:
                mod = T5ForConditionalGeneration.from_pretrained(
                    MODEL_SOURCE,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                )
            except Exception:
                mod = T5ForConditionalGeneration.from_pretrained(
                    MODEL_SOURCE,
                    low_cpu_mem_usage=True
                )

            mod.to(device)
            mod.eval()
            _tokenizer = tok
            _model = mod
            gc.collect()

    return _model, _tokenizer

# Templating - to tell where the html file is present
templates = Jinja2Templates(directory=".")

# Input schema for dialogue with validation
class DialogueInput(BaseModel):
    dialogue: str = Field(
        ...,
        min_length=10,
        description="Text or dialogue content to summarize (minimum 10 characters)."
    )

# Clean and normalize input text
def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)    # Normalize newlines
    text = re.sub(r"\s+", " ", text)     # Collapse multiple spaces
    text = re.sub(r"<.*?>", " ", text)   # Strip HTML tags
    text = text.strip()
    return text

# Summarization function
@torch.inference_mode()
def summarize_dialogue(dialogue: str) -> str:
    cleaned = clean_data(dialogue)
    if not cleaned:
        raise ValueError("Input text cannot be empty or solely whitespace/tags.")

    model_inst, tokenizer_inst = get_model_and_tokenizer()

    # T5 models expect the task prefix 'summarize: ' for summarization
    prompt = f"summarize: {cleaned}"

    # Tokenize without full-length padding for fast single-sample inference
    inputs = tokenizer_inst(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate summary with fast greedy decoding (or configurable beam search)
    targets = model_inst.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_SUMMARY_LENGTH,
        min_length=MIN_SUMMARY_LENGTH,
        num_beams=NUM_BEAMS,
        do_sample=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True if NUM_BEAMS > 1 else False
    )

    # Decode generated token IDs to text
    summary = tokenizer_inst.decode(targets[0], skip_special_tokens=True)
    return summary


# API Endpoints

@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):
    """
    Synchronous route handler so FastAPI automatically offloads
    CPU/GPU-bound model inference to a worker threadpool,
    preventing asyncio event-loop blocking.
    """
    try:
        summary = summarize_dialogue(dialogue_input.dialogue)
        return {"summary": summary}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

