import os
import re
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

# Model & tokenizer loading: use local weights if present, otherwise download from Hugging Face
LOCAL_MODEL_DIR = os.getenv("MODEL_PATH", "./saved_summary_model")
has_local_weights = os.path.exists(LOCAL_MODEL_DIR) and (
    os.path.exists(os.path.join(LOCAL_MODEL_DIR, "model.safetensors"))
    or os.path.exists(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin"))
)
MODEL_SOURCE = LOCAL_MODEL_DIR if has_local_weights else "t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_SOURCE)
model = T5ForConditionalGeneration.from_pretrained(MODEL_SOURCE, low_cpu_mem_usage=True)
model.to(device)
model.eval()

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

    # T5 models expect the task prefix 'summarize: ' for summarization
    prompt = f"summarize: {cleaned}"

    # Tokenize without full-length padding for fast single-sample inference
    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate summary with fast greedy decoding (or configurable beam search)
    targets = model.generate(
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
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
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
    return {"status": "healthy", "device": str(device)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
