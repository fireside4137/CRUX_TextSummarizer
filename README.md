# Crux | Text & Dialogue Summarizer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-fireside4137%2FText--Summarizer-yellow)](https://huggingface.co/fireside4137/Text-Summarizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Crux** is a full-stack, transformer-powered text and dialogue summarization web application. Built on top of **Hugging Face's T5 (Text-to-Text Transfer Transformer)** and **FastAPI**, it provides fast, abstractive summarization through a clean, modern web interface and an open REST API.

---

## Features

- **Abstractive Summarization:** Uses Google's T5 architecture fine-tuned for dialogue and text summarization, with fast greedy decoding (`num_beams=1`, `repetition_penalty=1.2`, `no_repeat_ngram_size=3`) for rapid CPU inference and configurable beam search support.
- **Hardware Acceleration:** Automatically detects and utilizes **Apple Silicon GPU (`mps`)**, NVIDIA GPU (`cuda`), or gracefully falls back to **CPU**.
- **Non-blocking Concurrency:** Machine learning inference is offloaded to worker threadpools, keeping the FastAPI event loop fully responsive for concurrent incoming requests.
- **Optimized Inference & Memory:** Employs PyTorch's `@torch.inference_mode()`, `torch.bfloat16`, dynamic tokenization, and lazy model loading to eliminate memory bloat and fit within strict cloud limits.
- **Dual Model Serving:** Automatically uses local fine-tuned weights (`./saved_summary_model`) if present on disk, or pulls the fine-tuned model directly from Hugging Face Hub ([`fireside4137/Text-Summarizer`](https://huggingface.co/fireside4137/Text-Summarizer)) on cloud instances.
- **Responsive UI:**
  - Live real-time character and word counters.
  - One-click "Copy Summary" button with visual feedback.
  - Keyboard shortcut support (`Ctrl/Cmd + Enter` to submit).
  - Clean animated loading state and styled error alerts.
- **Interactive API Documentation:** Auto-generated Swagger UI and OpenAPI documentation at `/docs`.

---

## Tech Stack

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic v2](https://docs.pydantic.dev/)
- **Machine Learning:** [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (Jinja2 Templates)

---

## Project Structure

```text
TextSummarizerApp/
├── app.py                   # FastAPI backend & inference pipeline
├── index.html               # Frontend interface (Jinja2 template)
├── requirements.txt         # Project dependencies
├── saved_summary_model/     # Model configuration & tokenizer configs
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── .env.example             # Environment configuration template
├── .gitignore               # Git rules (excludes heavy model weights & pycache)
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/fireside4137/TextSummarizerApp.git
cd TextSummarizerApp
```

### 2. Create and Activate a Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Once running, navigate to:
- **Web App:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## API Reference

### 1. Summarize Text

- **Endpoint:** `POST /summarize/`
- **Content-Type:** `application/json`

**Request Body:**
```json
{
  "dialogue": "Hannah: Hey, are you coming to the team lunch today? Mark: Yes, I will be there by 1 PM. Did someone book the table? Hannah: Yes, Sarah booked a table at the Italian place downtown."
}
```

**Response (200 OK):**
```json
{
  "summary": "hannah and mark are attending the team lunch at 1 pm. sarah booked a table at the italian place downtown."
}
```

### 2. Service Health

- **Endpoint:** `GET /health`

**Response (200 OK):**
```json
{
  "status": "healthy",
  "device": "mps"
}
```

---

## Model Deployment Strategy: Local vs. Hugging Face Hub

Because model weight files (`*.safetensors`, `*.bin`) exceed GitHub's 100 MB upload limit, heavy weights are excluded from Git via `.gitignore`. The architecture is decoupled to support two modes:

1. **Local Fine-Tuned Weights (`./saved_summary_model`):**
   - When developing locally on your machine, the application directly detects and loads `model.safetensors` from the local directory.
   - Zero internet bandwidth required; loads instantaneously on startup.

2. **Cloud Hub Weights ([`fireside4137/Text-Summarizer`](https://huggingface.co/fireside4137/Text-Summarizer)):**
   - For cloud platforms (like Render) or first-time cloning on external machines where the local weights file is absent, the app automatically streams your fine-tuned weights from the **Hugging Face Model Hub**.
   - Leverages `torch.bfloat16` and lazy loading to fit comfortably within 512MB RAM server tiers without triggering Out-of-Memory (OOM) crashes.

---

## Configuration

You can customize generation speed, lengths, and model paths via environment variables or a `.env` file:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NUM_BEAMS` | `1` | `1` for fast greedy search (~6–8s on CPU), `2`–`4` for beam search |
| `MAX_SUMMARY_LENGTH` | `100` | Maximum token length for the summary |
| `MIN_SUMMARY_LENGTH` | `10` | Minimum token length for the summary |
| `MODEL_PATH` | `./saved_summary_model` | Local directory or Hugging Face model identifier |
| `PORT` | `8000` | Port for the web server |

---

## License

This project is licensed under the [MIT License](LICENSE).
