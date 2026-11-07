# Crux | Text & Dialogue Summarizer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Crux** is a full-stack, transformer-powered text and dialogue summarization web application. Built on top of **Hugging Face's T5 (Text-to-Text Transfer Transformer)** and **FastAPI**, it provides fast, abstractive summarization through a clean, modern web interface and an open REST API.

---

## Features

- **Abstractive Summarization:** Uses Google's T5 architecture fine-tuned with beam search (`num_beams=4`, `length_penalty=2.0`, `no_repeat_ngram_size=3`) to generate coherent, fluent summaries.
- **Hardware Acceleration:** Automatically detects and utilizes **Apple Silicon GPU (`mps`)**, NVIDIA GPU (`cuda`), or gracefully falls back to **CPU**.
- **Non-blocking Concurrency:** Machine learning inference is offloaded to worker threadpools, keeping the FastAPI event loop fully responsive for concurrent incoming requests.
- **Optimized Inference:** Employs PyTorch's `@torch.inference_mode()` and dynamic tokenization to eliminate gradient overhead and unnecessary tensor padding.
- **Responsive UI:**
  - Live real-time character and word counters.
  - One-click "Copy Summary" button with visual feedback.
  - Keyboard shortcut support (`Ctrl/Cmd + Enter` to submit).
  - Clean animated loading state and styled error alerts.
- **Automatic Model Fallback:** Seamlessly detects local model weights in `./saved_summary_model` or falls back to Hugging Face Hub (`t5-small`).
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

## Model Weights Notice

Because model weight files (`*.safetensors`, `*.bin`) exceed GitHub's 100 MB upload limit, heavy weights are excluded via `.gitignore`. 

- **Local Running:** If you already have weights inside `./saved_summary_model/model.safetensors`, the app detects and loads them locally.
- **First-time Clone:** If no local weight file is found, the application automatically downloads the pre-trained `t5-small` weights from Hugging Face on the first run.

---

## License

This project is licensed under the [MIT License](LICENSE).
