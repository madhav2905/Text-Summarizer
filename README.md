# AI Text Summarizer
A full-stack NLP project that performs abstractive dialogue summarization using a fine-tuned T5 transformer model. The application is deployed locally with FastAPI and features a simple web interface for real-time summarization.

<img width="1512" height="908" alt="Screenshot 2026-04-03 at 12 11 37 AM" src="https://github.com/user-attachments/assets/bdfa9607-1d2f-4d9c-a595-6978c4ac6b21" />

## Features
* **Abstractive Summarization**: Generates human-like summaries instead of just extracting sentences.
* **Fine-Tuned Logic**: Optimized on the **SAMSum** dataset for high-quality chat processing.
* **FastAPI Backend**: High-performance Python API.
* **Minimalist UI**: Clean, responsive frontend with a built-in character counter and loading states.

## Tech Stack
* **Model**: Hugging Face `T5-Small`
* **Framework**: FastAPI
* **Frontend**: HTML, CSS, Jinja2

## Model Details

* **Model:** T5-small
* **Task:** Abstractive Dialogue Summarization
* **Dataset:** SAMSum
* **Framework:** HuggingFace Transformers

## Evaluation (ROUGE Scores)

| Metric     | Score |
| ---------- | ----- |
| ROUGE-1    | 0.49  |
| ROUGE-2    | 0.23  |
| ROUGE-L    | 0.40  |
| ROUGE-Lsum | 0.40  |

> These scores indicate strong lexical and structural similarity between generated and reference summaries.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/madhav2905/text-summarizer.git
cd text-summarizer
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
uvicorn app:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

## License

This project is for educational purposes.
