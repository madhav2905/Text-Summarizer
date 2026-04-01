from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load model
MODEL_PATH = "models/t5-samsum-model"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Clean function
def clean_text(text):
    if not text:
        return ""
    text = text.replace("\r\n", " ")
    text = text.strip()
    return text

# Generate summary
def generate_summary(text):
    if not text or not text.strip():
        return "No text provided to summarize."
        
    text = clean_text(text)
    input_text = "summarize: " + text

    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt" 
    ).to(device)

    outputs = model.generate(
    inputs["input_ids"],
    max_length=128,
    min_length=10,     
    num_beams=4,
    length_penalty=1.5,
    early_stopping=True
)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)  

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"dialogue": "", "summary": ""}
    )

# API endpoint
@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, text: str = Form(...)):
    summary = generate_summary(text)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"dialogue": text, "summary": summary}
    )