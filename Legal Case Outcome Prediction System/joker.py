from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from fastapi.responses import JSONResponse
import torch
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

# 🔁 Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# 🧠 Init preprocessing tools
ps = PorterStemmer()
lem = WordNetLemmatizer()
lst_stopwords = stopwords.words("english")

# ✅ Preprocessing function
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    lst_text = text.split()
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]
    if flg_stemm:
        lst_text = [ps.stem(word) for word in lst_text]
    if flg_lemm:
        lst_text = [lem.lemmatize(word) for word in lst_text]
    return " ".join(lst_text)

# ⚖️ FastAPI App
app = FastAPI(title="LegalBERT Case Predictor 🧑‍⚖️")

# 🔍 Load classification model & tokenizer
model_path = "./legalbert_model/legalbert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔮 Load text generation model for suggestions
suggestions_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1  # use -1 if you're on CPU
)

# 📝 Input schema
class InputText(BaseModel):
    facts: str

# 📌 Predict endpoint with suggestions
@app.post("/predict/")
async def predict(input: InputText):
    # ✅ Step 1: Preprocess input
    clean_text = utils_preprocess_text(
        input.facts,
        flg_stemm=False,
        flg_lemm=True,
        lst_stopwords=lst_stopwords
    )

    # ✅ Step 2: Tokenize and Predict
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = round(probs[0][pred_class].item() * 100, 2)

    # ✅ Step 3: Generate Legal Suggestions
    prompt = f"""You are an expert Indian legal advisor. Carefully study the case facts below, and suggest **three concrete legal actions** the First Party should take to improve their chances of winning in court. Be specific and legally sound.

Case Facts:
{input.facts}

Suggestions:"""

    result = suggestions_generator(
        prompt,
        max_length=256,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=2.2,
        num_return_sequences=1
    )

    # Extract and return suggestions
    suggestions = result[0]['generated_text'].strip()

    # ✅ Step 4: Return full response
    return {
        "prediction": "First party WINS ✅" if pred_class == 1 else "First party LOSES ❌",
        "confidence": f"{confidence}%",
        "legal_suggestions": suggestions
    }
