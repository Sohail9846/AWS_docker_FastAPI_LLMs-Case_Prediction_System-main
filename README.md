# Legal-Case-Outcome-Prediction-System
# ⚖️ Legal Case Outcome Predictor API

This project is a FastAPI-based backend service that uses a **fine-tuned LegalBERT model** to predict the outcome of Indian legal cases based on case facts.

---

## 🚀 What It Does

- Takes case facts as input
- Predicts whether the **First Party WINS or LOSES**
- Gives **confidence score**
- Generates 3 **legal suggestions** using `flan-t5-large`

---

## 🧠 Tech Stack

- **FastAPI** – for building the API
- **LegalBERT** – fine-tuned transformer for classification
- **FLAN-T5** – for generating legal suggestions
- **NLTK** – for text preprocessing
- **Docker** – containerization
- **AWS EC2** – deployment server

---

## 📦 Model

- Fine-tuned `nlpaueb/legal-bert-base-uncased`
- Achieved **92% accuracy** on the test set

---

## 🚢 Deployment

- Dockerized the FastAPI app
- Deployed it on an **AWS EC2 instance**

---

## 📬 Example Request

```json
POST /predict/
{
  "facts": "The petitioner was not given an opportunity to respond under Article 311..."
}
