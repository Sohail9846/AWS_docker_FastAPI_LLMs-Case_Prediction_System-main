
FROM python:3.10-slim


WORKDIR /app


COPY . .


RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


RUN python -m nltk.downloader stopwords wordnet


EXPOSE 8000

# 🏁 Run the FastAPI app
CMD ["uvicorn", "joker:app", "--host", "0.0.0.0", "--port", "8000"]
