FROM python:3.11.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "./candidates.jsonl", "--out", "./submission.csv", "--tfidf-scores", "./data/tfidf_scores.json"]
