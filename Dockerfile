FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across rebuilds
# unless requirements.txt actually changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train at build time so the image is self-contained — no need to mount a
# pre-trained model or run a separate training step before the API can serve.
RUN python src/train.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
