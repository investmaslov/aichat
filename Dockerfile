FROM python:3.12-slim

WORKDIR /app

# Сначала зависимости — слой кешируется, пока requirements.txt не меняется.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
