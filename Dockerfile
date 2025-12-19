# -----------------------------
# 1. Use official Python image
# -----------------------------
FROM python:3.12

# -----------------------------
# 2. Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. Copy dependency files
# -----------------------------
COPY requirements.txt .

# -----------------------------
# 4. Install dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 5. Copy all backend files
# -----------------------------
COPY . .

# -----------------------------
# 6. Expose FastAPI port
# -----------------------------
EXPOSE 7860

# -----------------------------
# 7. Start FastAPI app
# -----------------------------
# Uses `uvicorn` to run your app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
