# ──────────────────────────────────────────────
#  Dockerfile — SRE Incident Response OpenEnv
#  Designed for Hugging Face Spaces (port 7860)
# ──────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
