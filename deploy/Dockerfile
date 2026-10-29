# AcoustiGuard – Hugging Face Spaces (Docker, free CPU Basic: 2 vCPU, 16 GB RAM)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    ACOUSTIGUARD_DEVICE=cpu MPLCONFIGDIR=/tmp/mpl NUMBA_CACHE_DIR=/tmp/numba PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small (no CUDA libraries)
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --user torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements-deploy.txt

COPY --chown=user . .
RUN mkdir -p logs

EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
