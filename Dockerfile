# ponytail: torch + SGNet checkpoints are NOT installed here — without them the
# hybrid pipeline degrades to the classical branch (deep_available=False).
# To enable the deep branch: add torch (CPU wheel) + copy checkpoints/sgnet/*.pth
# and set SGNET_CKPT_DIR.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.web.txt .
RUN pip install -r requirements.web.txt

COPY . .

EXPOSE 8080
CMD ["python", "app.py"]
