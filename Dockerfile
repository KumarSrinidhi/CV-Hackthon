FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home --uid 1000 appuser
COPY app app
COPY classical classical
COPY data data
COPY deep deep
COPY eval eval
COPY hybrid hybrid
COPY main.py .
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:7860/health', timeout=3).close()"]
CMD ["python", "main.py", "--host", "0.0.0.0"]
