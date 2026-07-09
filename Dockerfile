FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "pytest", "-q"]
