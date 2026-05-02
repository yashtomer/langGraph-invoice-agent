FROM python:3.11-slim

# WeasyPrint native deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY templates ./templates

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["invoice-agent"]
