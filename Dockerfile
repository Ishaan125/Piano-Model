FROM python:3.11-slim

WORKDIR /app

# Copy only runtime requirements and the application files required for inference
COPY requirements.txt /app/requirements.txt

# Optional torch wheel URL can be provided as build-arg if you want GPU-enabled image
ARG TORCH_WHEEL=""
RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "$TORCH_WHEEL" ]; then \
         python -m pip install "$TORCH_WHEEL"; \
    fi \
    && python -m pip install -r /app/requirements.txt

# Copy only inference files (avoid copying training scripts)
COPY api.py /app/api.py
COPY load_model.py /app/load_model.py
COPY model.py /app/model.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Run uvicorn in the container
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
