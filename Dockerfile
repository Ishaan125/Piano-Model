FROM python:3.11-slim

WORKDIR /app

# Copy only runtime requirements and the application files required for inference
COPY requirements.txt /app/requirements.txt

# Optional args: pass a direct wheel URL via TORCH_WHEEL, or set TORCH_VERSION (e.g. 2.2.0+cpu)
ARG TORCH_WHEEL=""
ARG TORCH_VERSION="2.2.0+cpu"
RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "$TORCH_WHEEL" ]; then \
        echo "Installing torch from wheel: $TORCH_WHEEL"; \
        python -m pip install --no-cache-dir "$TORCH_WHEEL"; \
    elif [ -n "$TORCH_VERSION" ]; then \
        echo "Installing torch==$TORCH_VERSION from PyTorch CPU index"; \
        python -m pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Installing latest torch from PyTorch CPU index"; \
        python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy only inference files (avoid copying training scripts)
COPY api.py /app/api.py
COPY load_model.py /app/load_model.py
COPY model.py /app/model.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Run uvicorn in the container
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
