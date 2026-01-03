FROM python:3.11-slim

# system deps for common packages; adjust if you need more
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy dependency files first for layer caching
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
# install uvicorn with websocket support
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY . /app

# create a folder to store uploaded materials
RUN mkdir -p /app/materials

EXPOSE 8000

# run uvicorn; remove --reload for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


