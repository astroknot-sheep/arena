FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy submission files
COPY predict.py grade.py ./
COPY model.pkl ./

# Copy geo data needed by predict.py at inference
COPY data/geo/ ./data/geo/

ENTRYPOINT ["python", "grade.py"]