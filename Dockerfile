FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python -c "from torchvision import models; models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)"

COPY src/ ./src/
COPY weights/ ./weights/

ENTRYPOINT ["python", "src/predict.py"]
