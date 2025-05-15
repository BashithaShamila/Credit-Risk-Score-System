FROM python:3.10-slim

WORKDIR /app

# Copy requirements.txt first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the current directory to /app
COPY . /app

# Make sure the model directory exists
RUN mkdir -p models

# Set the command to run the prediction script
CMD ["python", "predict_credit_score.py"]