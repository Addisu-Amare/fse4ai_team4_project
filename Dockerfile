# Base image for Python
FROM python:3.8-slim AS base

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY ./breast_cancer_classifier/preprocessing.py .
COPY ./breast_cancer_classifier/training.py .
COPY ./breast_cancer_classifier/postprocessing.py .
COPY ./breast_cancer_classifier/data ./data
COPY app.py .
COPY templates/ ./templates/
COPY test_app.py .

# Preprocessing stage
FROM base AS preprocessing
CMD ["python", "-u", "preprocessing.py"]

# Training stage
FROM base AS training
CMD ["python", "-u", "training.py"]

# Postprocessing stage
FROM base AS postprocessing
CMD ["python", "-u", "postprocessing.py"]

# Test final Flask app
FROM base AS testing
COPY breast_cancer_detector.pickle .   
CMD ["python", "-u", "test_app.py"]

# Final stage for running the Flask app
FROM base AS final
COPY breast_cancer_detector.pickle .   
CMD ["python", "-u", "app.py"]
