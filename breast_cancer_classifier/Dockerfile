# Base image for Python
FROM python:3.8-slim AS base

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY preprocessing.py .
COPY training.py .
COPY postprocessing.py .
COPY app.py .
COPY breast_cancer_detector.pickle .   
COPY templates/index.html ./templates/index.html

# Preprocessing stage
FROM base AS preprocessing
RUN python preprocessing.py 

# Training stage
FROM base AS training
RUN python training.py 

# Postprocessing stage
FROM base AS postprocessing
RUN python postprocessing.py 

# Final stage for running the Flask app
FROM base AS final

# Ensure Flask app runs on all interfaces (0.0.0.0)
CMD ["python", "-u", "app.py"]