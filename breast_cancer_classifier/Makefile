# Makefile for Breast Cancer Classification App

# Variables
IMAGE_NAME = breast-cancer-classifier
CONTAINER_NAME = breast_cancer_app

.PHONY: all build run clean preprocess train postprocess

# Default target
all: build run

# Build the Docker image and show output for all stages (default)
build:
	docker build -t $(IMAGE_NAME) .

# Build specific stages
preprocess:
	docker build --target preprocessing -t $(IMAGE_NAME) .

train:
	docker build --target training -t $(IMAGE_NAME) .

postprocess:
	docker build --target postprocessing -t $(IMAGE_NAME) .

# Run the Docker container and show output
run:
	docker run --rm -p 5000:5000 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# Clean up any stopped containers (optional)
clean:
	docker rm -f $(CONTAINER_NAME) || true