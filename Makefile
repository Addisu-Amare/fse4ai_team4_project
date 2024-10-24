# Define variables
IMAGE_NAME = my-python-app
CONTAINER_NAME = my-python-container

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
run:
	docker run --name $(CONTAINER_NAME) -d $(IMAGE_NAME)

# Stop the running container
stop:
	docker stop $(CONTAINER_NAME)

# Remove the stopped container
rm:
	docker rm $(CONTAINER_NAME)

# Clean up the image and container
clean: stop rm
	docker rmi $(IMAGE_NAME)

# Remove all stopped containers and unused images
prune:
	docker system prune -f

.PHONY: build run stop rm clean prune