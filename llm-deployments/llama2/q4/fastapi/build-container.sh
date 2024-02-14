#!/bin/sh

export DOCKER_IMAGE_NAME="myapp"
export DOCKERHUB_REPO_NAME="myapp-repo"

# Check for required environment variables
if [ -z "$DOCKER_IMAGE_NAME" ] || [ -z "$DOCKERHUB_REPO_NAME" ]; then
  echo "One or more required environment variables are missing."
  echo "Make sure DOCKER_IMAGE_NAME and DOCKERHUB_REPO_NAME are set."
  exit 1
fi

# Prompt for Docker Hub credentials
echo "Enter your Docker Hub username:"
read DOCKERHUB_USERNAME
echo "Enter your Docker Hub password:"
read -s DOCKERHUB_PASSWORD

# Login to Docker Hub
echo $DOCKERHUB_PASSWORD | docker login --username $DOCKERHUB_USERNAME --password-stdin
if [ $? -ne 0 ]; then
  echo "Docker login failed"
  exit 1
fi

# Build the Docker image
docker build -t $DOCKER_IMAGE_NAME .

# Tag the Docker image for Docker Hub
docker tag $DOCKER_IMAGE_NAME $DOCKERHUB_USERNAME/$DOCKERHUB_REPO_NAME

# Push the Docker image to Docker Hub
docker push $DOCKERHUB_USERNAME/$DOCKERHUB_REPO_NAME

# Logout from Docker Hub
docker logout

echo "Docker image pushed successfully to Docker Hub."
