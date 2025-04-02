#!/bin/bash

# This script helps to deploy the Deepfake Detection application

# Build the Docker image
echo "Building Docker image..."
docker build -t deepfake-detector:latest .

# Run the container
echo "Starting the container..."
docker run -d --name deepfake-detector -p 5000:5000 deepfake-detector:latest

echo "Application is running at http://localhost:5000"
echo "Use Ctrl+C to stop the application logs"

# Show logs
docker logs -f deepfake-detector 