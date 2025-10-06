# Use the official Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY . $APP_HOME

# Define the command to run the application (uvicorn)
# Ensure the host/port match your setup. 0.0.0.0 makes it accessible outside the container.
