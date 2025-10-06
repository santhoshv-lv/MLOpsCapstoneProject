# Use the official Python slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV DATA_DIR=/root/airflow/data
WORKDIR $APP_HOME

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . $APP_HOME


# Expose FastAPI port
EXPOSE 8090

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
