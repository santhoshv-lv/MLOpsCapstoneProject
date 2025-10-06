 MLOPS Capstone Project: Retail Store & Customer Insights
This repository contains the complete, end-to-end MLOps pipeline for a retail chain. The goal is to automate the analysis of daily sales data, build a customer segmentation model (K-Means), and deploy actionable insights via a high-performance FastAPI service and a user-friendly Streamlit dashboard.

The project covers all phases, from processing daily sales data to supporting CI/CD workflows.

Project Architecture & Components
The system is built on a robust MLOps architecture:

Component	Tool / Technology	Purpose
Orchestration / ETL	Apache Airflow	Automates data ingestion (Kaggle API), cleaning, RFM calculation, and model training.
Model Serving Layer	FastAPI	Loads the assets (model, scaler, data) and serves real-time KPIs and predictions via high-performance APIs.
Visualization / Dashboard	Streamlit	Consumes the FastAPI endpoints to present user-friendly, interactive dashboards.
Machine Learning	scikit-learn (K-Means)	Builds the customer segmentation model.

Export to Sheets
Setup and Installation
Prerequisites
You must have the following installed and configured:

Python 3.10+ (in a Virtual Environment)

Apache Airflow (Installed in your airflow_venv)

Kaggle API (Installed via pip install kaggle and configured with credentials in your environment).

A. Environment Setup
Ensure all necessary dependencies are installed in your active virtual environment (airflow_venv):

Bash

# Activate your environment
source airflow_venv/bin/activate

# Install all required Python packages for the pipeline
pip install -r requirements.txt

# Install packages for the dashboard
pip install streamlit matplotlib seaborn
B. Directory Structure
Ensure your Airflow and Project directories are set up correctly:

Component	Host Path	Files
Airflow DAGs	~/airflow/dags/	retail_mlops_data_pipeline.py
ML Project	~/MLOpsCapstoneProject/	app.py, dashboard.py, requirements.txt
Shared Data	/root/airflow/data/	CRITICAL: Location for all assets (.csv, .joblib).

Export to Sheets
Running the End-to-End Pipeline
Step 1: Run the Automated Data Pipeline (Airflow)
This step automates the ETL and trains the model.

Set Environment Variables: Ensure your scheduler terminal has the necessary environment variables exported:

Bash

export AIRFLOW_HOME=~/airflow
export KAGGLE_USERNAME="..."
export KAGGLE_KEY="..."
Start Scheduler:

Bash

airflow scheduler
Trigger DAG:

Access the Airflow UI (http://localhost:8080).

Find the retail_mlops_data_pipeline DAG and manually trigger a new run.

Verification: Wait for all tasks to complete successfully. This process automatically downloads the data, performs RFM, trains the K-Means model, and saves the final assets.

Step 2: Run the API Serving Layer (FastAPI)
This step deploys the service that serves your insights and model predictions.

Start API: In a new terminal session, navigate to the project directory and run:

Bash

uvicorn app:app --reload --host 0.0.0.0 --port 8000
Verification: Access the documentation: http://<YOUR_IP_ADDRESS>:8000/docs

Test the endpoints for Store Performance, Top Customers, Sales Trends, and Customer Segment Prediction.

Step 3: Run the User Dashboard (Streamlit)
This launches the visualization layer for stakeholders.

Start Dashboard: In a third terminal session, navigate to the project directory and run:

Bash

streamlit run dashboard.py
Verification: Open the URL provided by Streamlit (usually http://localhost:8501) to see the live charts and test real-time customer segmentation.

CI/CD & MLOPS Practices
The CI/CD pipeline is defined in .github/workflows/ci-cd.yml:

Continuous Integration (CI): Runs automatic tests and code analysis on every Pull Request to ensure code quality.

Continuous Deployment (CD): Automatically builds the Docker image and deploys the service using SSH upon successful merging to the main branch.

Retraining Loop: The Airflow DAG is scheduled to run daily, ensuring the model is constantly retrained and updated assets are saved for the FastAPI app to consume, closing the MLOps loop.