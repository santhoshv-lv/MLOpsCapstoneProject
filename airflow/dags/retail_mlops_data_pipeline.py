from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import pandas as pd
import os
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import subprocess
import re

# --- 1. CONFIGURATION ---
DATA_DIR = "/root/airflow/data" 
KAGGLE_DATASET_IDENTIFIER = "mehmettahiraslan/customer-shopping-dataset"
KAGGLE_FILE_NAME = "customer_shopping_data.csv"
CLEANED_RFM_FILE_NAME = "rfm_with_cluster.csv"
MODEL_FILE_NAME = "kmeans_model.joblib"
SCALER_FILE_NAME = "scaler.joblib" 

FULL_FILE_PATH = os.path.join(DATA_DIR, KAGGLE_FILE_NAME)
CLEANED_RFM_FILE_PATH = os.path.join(DATA_DIR, CLEANED_RFM_FILE_NAME)
MODEL_FILE_PATH = os.path.join(DATA_DIR, MODEL_FILE_NAME)
SCALER_FILE_PATH = os.path.join(DATA_DIR, SCALER_FILE_NAME)

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 2. PYTHON OPERATOR FUNCTIONS ---

def download_data():
    """
    Downloads the dataset using the Kaggle API. Requires KAGGLE_USERNAME/KEY ENV VARS.
    """
    KAGGLE_EXECUTABLE_PATH = "/root/airflow_venv/bin/kaggle" # Use the explicit path

    print(f"Starting automated download of {KAGGLE_DATASET_IDENTIFIER}...")
    
    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
        print("ERROR: Kaggle credentials (KAGGLE_USERNAME/KEY) are not set.")
        raise EnvironmentError("Kaggle credentials missing.")

    try:
        subprocess.run([
            KAGGLE_EXECUTABLE_PATH, "datasets", "download", 
            KAGGLE_DATASET_IDENTIFIER, 
            "--path", DATA_DIR, 
            "--unzip"
        ], check=True)

        print(f"Successfully downloaded and unzipped data to {DATA_DIR}")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Kaggle download failed. Check API key validity.")
        raise e

def process_and_train():
    """Performs data cleaning, feature engineering, RFM, and ML model training."""

    if not os.path.exists(FULL_FILE_PATH):
         raise FileNotFoundError(f"Raw data not found at {FULL_FILE_PATH}.")

    # 1. Load Data
    df = pd.read_csv(FULL_FILE_PATH)
    
    # 2. Data Cleaning & Feature Engineering
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)
    df['total_price'] = df['quantity'] * df['price']
    
    # 3. RFM Analysis
    most_recent_date = df['invoice_date'].max()
    rfm_df = df.groupby('customer_id').agg(
        recency=('invoice_date', lambda x: (most_recent_date - x.max()).days),
        frequency=('invoice_no', 'count'),
        monetary=('total_price', 'sum')
    ).reset_index()

    # 4. K-Means Clustering & Scaling (CRITICAL FIX)
    scaler = StandardScaler()
    
    # **FIX**: Fit the scaler first!
    rfm_scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
    
    # **THEN SAVE** the fitted scaler object
    joblib.dump(scaler, SCALER_FILE_PATH) 
    
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 5. Save Model and Final Data
    joblib.dump(kmeans, MODEL_FILE_PATH)
    rfm_df.to_csv(CLEANED_RFM_FILE_PATH, index=False)
    
    print(f"Model assets saved: {MODEL_FILE_PATH}, {SCALER_FILE_PATH}, {CLEANED_RFM_FILE_PATH}")

def load_data_to_warehouse():
    """Simulates loading the final cleaned data to a data warehouse."""
    print("SUCCESS: Data pipeline complete and model assets are ready for FastAPI.")


# --- 3. AIRFLOW DAG DEFINITION ---

with DAG(
    dag_id='retail_mlops_data_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    default_args={'owner': 'airflow', 'depends_on_past': False, 'retries': 1},
    tags=['mlops', 'retail', 'segmentation']
) as dag:
    
    task_create_dir = BashOperator(
        task_id='create_data_directory',
        bash_command=f"mkdir -p {DATA_DIR}",
    )

    task_download_data = PythonOperator(
        task_id='download_raw_data_kaggle',
        python_callable=download_data,
    )

    task_process_and_train = PythonOperator(
        task_id='process_data_and_train_model',
        python_callable=process_and_train,
    )

    task_load_data = PythonOperator(
        task_id='load_cleaned_data_to_warehouse',
        python_callable=load_data_to_warehouse,
    )
    
    task_create_dir >> task_download_data >> task_process_and_train >> task_load_data
