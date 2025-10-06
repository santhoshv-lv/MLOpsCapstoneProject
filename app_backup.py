from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd
from typing import List, Dict
import joblib
import os
import re
import time # Added for measuring latency
from pydantic import BaseModel
from loguru import logger # Added for structured logging
from sklearn.preprocessing import StandardScaler # Required for type checking, though loaded via joblib


# --- CONFIGURATION (MUST MATCH AIRFLOW PATHS) ---
DATA_DIR = "/root/airflow/data"
RFM_FILE_NAME = "rfm_with_cluster.csv"
MODEL_FILE_NAME = "kmeans_model.joblib"
SCALER_FILE_NAME = "scaler.joblib"

RFM_FILE_PATH = os.path.join(DATA_DIR, RFM_FILE_NAME)
MODEL_FILE_PATH = os.path.join(DATA_DIR, MODEL_FILE_NAME)
SCALER_FILE_PATH = os.path.join(DATA_DIR, SCALER_FILE_NAME)
RAW_FILE_PATH = os.path.join(DATA_DIR, "customer_shopping_data.csv")

# Set up logging to save output to a file for monitoring dashboards
logger.add("api_logs.json", rotation="10 MB", compression="zip", serialize=True)


# Pydantic model for structured prediction request body
class RFMInput(BaseModel):
    recency: int
    frequency: int
    monetary: float

# Initialize the FastAPI app
app = FastAPI(
    title="Retail Store & Customer Insights API",
    description="API for serving retail analytics and ML model insights."
)

# Global variables to store the processed data, model, and scaler
df_raw = None
df_rfm = None
kmeans_model = None
scaler_model = None

# --- NEW: API MONITORING MIDDLEWARE ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs the details of every API request, including latency and status code."""
    start_time = time.time()
    
    # 1. Process the request and get the response
    response = await call_next(request)
    
    # 2. Calculate latency
    process_time = time.time() - start_time
    latency_ms = round(process_time * 1000, 2)
    
    # 3. Log the structured data
    logger.info({
        "event": "api_request",
        "client_host": request.client.host,
        "endpoint": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "user_agent": request.headers.get('user-agent')
    })
    
    return response

# --- STARTUP FUNCTION ---
@app.on_event("startup")
def load_data_and_model_assets():
    """Loads all data and model assets saved by the Airflow pipeline on startup."""
    global df_raw, df_rfm, kmeans_model, scaler_model

    try:
        # Load the raw data (for performance and trend endpoints)
        df_raw = pd.read_csv(RAW_FILE_PATH)
        df_raw['invoice_date'] = pd.to_datetime(df_raw['invoice_date'], dayfirst=True)
        df_raw['total_price'] = df_raw['quantity'] * df_raw['price']

        # Load the final RFM data 
        df_rfm = pd.read_csv(RFM_FILE_PATH)

        # 1. Clean column names 
        df_rfm.columns = df_rfm.columns.str.lower().str.strip()

        # 2. Robust check for 'monetary' column
        if 'monetary' not in df_rfm.columns:
            monetary_cols = [col for col in df_rfm.columns if re.search(r'monetary', col, re.IGNORECASE)]
            
            if monetary_cols:
                df_rfm.rename(columns={monetary_cols[0]: 'monetary'}, inplace=True)
            else:
                raise KeyError(f"Missing 'monetary' column. Columns: {df_rfm.columns.tolist()}")

        # Load the trained K-Means model and the fitted StandardScaler
        kmeans_model = joblib.load(MODEL_FILE_PATH)
        scaler_model = joblib.load(SCALER_FILE_PATH) 

        print("Data, model, and scaler loaded successfully from pipeline assets.")

    except Exception as e:
        logger.error(f"FATAL ERROR during startup: {e}")
        # Reraise the exception to ensure Uvicorn reports a startup failure
        raise e

# --- API ENDPOINTS ---

@app.get("/stores/performance", summary="Get a list of top-performing stores")
def get_store_performance() -> Dict[str, float]:
    """Returns the total revenue for each shopping mall."""
    if df_raw is not None:
        store_performance = df_raw.groupby('shopping_mall')['total_price'].sum().to_dict()
        return store_performance
    return {}

@app.get("/customers/top", summary="Get the top 10% of customers")
def get_top_customers() -> List[Dict]:
    """Returns a list of the top customers based on their monetary value."""
    if df_rfm is not None:
        top_10_percent = int(len(df_rfm) * 0.1)
        top_n = max(1, top_10_percent) 
        
        top_customers_df = df_rfm.sort_values(by='monetary', ascending=False).head(top_n)
        return top_customers_df.to_dict('records')
    return []

@app.get("/sales/monthly-trends", summary="Get monthly sales trends")
def get_monthly_sales_trends() -> Dict[str, float]:
    """Returns total sales data aggregated by month to show seasonal trends."""
    if df_raw is not None:
        
        monthly_sales = df_raw.groupby(
            [df_raw['invoice_date'].dt.year.rename('year'),
             df_raw['invoice_date'].dt.month.rename('month')]
        )['total_price'].sum().reset_index()

        monthly_sales['date'] = monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str).str.zfill(2)
        return monthly_sales.set_index('date')['total_price'].to_dict()
    return {}

@app.post("/predict/segment", summary="Predict the segment for a new customer")
def predict_customer_segment(customer: RFMInput) -> Dict[str, str]:
    """
    Predicts the customer cluster (segment) based on new RFM inputs (Recency, Frequency, Monetary).
    """
    if kmeans_model is not None and scaler_model is not None:
        
        new_customer_data = pd.DataFrame([[customer.recency, customer.frequency, customer.monetary]],
                                         columns=['recency', 'frequency', 'monetary'])
        
        # Scale the data using the loaded, fitted scaler
        new_customer_scaled = scaler_model.transform(new_customer_data)
        
        predicted_cluster = kmeans_model.predict(new_customer_scaled)

        return {"predicted_cluster": str(predicted_cluster[0]),
                "message": "Prediction made successfully using loaded K-Means model and StandardScaler."}

    return {"error": "Model or Scaler not loaded. Airflow pipeline may have have failed."}
