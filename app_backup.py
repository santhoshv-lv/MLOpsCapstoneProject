from fastapi import FastAPI
import pandas as pd
from typing import List, Dict
import joblib
import os
import re
from pydantic import BaseModel

# --- CONFIGURATION (MUST MATCH AIRFLOW PATHS) ---
# NOTE: Using the /root/airflow/data path for consistency
DATA_DIR = "/root/airflow/data"
RFM_FILE_NAME = "rfm_with_cluster.csv"
MODEL_FILE_NAME = "kmeans_model.joblib"
RFM_FILE_PATH = os.path.join(DATA_DIR, RFM_FILE_NAME)
MODEL_FILE_PATH = os.path.join(DATA_DIR, MODEL_FILE_NAME)
RAW_FILE_PATH = os.path.join(DATA_DIR, "customer_shopping_data.csv")

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

# Global variables to store the processed data and model
df_raw = None
df_rfm = None
kmeans_model = None

@app.on_event("startup")
def load_data_and_model_assets():
    """
    This function loads the pre-processed data and the trained model saved by the Airflow pipeline.
    """
    global df_raw, df_rfm, kmeans_model

    try:
        # Load the raw data (for performance and trend endpoints)
        df_raw = pd.read_csv(RAW_FILE_PATH)
        df_raw['invoice_date'] = pd.to_datetime(df_raw['invoice_date'], dayfirst=True)
        df_raw['total_price'] = df_raw['quantity'] * df_raw['price']

        # Load the final RFM data with cluster labels (This file is now CORRECT)
        df_rfm = pd.read_csv(RFM_FILE_PATH)

        # 1. Clean the column names (convert to lowercase, remove whitespace)
        df_rfm.columns = df_rfm.columns.str.lower().str.strip()

        # 2. **ROBUST FIX:** Check for and rename the 'monetary' column
        if 'monetary' not in df_rfm.columns:
            monetary_cols = [col for col in df_rfm.columns if re.search(r'monetary', col, re.IGNORECASE)]
            
            if monetary_cols:
                df_rfm.rename(columns={monetary_cols[0]: 'monetary'}, inplace=True)
            else:
                raise KeyError(f"Missing required RFM column 'monetary'. Columns found: {df_rfm.columns.tolist()}")

        # Load the trained K-Means model
        kmeans_model = joblib.load(MODEL_FILE_PATH)

        print("Data and model loaded successfully from pipeline assets.")

    except Exception as e:
        print(f"ERROR: Application startup failed. Details: {e}")
        # Rerunning Airflow is the only fix if this fails
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

        # This line should now succeed because 'monetary' is guaranteed to exist
        top_customers_df = df_rfm.sort_values(by='monetary', ascending=False).head(top_n)
        return top_customers_df.to_dict('records')
    return []

@app.get("/sales/monthly-trends", summary="Get monthly sales trends")
def get_monthly_sales_trends() -> Dict[str, float]:
    """Returns total sales data aggregated by month to show seasonal trends."""
    if df_raw is not None:
        
        # --- FIX: Group by explicit Series rename to avoid 'ValueError: cannot insert invoice_date' ---
        monthly_sales = df_raw.groupby(
            [df_raw['invoice_date'].dt.year.rename('year'),
             df_raw['invoice_date'].dt.month.rename('month')]
        )['total_price'].sum().reset_index()

        # The subsequent logic is correct for combining the month/year
        monthly_sales['date'] = monthly_sales['year'].astype(str) + '-' + monthly_sales['month'].astype(str).str.zfill(2)
        return monthly_sales.set_index('date')['total_price'].to_dict()
    return {}

@app.post("/predict/segment", summary="Predict the segment for a new customer")
def predict_customer_segment(customer: RFMInput) -> Dict[str, str]:
    """
    Predicts the customer cluster (segment) based on new RFM inputs (Recency, Frequency, Monetary).
    """
    if kmeans_model is not None:
        # NOTE: A saved StandardScaler object must be loaded and used for production accuracy!
        new_customer_data = pd.DataFrame([[customer.recency, customer.frequency, customer.monetary]],
                                         columns=['recency', 'frequency', 'monetary'])

        predicted_cluster = kmeans_model.predict(new_customer_data.values)

        return {"predicted_cluster": str(predicted_cluster[0]),
                "message": "WARNING: Load and use the saved StandardScaler object for production accuracy!"}

    return {"error": "Model not loaded or Airflow pipeline failed."}
