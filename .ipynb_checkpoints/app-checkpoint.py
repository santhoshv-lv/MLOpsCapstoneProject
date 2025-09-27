from fastapi import FastAPI
import pandas as pd
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# Suppress FutureWarning from scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize the FastAPI app
app = FastAPI(
    title="Retail Store & Customer Insights API",
    description="API for serving retail analytics and ML model insights."
)

# Global variables to store the processed data and model
df = None
rfm_df = None
kmeans_model = None

@app.on_event("startup")
def load_data_and_train_model():
    """
    This function runs once when the application starts up.
    It loads the data, performs all the processing, and trains the model.
    """
    global df, rfm_df, kmeans_model

    try:
        # Step 1: Data Ingestion and Cleaning
        df = pd.read_csv('customer_shopping_data.csv')
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)

        # Step 2: Feature Engineering and Analysis
        df['total_price'] = df['quantity'] * df['price']
        most_recent_date = df['invoice_date'].max()
        rfm_df = df.groupby('customer_id').agg(
            recency=('invoice_date', lambda x: (most_recent_date - x.max()).days),
            frequency=('invoice_no', 'count'),
            monetary=('total_price', 'sum')
        ).reset_index()

        # Step 3: Customer Segmentation using Machine Learning
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
        
        # Train the K-Means model with a chosen optimal_k (e.g., 3 from our analysis)
        optimal_k = 3
        kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        rfm_df['Cluster'] = kmeans_model.fit_predict(rfm_scaled)

        print("Data processing and model training complete on startup.")

    except FileNotFoundError:
        print("Error: 'customer_shopping_data.csv' was not found. Please ensure it's in the correct directory.")
        # You can add more robust error handling here for production

# Define the API Endpoints

@app.get("/stores/performance", summary="Get a list of top-performing stores")
def get_store_performance() -> Dict[str, float]:
    """Returns the total revenue for each shopping mall."""
    if df is not None:
        store_performance = df.groupby('shopping_mall')['total_price'].sum().to_dict()
        return store_performance
    return {}

@app.get("/customers/top", summary="Get the top 10% of customers")
def get_top_customers() -> List[Dict]:
    """Returns a list of the top customers based on their monetary value."""
    if rfm_df is not None:
        top_10_percent = int(len(rfm_df) * 0.1)
        # Use a minimum of 1 for cases where 10% is less than 1
        top_n = max(1, top_10_percent)
        top_customers_df = rfm_df.sort_values(by='monetary', ascending=False).head(top_n)
        return top_customers_df.to_dict('records')
    return []

@app.get("/sales/monthly-trends", summary="Get monthly sales trends")
def get_monthly_sales_trends() -> Dict[str, float]:
    """Returns total sales data aggregated by month to show seasonal trends."""
    if df is not None:
        monthly_sales = df.groupby(['invoice_year', 'invoice_month'])['total_price'].sum().reset_index()
        monthly_sales['date'] = monthly_sales['invoice_year'].astype(str) + '-' + monthly_sales['invoice_month'].astype(str).str.zfill(2)
        return monthly_sales.set_index('date')['total_price'].to_dict()
    return {}