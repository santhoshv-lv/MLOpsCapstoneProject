from fastapi import FastAPI
import pandas as pd
from typing import List, Dict

# Initialize the FastAPI app
app = FastAPI(
    title="Retail Store & Customer Insights API",
    description="API for serving retail analytics and ML model insights."
)

# Load your cleaned data and generated insights
# In a real-world scenario, this data would come from a database or a data warehouse
# For this example, we'll assume the dataframes from previous steps are available

# Sample dataframes (replace with your actual dataframes)
rfm_df = pd.DataFrame({
    'customer_id': ['C100004', 'C100005', 'C100006'],
    'recency': [467, 5, 97],
    'frequency': [1, 1, 1],
    'monetary': [7502.00, 2400.68, 322.56],
    'Cluster': [1, 0, 0]
})
store_performance_data = {'Metropol AVM': 3000.85, 'Kanyon': 1500.40}
monthly_sales_data = {'2021-05': 3000.85, '2021-11': 300.08, '2021-12': 5401.53, '2022-08': 7502.00}

# Define the API Endpoints

@app.get("/stores/performance", summary="Get a list of top-performing stores")
def get_store_performance() -> Dict[str, float]:
    """
    Returns the total revenue for each shopping mall.
    """
    return store_performance_data

@app.get("/customers/top", summary="Get the top 10% of customers")
def get_top_customers() -> List[Dict]:
    """
    Returns a list of the top customers based on their monetary value.
    """
    # Assuming you have a full rfm_df from your analysis
    top_customers_df = rfm_df.sort_values(by='monetary', ascending=False).head(int(len(rfm_df) * 0.1))
    return top_customers_df.to_dict('records')

@app.get("/sales/monthly-trends", summary="Get monthly sales trends")
def get_monthly_sales_trends() -> Dict[str, float]:
    """
    Returns total sales data aggregated by month to show seasonal trends.
    """
    return monthly_sales_data

# To run the app, save the file and use the following command in your terminal:
# uvicorn app:app --reload