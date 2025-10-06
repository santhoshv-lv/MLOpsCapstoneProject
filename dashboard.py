import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
# Replace with your public IP or localhost address
API_URL = "http://54.82.75.213:8000" 

st.set_page_config(layout="wide", page_title="Retail Insights Dashboard")

# --- FUNCTIONS TO CALL FASTAPI ENDPOINTS ---

@st.cache_data
def fetch_api_data(endpoint):
    """Fetches data from the FastAPI backend."""
    try:
        response = requests.get(f"{API_URL}/{endpoint}")
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API at {endpoint}: {e}")
        return None

def predict_segment(recency, frequency, monetary):
    """Posts customer data to the prediction endpoint."""
    payload = {
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary
    }
    try:
        response = requests.post(f"{API_URL}/predict/segment", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Prediction failed. Ensure API is running. Details: {e}")
        return {"predicted_cluster": "Error"}


# --- DASHBOARD LAYOUT ---

st.title(" Retail Customer & Store Insights Dashboard")

# ----------------- ROW 1: Store Performance & Trends -----------------
st.header("1. Store Performance & Sales Trends")
col1, col2 = st.columns(2)

# Column 1: Store Performance
store_data = fetch_api_data("stores/performance")
if store_data:
    df_store = pd.DataFrame(store_data.items(), columns=['Mall', 'Revenue ($)']).sort_values('Revenue ($)', ascending=False)
    
    with col1:
        st.subheader("Top Performing Stores (Total Revenue)")
        # Plotting the data
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Revenue ($)', y='Mall', data=df_store, palette='viridis', ax=ax)
        ax.set_title("Revenue by Shopping Mall")
        st.pyplot(fig)
        
# Column 2: Monthly Sales Trends
trends_data = fetch_api_data("sales/monthly-trends")
if trends_data:
    df_trends = pd.DataFrame(trends_data.items(), columns=['Date', 'Sales']).sort_values('Date')
    df_trends['Date'] = pd.to_datetime(df_trends['Date'])
    
    with col2:
        st.subheader("Monthly Sales Volume Trends")
        st.line_chart(df_trends.set_index('Date'))
        st.info("Insight: Use this trend for inventory forecasting and seasonal marketing.")

st.markdown("---")

# ----------------- ROW 2: Customer Segmentation & Prediction -----------------
st.header("2. Customer Loyalty & Segmentation")
col3, col4 = st.columns([0.7, 0.3])

# Column 3: Top Customers Table
top_cust_data = fetch_api_data("customers/top")
if top_cust_data:
    df_top_cust = pd.DataFrame(top_cust_data)
    
    with col3:
        st.subheader(f"Top {len(df_top_cust)} Customers by Monetary Value")
        st.dataframe(df_top_cust)
        st.info("Action: These customers are ideal targets for exclusive loyalty programs.")

# Column 4: Real-time Prediction
with col4:
    st.subheader("Predict Customer Segment")
    st.markdown("Enter customer RFM metrics:")
    
    r = st.number_input("Recency (Days Ago)", min_value=0, value=30)
    f = st.number_input("Frequency (Purchases)", min_value=1, value=5)
    m = st.number_input("Monetary ($)", min_value=10.0, value=500.0)

    if st.button("Predict Segment"):
        with st.spinner('Predicting...'):
            prediction = predict_segment(r, f, m)
            if prediction and prediction.get("predicted_cluster") != "Error":
                cluster_map = {
                    '0': 'Loyal Champions ', 
                    '1': 'At-Risk/Churning ⚠️', 
                    '2': 'New Customers '
                    # Map your cluster IDs to actual names here
                }
                
                predicted_id = prediction.get("predicted_cluster")
                segment_name = cluster_map.get(predicted_id, f"Cluster {predicted_id}")

                st.success(f"### Predicted Segment: {segment_name}")
                st.markdown(f"**Cluster ID:** `{predicted_id}`")
            else:
                st.error("Prediction failed. Check API logs.")
