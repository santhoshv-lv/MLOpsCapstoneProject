# 1. Activate your environment
source ~/airflow_venv/bin/activate

# 2. Set AIRFLOW_HOME
export AIRFLOW_HOME=~/airflow

# 3. EXPORT KAGGLE CREDENTIALS MANUALLY (CRITICAL)
export KAGGLE_USERNAME="santhoshvlatentview"
export KAGGLE_KEY="b7cfd8f85a1deb57d0aa0f15aacf6b50"

# 4. Start the Airflow Scheduler in the same session
airflow scheduler
