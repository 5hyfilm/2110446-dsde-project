from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import joblib
import logging
import os
import pickle
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

from model_pipeline import final_pipeline_fn

sys.path.append("/opt/airflow/dags/models")
# PostgreSQL connection settings
POSTGRES_USER = os.getenv('POSTGRES_USER', 'airflow')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'airflow')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'airflow')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

logger = logging.getLogger(__name__)

def etl_postgres_pipeline():
    # Connect to PostgreSQL
    engine = create_engine(
        f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
    )
    

    # Step 1: Extract data from table
    query = """
        SELECT id, type, org_action as organization, description as comment, timestamp, last_activity FROM issues
        WHERE response_time IS NOT NULL AND predicted_response_time IS NULL
        LIMIT 10
    """
    df = pd.read_sql(query, engine)
    data_df = df[['type', 'organization', 'comment', 'timestamp', 'last_activity']]

    logger.info("Pipeline loaded successfully.")
    logger.info("Data for prediction: %s", data_df.head())
    # Step 2: Load pipeline and make predictions
    # pipeline = joblib.load('/opt/airflow/dags/models/full_pipeline.pkl')

    # with open("/opt/airflow/dags/models/today.pkl", "rb") as f:
    #     my_model = pickle.load(f)

    # Load the pipeline
    # with open("/opt/airflow/dags/models/final_pipeline_new.pkl", "rb") as f:
    #     final_pipeline_fn = pickle.load(f)

    # predictions = final_pipeline_fn.predict(df)
    predictions = final_pipeline_fn(data_df)

    # predictions = pipeline(data_df)  # use appropriate column

    # Step 3: Merge predictions back into original data (or build new DataFrame)
    df_result = pd.DataFrame({
        'id': df['id'],
        'predicted_response_time': predictions  # if multilabel, convert appropriately
    })

    df_result['id'] = df_result['id'].astype(int)
    df_result['predicted_response_time'] = df_result['predicted_response_time'].astype(int)  # or int, depending on your use case


    

    # Step 4: Load/update into PostgreSQL
    # df_result.to_sql('issues', engine, if_exists='replace', index=False)

    # with engine.connect() as conn:
    #     for _, row in df_result.iterrows():
    #         conn.execute(
    #             text("""
    #                 UPDATE issues
    #                 SET predicted_response_time = :predicted_response_time
    #                 WHERE id = :id
    #             """),
    #             {"predicted_response_time": row["predicted_response_time"], "id": row["id"]}
    #         )
    #     conn.commit()
    
    cursor = engine.cursor()
    for _, row in df_result.iterrows():
        cursor.execute("""
            UPDATE issues
            SET predicted_response_time = %s
            WHERE id = %s
        """, (row['predicted_response_time'], row['id']))
    
    engine.commit()
    
    # OR to update existing table row-by-row:
    # with engine.begin() as conn:
    #     for _, row in df_result.iterrows():
    #         conn.execute(
    #             sqlalchemy.text("UPDATE raw_data SET predicted_label = :label WHERE id = :id"),
    #             {"label": row["predicted_labels"], "id": row["id"]}
    #         )

# Define DAG

default_args = {
    'owner': 'airflow',
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='predict_response_time_from_model_dag',
    schedule=None,
    default_args=default_args,
    tags=['dsde'],
    catchup=False,
) as dag:

    run_etl_task = PythonOperator(
        task_id='run_etl_pipeline',
        python_callable=etl_postgres_pipeline,
    )

    run_etl_task
