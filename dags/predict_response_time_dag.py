from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
import os
import logging
import random
import psycopg2
from datetime import datetime, timedelta

POSTGRES_USER = os.getenv('POSTGRES_USER', 'airflow')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'airflow')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'airflow')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

logger = logging.getLogger(__name__)

def predict_response_time():
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, response_time FROM issues
        WHERE response_time IS NOT NULL AND predicted_response_time IS NULL
    """)
    rows = cursor.fetchall()

    for row in rows:
        issue_id, response_time = row
        if response_time is not None:
            offset = int(response_time * 5.0)
            lower = max(0, response_time - offset)
            upper = response_time + offset
            prediction = random.randint(lower, upper)

            cursor.execute("""
                UPDATE issues
                SET predicted_response_time = %s
                WHERE id = %s
            """, (prediction, issue_id))

    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Predictions updated.")

default_args = {
    'owner': 'airflow',
    'start_date': datetime.now() - timedelta(days=1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='predict_response_time_dag',
    description='Randomly predict response time after fetching issue data',
    schedule=None,
    default_args=default_args,
    catchup=False,
    tags=['dsde'],
) as dag:

    predict_task = PythonOperator(
        task_id='predict_response_time',
        python_callable=predict_response_time,
    )

    predict_task
