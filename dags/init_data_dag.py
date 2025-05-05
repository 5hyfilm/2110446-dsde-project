from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
import psycopg2
import os
from datetime import datetime, timedelta

POSTGRES_USER = os.getenv('POSTGRES_USER', 'airflow')  # Default to 'airflow' if not set
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'airflow')  # Default to 'airflow' if not set
POSTGRES_DB = os.getenv('POSTGRES_DB', 'airflow')  # Default to 'airflow' if not set
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')  # Default to 'postgres' service in Docker Compose
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')  # Default to '5432'

def initialize_data():
    # Set up the connection to Postgres using Airflow's PostgresHook
    hook = PostgresHook(postgres_conn_id='postgres_default')

    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )
        
    cursor = conn.cursor()
    
    # Example SQL for initializing data
    init_sql = """
    CREATE EXTENSION IF NOT EXISTS postgis;
    CREATE EXTENSION IF NOT EXISTS postgis_topology;
    
    CREATE TABLE IF NOT EXISTS issues (
        id SERIAL PRIMARY KEY,
        message_id INT UNIQUE NOT NULL,
        type VARCHAR(50),
        coordinates GEOGRAPHY(Point, 4326),
        problem_type_fondue TEXT[], 
        org TEXT[],              
        org_action TEXT[],       
        description TEXT,
        photo_url TEXT,
        address TEXT,
        subdistrict VARCHAR(255),
        district VARCHAR(255),
        province VARCHAR(255),
        timestamp TIMESTAMP,
        state VARCHAR(50),
        last_activity TIMESTAMP
    );
    """
    
    cursor.execute(init_sql)
    conn.commit()
    cursor.close()


days_ago = datetime.now() - timedelta(days=1)
# Define the DAG
dag = DAG(
    'init_data_dag',  # DAG name
    tags=['dsde'],
    description='A simple DAG to initialize Postgres data',
    schedule='@once',  # Run once
    start_date=days_ago,  # Start date of the DAG
    catchup=False,  # Don't backfill for past runs
)

# Create a task to initialize the data
init_data_task = PythonOperator(
    task_id='init_data_dag',
    python_callable=initialize_data,
    dag=dag,
)
