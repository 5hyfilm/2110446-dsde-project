import os
import psycopg2
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Get DB parameters from environment variables
POSTGRES_USER = os.getenv('POSTGRES_USER', 'airflow')  # Default to 'airflow' if not set
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'airflow')  # Default to 'airflow' if not set
POSTGRES_DB = os.getenv('POSTGRES_DB', 'airflow')  # Default to 'airflow' if not set
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')  # Default to 'postgres' service in Docker Compose
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')  # Default to '5432'

# Function to create a table using psycopg2
def create_table():
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        
        cursor = conn.cursor()
        
        # SQL statement to create a table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS demo_table (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Execute the query
        cursor.execute(create_table_sql)
        
        # Commit the changes to the database
        conn.commit()
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        print("Table created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")

# Function to insert data into the table
def insert_data():
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        
        cursor = conn.cursor()
        
        # SQL statement to insert data
        insert_data_sql = "INSERT INTO demo_table (name) VALUES (%s);"
        
        # Insert data
        cursor.execute(insert_data_sql, ('Hook User',))
        
        # Commit the changes to the database
        conn.commit()
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        print("Data inserted successfully.")
    except Exception as e:
        print(f"Error inserting data: {e}")

default_args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='my_first_dag',
    tags=['my_first_dag'],
    description='A DAG that connects to PostgreSQL using psycopg2',
    schedule=None,  # Set your own schedule
    start_date=datetime.now() - timedelta(days=1),
    catchup=False,
    default_args=default_args,
) as dag:

    create_table_task = PythonOperator(
        task_id='create_table_task',
        python_callable=create_table,
    )

    insert_data_task = PythonOperator(
        task_id='insert_data_task',
        python_callable=insert_data,
    )

    create_table_task >> insert_data_task
