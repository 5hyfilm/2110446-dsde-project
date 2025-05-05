from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import psycopg2
import logging
from psycopg2.extras import RealDictCursor
import json
import os
from datetime import datetime, timedelta
# Database connection parameters (read from environment variables for security)
POSTGRES_USER = os.getenv('POSTGRES_USER', 'airflow')  # Default to 'airflow' if not set
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'airflow')  # Default to 'airflow' if not set
POSTGRES_DB = os.getenv('POSTGRES_DB', 'airflow')  # Default to 'airflow' if not set
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')  # Default to 'postgres' service in Docker Compose
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')  # Default to '5432'
logger = logging.getLogger(__name__)

# Function to fetch and insert data into PostgreSQL
def fetch_and_insert_data_into_db():
    # API endpoint
    logging.debug("Fetching data from API...")
    api_url = "https://publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1"  # Replace with your actual API URL
    
    # Make the request to the API
    response = requests.get(api_url)
    logging.debug("completed fetch_and_insert_data...")
    logging.debug("API response status code: %s", response.status_code)
    if response.status_code == 200:
        data = response.json()

        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )

        cursor = conn.cursor()
        
        # Iterate over the features in the response
        for feature in data.get('features', []):
            logging.debug("Processing feature with message_id: %s", feature.get('properties', {}).get('message_id'))
            message_id = feature.get('properties', {}).get('message_id')
            
            if message_id:
                # Check if the message_id already exists in the database
                cursor.execute("SELECT message_id FROM issues WHERE message_id = %s", (message_id,))
                existing_message = cursor.fetchone()

                data = {
                    'message_id': message_id,
                    'type': feature.get('type'),
                    'coordinates': feature['geometry']['coordinates'],
                    'problem_type_fondue': feature['properties'].get('problem_type_fondue'),
                    'org': feature['properties'].get('org'),
                    'org_action': feature['properties'].get('org_action'),
                    'description': feature['properties'].get('description'),
                    'photo_url': feature['properties'].get('photo_url'),
                    'address': feature['properties'].get('address'),
                    'subdistrict': feature['properties'].get('subdistrict'),
                    'district': feature['properties'].get('district'),
                    'province': feature['properties'].get('province'),
                    'timestamp': feature['properties'].get('timestamp'),
                    'state': feature['properties'].get('state'),
                    'last_activity': feature['properties'].get('last_activity')
                }

                logging.info("Data to be inserted: %s", data)
                

                if not existing_message:
                    # Insert the data into the database
                    cursor.execute("""
                        INSERT INTO issues (
                            message_id, type, coordinates, problem_type_fondue, org, org_action,
                            description, photo_url, address, subdistrict, district, province,
                            timestamp, state, last_activity
                        )
                        VALUES (
                            %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s
                        )
                    """, (
                        data['message_id'],
                        data['type'],
                        data['coordinates'][0],  # Longitude
                        data['coordinates'][1],  # Latitude
                        data['problem_type_fondue'],
                        data['org'],
                        data['org_action'],
                        data['description'],
                        data['photo_url'],
                        data['address'],
                        data['subdistrict'],
                        data['district'],
                        data['province'],
                        data['timestamp'],
                        data['state'],
                        data['last_activity']
                    ))
                    conn.commit()

        # Close the cursor and the connection
        cursor.close()
        conn.close()

        print("Data successfully fetched and inserted.")
    else:
        print(f"Failed to retrieve data from API. Status code: {response.status_code}")

# Default arguments for the DAG
days_ago = datetime.now() - timedelta(days=1)
default_args = {
    'owner': 'airflow',
    'start_date': days_ago,
}

# Define the DAG
with DAG(
    'get_issue_dag',
    tags=['dsde'],
    default_args=default_args,
    description='Fetch data from API and insert into PostgreSQL',
    schedule=None,  # Set to None to run manually or set a schedule
    catchup=False,
) as dag:

    # Step 1: Fetch and insert data into the database
    fetch_and_insert_data = PythonOperator(
        task_id='fetch_and_insert_data',
        python_callable=fetch_and_insert_data_into_db,
    )

    fetch_and_insert_data
