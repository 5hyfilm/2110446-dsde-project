import os
import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import requests
import psycopg2

# PostgreSQL connection from environment variables
POSTGRES_USER = os.getenv('POSTGRES_USER', 'airflow')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'airflow')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'airflow')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

# Logging setup
logger = logging.getLogger(__name__)

# Function to fetch and insert/update data
def fetch_and_insert_data_into_db():
    api_url = "https://publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1"
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        conn = psycopg2.connect(
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        cursor = conn.cursor()

        for feature in data.get('features', []):
            props = feature.get('properties', {})
            message_id = props.get('message_id')
            if not message_id:
                continue

            timestamp = props.get('timestamp')
            last_activity = props.get('last_activity')

            # Calculate response time in hours
            response_time = None
            if timestamp and last_activity:
                try:
                    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    la = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
                    delta = la - ts
                    response_time = int(delta.total_seconds() // 3600)
                except Exception as e:
                    logger.warning("Time parse error for ID %s: %s", message_id, e)

            cursor.execute("""
                INSERT INTO issues (
                    message_id, type, coordinates, problem_type_fondue, org, org_action,
                    description, photo_url, address, subdistrict, district, province,
                    timestamp, state, last_activity, response_time
                )
                VALUES (
                    %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (message_id) DO UPDATE SET
                    type = EXCLUDED.type,
                    coordinates = EXCLUDED.coordinates,
                    problem_type_fondue = EXCLUDED.problem_type_fondue,
                    org = EXCLUDED.org,
                    org_action = EXCLUDED.org_action,
                    description = EXCLUDED.description,
                    photo_url = EXCLUDED.photo_url,
                    address = EXCLUDED.address,
                    subdistrict = EXCLUDED.subdistrict,
                    district = EXCLUDED.district,
                    province = EXCLUDED.province,
                    timestamp = EXCLUDED.timestamp,
                    state = EXCLUDED.state,
                    last_activity = EXCLUDED.last_activity,
                    response_time = EXCLUDED.response_time
            """, (
                message_id,
                props.get('type'),
                feature['geometry']['coordinates'][0],  # lon
                feature['geometry']['coordinates'][1],  # lat
                props.get('problem_type_fondue'),
                props.get('org'),
                props.get('org_action'),
                props.get('description'),
                props.get('photo_url'),
                props.get('address'),
                props.get('subdistrict'),
                props.get('district'),
                props.get('province'),
                timestamp,
                props.get('state'),
                last_activity,
                response_time
            ))

        conn.commit()
        cursor.close()
        conn.close()
        print("Data successfully fetched and inserted/updated.")
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")

# DAG setup
default_args = {
    'owner': 'airflow',
    'start_date': datetime.now() - timedelta(days=1),
}

with DAG(
    'get_issue_dag',
    default_args=default_args,
    description='Fetch Traffy data and insert/update PostgreSQL with response_time',
    schedule='@daily',
    catchup=False,
    tags=['dsde'],
) as dag:

    fetch_and_insert_data = PythonOperator(
        task_id='fetch_and_insert_data',
        python_callable=fetch_and_insert_data_into_db,
    )

    trigger_predict_dag = TriggerDagRunOperator(
        task_id="trigger_predict_response_time_from_model_dag",
        trigger_dag_id="predict_response_time_from_model_dag",
    )

    # fetch_and_insert_data 
    fetch_and_insert_data >> trigger_predict_dag
