FROM apache/airflow:2.8.2

# Install Postgres provider
RUN pip install apache-airflow-providers-postgres
