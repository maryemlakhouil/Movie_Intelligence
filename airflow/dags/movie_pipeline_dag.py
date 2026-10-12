"""DAG : Extraction -> Nettoyage -> Features -> MongoDB -> NLP -> ML (classification, clustering)."""
from datetime import datetime, timedelta

try:
    from airflow.providers.standard.operators.python import PythonOperator
    from airflow.sdk import DAG
except ImportError:  
    from airflow import DAG
    from airflow.operators.python import PythonOperator

from src.pipeline import STAGES, run_stage  

with DAG(
    dag_id="movie_pipeline",
    description="Pipeline films TMDB : extraction, nettoyage, features, MongoDB, NLP, modèles",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["movies"],
) as dag:
    tasks = [PythonOperator(task_id=name, python_callable=run_stage, op_args=[name]) for name in STAGES]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream   
