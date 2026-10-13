# Image du projet : Airflow + nos dépendances Python (pandas, scikit-learn, pymongo...)
FROM apache/airflow:3.2.2

USER airflow
COPY requirements.txt /tmp/requirements.txt
# On réinstalle la même version d'Airflow 
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt

# Le code est monté dans /opt/project (voir docker-compose.yml) : `import src...` doit fonctionner
ENV PYTHONPATH=/opt/project \
    MPLCONFIGDIR=/tmp/matplotlib
