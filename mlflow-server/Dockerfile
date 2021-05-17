# mlflow-server

FROM python:3.7

USER root

WORKDIR /app

ADD ./ /app

ENV PATH "$PATH:/root/.local/bin"

RUN pip install --no-cache mlflow && \
    pip install --no-cache awscli --upgrade --user && \
    pip install --no-cache boto3

# EXPOSE 5000

CMD ["python", "run_MLFlow_server.py"]
