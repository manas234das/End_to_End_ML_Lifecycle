# demo-v1

FROM python:3.7

# MAINTAINER manas.das@rubikonlabs.com

USER root

WORKDIR /app

ADD ./ /app

ENV PATH "$PATH:/root/.local/bin"

RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt
RUN pip install --no-cache mlflow
RUN pip install --no-cache xgboost
RUN pip install --no-cache awscli --upgrade --user && \
    pip install --no-cache boto3

ENV AWS_ACCESS_KEY_ID=AKIA4HS2Q4ZCFYQFUAVH
ENV AWS_SECRET_ACCESS_KEY=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
ENV AWS_DEFAULT_REGION=us-east-1

# EXPOSE 8200

CMD ["python", "run_app_server.py"]
