FROM registry.access.redhat.com/ubi9/python-312 AS base

RUN pip install --no-cache-dir mlflow

# to access run secrets
USER 0

RUN --mount=type=secret,id=dagshub_username \
    --mount=type=secret,id=dagshub_token \
    --mount=type=secret,id=mlflow_uri \
    export MLFLOW_TRACKING_USERNAME=$(cat /run/secrets/dagshub_username) && \
    export MLFLOW_TRACKING_PASSWORD=$(cat /run/secrets/dagshub_token) && \
    export MLFLOW_TRACKING_URI=$(cat /run/secrets/mlflow_uri) && \
    mkdir -p /tmp/models && \
    python -c "import mlflow; from mlflow import MlflowClient; client = MlflowClient(); mlflow.artifacts.download_artifacts(artifact_uri=client.get_model_version_by_alias(name='demo_model', alias='Champion').source, dst_path='/tmp/models')"

## FINAL STAGE (Reduces Size) ##
FROM registry.access.redhat.com/ubi9/python-312

# bind mount anc copy only pkl file
# COPY --from=base /tmp/models/artifacts models

# don't bloat final layer with mlflow/pyarrow/etc etc. simple scikit learn model only needs scikit learn to be installed
RUN --mount=type=bind,from=base,src=/tmp/models/artifacts/,target=/tmp/artifacts/,rw \
    pip install --no-cache-dir scikit-learn && \
    mkdir models && \
    cp /tmp/artifacts/model.pkl models/