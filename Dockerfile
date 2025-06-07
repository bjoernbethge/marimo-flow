FROM ghcr.io/mlflow/mlflow:latest

WORKDIR /mlflow

ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

ENV MLFLOW_DEFAULT_MODEL_ROOT=/mlflow/models

ENV MLFLOW_DEFAULT_EXPERIMENT_ROOT=/mlflow/experiments

ENV MLFLOW_DEFAULT_RUN_ROOT=/mlflow/runs

ENV MLFLOW_DEFAULT_PROMPT_ROOT=/mlflow/prompts

ENV MLFLOW_HOST=0.0.0.0

ENV MLFLOW_PORT=5000

COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

RUN --mount=type=bind,source=./data/mlflow,target=./mlflow

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri sqlite:///db/mlflow.db \
    --default-artifact-root /artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts

