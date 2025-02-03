ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS python-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-suggests --no-install-recommends --yes curl gcc build-essential  \
    && rm -rf /var/lib/apt/lists/*

FROM python-base AS uv-base

ARG UV_VERSION=0.4.10
ARG UV_LINK_MODE=copy

ENV PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_VERSION=${UV_VERSION} \
    UV_LINK_MODE=${UV_LINK_MODE}

RUN pip install uv==$UV_VERSION

FROM uv-base AS build-base

ARG USER=user

ENV HOME=/home/${USER}
ENV HF_HOME=${HOME}/.cache/huggingface \
    WORKSPACE=${HOME}/app/

RUN mkdir -p ${WORKSPACE}

WORKDIR ${WORKSPACE}

COPY pyproject.toml requirements.lock ${WORKSPACE}/

RUN uv pip install --no-cache --system -r requirements.lock

FROM build-base AS build

ARG PORT=8000
ARG PACKAGE_NAME=ts_clf_event

ENV PORT=${PORT} \
    PACKAGE_NAME=${PACKAGE_NAME}

COPY --from=build-base ${WORKSPACE} ${WORKSPACE}

COPY ./src/${PACKAGE_NAME} ${WORKSPACE}/${PACKAGE_NAME}

# Copy the CSV file into the image for simplicity.
#COPY ./data/test_dataframe.csv /home/user/data/test_dataframe.csv

# Same for the model
#COPY ./output/models/RF_model.pkl /home/user/output/models/RF_model.pkl

EXPOSE ${PORT}

ARG USER
ARG UID=10001

RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "${HOME}" \
    --no-create-home \
    --uid "${UID}" \
    ${USER}

RUN chown -R ${USER} ${HOME}

USER ${USER}

CMD ["uvicorn", "ts_clf_event.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
