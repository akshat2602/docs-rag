FROM python:3.10-slim-bullseye
# set work directory
WORKDIR /src

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get upgrade

# install poetry
RUN pip install poetry

# install dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi

# copy project
COPY . .

EXPOSE 5000