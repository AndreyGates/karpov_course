# Use python based image
#FROM apache/airflow:2.0.0-python3.8
FROM python:3.8-slim

# Arguments, passed from the command line when building the image
# using the --build-arg flag
ARG DB_PATH

# Python environment
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
#ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory to /app
WORKDIR /app

# Set the user to root
#USER root

# Copy the requirements.txt file to the image
COPY requirements.txt /tmp/requirements.txt

# Linux essentials
RUN rm -rf /var/lib/apt/lists/* && \
# Create a virtual environment and install the required packages
    python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install -r /tmp/requirements.txt

# Copy the Python script to the image
COPY filter.py .
COPY ${DB_PATH} users.db

# Set the default command to run when a container is created from the image
CMD ["python", "filter.py", "--db-path", "users.db"]
