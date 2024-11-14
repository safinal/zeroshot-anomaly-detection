# -----------------------------------------------------------------------------
#  A sample Dockerfile to help you replicate our test environment
# -----------------------------------------------------------------------------

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY . .

# Install your python and apt requirements
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install $(cat apt_requirements.txt) -y
RUN chmod +x run.sh

CMD ["python3", "runner.py"]