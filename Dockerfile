FROM python:3.8.15

ENV PYTHONUNBUFFERED=True

# Set up working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the necessary files
COPY predict.py ./predict.py
COPY body_performance_model.bin ./body_performance_model.bin

# Expose the port that Flask app will run on
EXPOSE 9696

# Run gunicorn to serve the Flask app
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]