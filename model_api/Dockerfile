# Use the official Python image from the Docker Hub
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY app.py .

# Expose the port that the Flask app will run on
EXPOSE 7777

# Define the command to run the Flask app
CMD ["python", "app.py"]