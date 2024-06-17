# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable correctly
ENV TF_CPP_MIN_LOG_LEVEL=2  # Suppress TensorFlow GPU warnings

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
