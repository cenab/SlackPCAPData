# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container at /app
COPY . .

# Make sure the shell scripts have execute permissions
RUN chmod +x tcpdump_files/*.sh

# Expose any necessary ports (if applicable)
# EXPOSE 5000

# Run the application
CMD ["python", "start_data_collection.py"]
