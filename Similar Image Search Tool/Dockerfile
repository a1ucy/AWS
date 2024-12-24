# Use the official Python image as a base
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Run the Streamlit application
CMD ["streamlit", "run", "--server.port", "8080", "main.py"]
