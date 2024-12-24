## Use the official Streamlit image as a base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the local directory to /app in the container
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Command to run the Streamlit application
CMD ["streamlit", "run", "application.py", "--server.port", "8080", "--server.enableXsrfProtection=false"]
