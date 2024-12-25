# Use the official Streamlit image as a base
FROM python:3.11

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Run the specified command to start the Streamlit application
CMD ["streamlit", "run", "--server.port", "8080", "application.py"]
