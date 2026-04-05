FROM python:3.11-slim
WORKDIR /app

# Install backend deps (no TensorFlow here yet)
COPY Backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY Backend ./Backend

# Copy model code + weights into backend image
# Adjust the path if export/models lives elsewhere
COPY export/models ./export/models

CMD ["uvicorn", "Backend.backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
