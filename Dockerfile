FROM python:3.11.2
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
