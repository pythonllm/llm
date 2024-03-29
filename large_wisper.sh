#!/bin/bash

PID=$(lsof -ti:8000 | head -n 1)

if [ ! -z "$PID" ]; then
  kill $PID
  sleep 2
fi

cd /wisper_large

gunicorn -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000 --timeout 240 --daemon
