  GNU nano 6.2                                                                                     Dockerfile                                                                                              
# Use an NVIDIA CUDA base image with Python, matching your CUDA version
FROM nvidia/cuda:12.3.0-base-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and other necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-venv python3.11-dev \
    build-essential cmake git libopenblas-dev

# Update alternatives to use Python 3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    python3 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Ensure we upgrade pip to the latest version and install the requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Now, install llama-cpp-python with CUDA support enabled
# Note: Setting CMAKE_ARGS before the pip install command to ensure CUDA support
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
RUN pip install llama-cpp-python

# Copy the Python script and model file into the container
COPY connection.py /app/
COPY ./solar-10.7b-instruct-v1.0-uncensored.Q8_0.gguf /app/models/

# Run your script when the container launches
CMD ["python3", "-m", "llama_cpp.server", "--model", "solar-10.7b-instruct-v1.0-uncensored.Q8_0.gguf", "--n_gpu_layers=-1", "--n_batch=512", "--n_threads=24", "--host", "0.0.0.0"]




