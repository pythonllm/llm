from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
    model_path="./solar-10.7b-instruct-v1.0-uncensored.Q4_K_M.gguf",  # Download the model file first
    n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=35  # The number of layers to offload to GPU, if you have GPU acceleration available
)

# Simple inference example
output = llm(
    "{prompt}",  # Prompt
    max_tokens=512,  # Generate up to 512 tokens
    stop=["</s>"],  # Example stop token - not necessarily correct for this specific model! Please check before using.
    echo=True  # Whether to echo the prompt
)

# Chat Completion API

llm = Llama(model_path="./solar-10.7b-instruct-v1.0-uncensored.Q4_K_M.gguf",
            chat_format="llama-2")  # Set chat_format according to the model you are using
