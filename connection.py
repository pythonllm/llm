from llama_cpp import Llama

llm = Llama(
    model_path="./solar-10.7b-instruct-v1.0-uncensored.Q8_0.gguf",
    n_gpu_layers=-1,  # Utilize GPU for all layers
    n_batch=512,  # Number of tokens to process in parallel
    n_threads=24,  # Utilize all vCPUs for threading
    verbose=True,  # Enable verbose logging
    max_tokens=512,
)


prompt_temp = "You are Paolina, a very attractive 20-year-old, flirty, and playful adult model female obsessed with sex. Form detailed answers in a brutal sexual manner"


output = llm(
    prompt_temp,
    max_tokens=32,  # Generate up to 512 tokens
    stop=["</s>"],  # Assuming "</s>" is a valid stop token for your model
    echo=True  # Echo the prompt in the output
)



