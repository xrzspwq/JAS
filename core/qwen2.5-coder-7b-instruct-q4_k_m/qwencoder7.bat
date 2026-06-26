@REM curl -L -o qwen2.5-coder-7b-instruct-q4_k_m.gguf "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf?download=true"

@REM echo FROM ./qwen2.5-coder-7b-instruct-q4_k_m.gguf > qwencoder7.Modelfile

ollama create qwencoder7 -f qwencoder7.Modelfile
