@REM curl -L -o qwen2.5-coder-3b-instruct-q8_0.gguf "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q8_0.gguf?download=true"

@REM echo FROM ./qwen2.5-coder-3b-instruct-q8_0.gguf > qwencoder.Modelfile

ollama create qwencoder -f qwencoder.Modelfile
