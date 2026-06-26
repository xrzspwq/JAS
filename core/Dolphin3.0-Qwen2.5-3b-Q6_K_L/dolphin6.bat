@REM curl -L -o Dolphin3.0-Qwen2.5-3b-Q6_K_L.gguf "https://huggingface.co/bartowski/Dolphin3.0-Qwen2.5-3b-GGUF/resolve/main/Dolphin3.0-Qwen2.5-3b-Q6_K_L.gguf"

@REM echo FROM ./Dolphin3.0-Qwen2.5-3b-Q6_K_L.gguf > dolphin6.Modelfile

ollama create dolphin6 -f dolphin6.Modelfile
