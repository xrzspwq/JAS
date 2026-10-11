Just Artificial Stupidity

pip install -U huggingface_hub hf_transfer
python jas_download_models.py --list                 # preview, download nothing
python jas_download_models.py                        # download everything
python jas_download_models.py --quality high         # Q5_K_M for the 30B models (~22 GB each)
python jas_download_models.py --only fast main       # subset
python jas_download_models.py --with-optional        # also gpt-oss-20b
python jas_download_models.py --dir D:/jas/models --yes

llama-swap --config models\llama-swap.config.yaml --listen 127.0.0.1:8080