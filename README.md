Env
```
conda activate metalora
```
Including 
```
conda create -n metalora python==3.12 -y
conda activate metalora
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 change based on your cuda version
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install huggingface modelscope transformers datasets scikit-learn hydra-core tensorboard openai rouge seaborn matplotlib
```

Download Model
```
modelscope download --model Qwen/Qwen3-8B --local_dir models/Qwen3-8B
```

Download dataset
```
export HF_ENDPOINT=https://hf-mirror.com
hf download fxmeng/transmla_pretrain_6B_tokens --repo-type dataset --local-dir data/transmla_pretrain_6B_tokens
hf download bigai-nlco/LooGLE --repo-type dataset --local-dir data/loogle
hf download rajpurkar/squad --repo-type dataset --local-dir data/squad
hf download ArmelR/the-pile-splitted --repo-type dataset --local-dir data/Pile
```