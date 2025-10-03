Env
```
conda activate metalora
```
Including 
```
conda create -n metalora python==3.12
conda activate metalora
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install accelerate huggingface modelscope transformers datasets scikit-learn hydra-core tensorboard
```

Download Model
```
modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
```

Download dataset
```
export HF_ENDPOINT=https://hf-mirror.com
hf download --resume-download fxmeng/transmla_pretrain_6B_tokens --repo-type dataset --local-dir data/transmla_pretrain_6B_tokens
```