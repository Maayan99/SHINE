# 🔆 SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass

<div align="center">

An example
[example](figures/example.pdf)

Overall Architecture
[overall architecture](figures/overall_architecture.pdf)

Hypernetwork Architecture
[hypernetwork architecture](figures/hypernetwork_architecture.pdf)


</div>


<!-- <div align="center">

这里放置徽章，例如 arXiv 链接、许可证、Python 版本等
[![arXiv](https://img.shields.io/badge/arXiv-[Paper ID]-b31b1b.svg)](https://arxiv.org/abs/[Paper ID])
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

</div> -->

<!-- ## ✨ Features

- 🔥 **[核心特性 1]**: [简短描述，例如：High efficiency implementation...]
- 🧠 **[核心特性 2]**: [简短描述，例如：Context-aware mechanism...]
- 🎯 **[核心特性 3]**: [简短描述，例如：State-of-the-art performance on...]
- ⚡ **[核心特性 4]**: [简短描述，例如：Easy integration with existing pipelines...] -->

<!-- ## 🎯 What is [Project Name]?

<div align="center">
  <!-- 替换为你的架构图或演示图 -->
  <!-- <img src="docs/framework.jpg" alt="Framework Overview" width="600"/>
</div> -->
<!-- 
[Project Name] is a framework for [简述项目的主要功能和目标]. It addresses the challenge of [描述解决的问题] by [描述你的方法/技术手段].

Compared to conventional solutions:

- **vs Method A**: [描述对比优势，例如：More efficient memory usage.]
- **vs Method B**: [描述对比优势，例如：Better accuracy without retraining.]
- It supports [列举支持的任务或场景]. -->

## ⚡ Quick Start
First clone this repo and cd into it
```bash
git clone <repo_name>
cd SHINE
```

### Environment
Create the conda env using the following commands
```bash
conda create -n shine python==3.12 -y
conda activate shine
# Change the pytorch version based on your device
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install huggingface==0.0.1 modelscope==1.31.0 transformers==4.57.1 datasets==4.4.1 scikit-learn==1.7.2 hydra-core==1.3.2 tensorboard==2.20.0 openai==2.6.1 rouge==1.0.1 seaborn==0.13.2 matplotlib==3.10.7 multiprocess==0.70.16
```

### Models
Backbone LLM can be download directly from modelscope
```bash
modelscope download --model Qwen/Qwen3-8B --local_dir models/Qwen3-8B
```

Download hypernetwork checkpoint
```bash
# Release soon
```

### Datasets
Download the pretraining dataset
```bash
hf download fxmeng/transmla_pretrain_6B_tokens --repo-type dataset --local-dir data/transmla_pretrain_6B_tokens
```

Download instruction finetuning dataset
```bash
#Release soon
```
The dataset generation script is provided in [generate_data](generate_data)

If can't connect to huggingface, try using the mirror
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 🚀 Inference 



## ⚙️ Training

Pretrain
```bash
sh scripts/Qwen3-8B/pretrain_group_4layer_8lora_128metalora.sh
```
Test code is provided in [test_pretrain.py](text_pretrain.py)

Instruction Fine-Tuning MQA
```bash
sh scripts/Qwen3-8B/meta_train_parallel_ift_pwc_4layer_8lora_128metalora.sh
```
Test code is provided in [test_pwc.py](test_pwc.py)

Instruction Fine-Tuning 1QA
```bash
sh scripts/Qwen3-8B/meta_train_parallel_ift_c1qa_4layer_8lora_128metalora.sh
```
Test code is provided in [test.py](test.py)


<!-- ## 📖 Citation

# If you find this work useful, please cite our paper:

# ```bibtex
# @inproceedings{
# chen2025generative,
# title={Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass},
# author={Tong Chen and Hao Fang and Patrick Xia and Xiaodong Liu and Benjamin Van Durme and Luke Zettlemoyer and Jianfeng Gao and Hao Cheng},
# booktitle={The Thirteenth International Conference on Learning Representations},
# year={2025},
# url={https://openreview.net/forum?id=bc3sUsS6ck}
# }
# ``` -->
