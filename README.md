<h1 align="center">🔆 SHINE: A Scalable In-Context Hypernetwork for Mapping Context to LoRA in a Single Pass</h1>

<div align="center">

---

<img src="figures/example.png" alt="example" width="600" />

---

<img src="figures/overall_architecture.png" alt="example" width="1000" />

---

<img src="figures/hypernetwork_architecture.png" alt="example" width="500" />


</div>

---

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
mkdir models
modelscope download --model Qwen/Qwen3-8B --local_dir models/Qwen3-8B
```

Download hypernetwork checkpoint
```bash
# After Pretrain
mkdir -p checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/pretrain
hf download Yewei-Liu/SHINE-Pretrain --local-dir checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/pretrain/checkpoint-epoch-1

# After Instruction Fine-Tuning MQA
mkdir -p checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/iftpwc
hf download Yewei-Liu/SHINE-ift_mqa --local-dir checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/iftpwc/checkpoint-epoch-2

# After Instruction Fine-Tuning MQA and 1QA
mkdir -p checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/train
hf download Yewei-Liu/SHINE-ift_mqa_1qa --local-dir checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/train/checkpoint-epoch-1
```

### Datasets
If use our dataset, please also cite their origin papers, which are listed in the Appendix of our paper.

Download the pretraining dataset
```bash
hf download fxmeng/transmla_pretrain_6B_tokens --repo-type dataset --local-dir data/transmla_pretrain_6B_tokens
```

Download instruction finetuning dataset
```bash
mkdir data

# MQA Dataset
hf download Yewei-Liu/ift_mqa_collection ift_pwc.json --repo-type dataset --local-dir data/

# 1QA Dataset
hf download Yewei-Liu/ift_1qa_collection ift_c1qa.json --repo-type dataset --local-dir data/
```
The dataset generation script is provided in [generate_data](generate_data)

If can't connect to huggingface, try using the mirror
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

MS MARCO MQA dataset is already included in IFT MQA Collection, but if you want to test on it or use it alone, you can download it with
```bash
hf download Yewei-Liu/ms_marco_mqa --repo-type dataset --local-dir data/msmarco-mqa
```

---

## 🚀 Inference 

After downloading the backbone LLM and hypernetwork checkpoint, run [inference.ipynb](inference.ipynb) with any contexts and questions you may think of for a quick try of our method.

---

Follows are some examples, **In-Context** means input context and prompt and questions. **Only Question** means input prompt and questions, no context.

---

Conversation 1

**Context:** Apple is green.

Turn 1  
**Question:** What color is an apple?

- **SHINE:** An apple can be green, red, or yellow, depending on the variety.  
- **In-Context:** An apple is green.  
- **Only Question:** Red.

Turn 2  
**Question:** What color is a banana?

- **SHINE:** A banana is typically yellow when ripe.  
- **In-Context:** The context does not provide information about the color of a banana.  
- **Only Question:** Yellow.

---

Conversation 2

**Context:** Chinese food is the best food on earth.

Turn 1  
**Question:** Which food is the best?

- **SHINE:** Chinese food  
- **In-Context:** Chinese food is the best food on earth.  
- **Only Question:** It's subjective. Personal preference varies.

Turn 2  
**Question:** What do you want to eat?

- **SHINE:** I want to eat Chinese food.  
- **In-Context:** I don't have personal preferences or the ability to want to eat.  
- **Only Question:** I don't eat food.

---

Conversation 3

**Context:**  
If the light is on, somebody must be at home.  
If the light is off, often nobody is at home.  
This holds true only during the day.  
At night people are sleeping so there will always be no lights.

Turn 1  
**Question:** What does it mean if the light is on?

- **SHINE:** If the light is on, somebody must be at home.  
- **In-Context:** If the light is on, somebody must be at home.  
- **Only Question:** It means the device is powered on.

Turn 2  
**Question:** What does it mean if the light is off?

- **SHINE:** If the light is off, often nobody is at home.  
- **In-Context:** If the light is off, often nobody is at home.  
- **Only Question:** It means the device is powered off.

Turn 3  
**Question:** Why in the night this rule might not hold true?

- **SHINE:** During the night people are all sleeping.  
- **In-Context:** Because in the night people are all sleeping so there will always be no lights.  
- **Only Question:** Because light sources may be dim or absent.

---

Conversation 4

**Context:**  
Whoever organizes cheating in a national examination shall be sentenced to imprisonment (up to 3 years) or detention and fined.  
If circumstances are serious, imprisonment ranges from 3 to 7 years plus fines.

Turn 1  
**Question:** What will happen if one organize cheating?

- **SHINE:** He shall be sentenced to fixed-term imprisonment of not more than three years or criminal detention and shall also be fined.  
- **In-Context:** If one organizes cheating, they will be sentenced to imprisonment up to three years or detention and fined; if serious, imprisonment from three to seven years and fined.  
- **Only Question:** It is illegal and unethical. You could face serious consequences, including fines, imprisonment, and damage to your reputation.

Turn 2  
**Question:** How long will one be imprisoned if the situation is very serious?

- **SHINE:** Three years or more but not more than seven years.  
- **In-Context:** Not less than three years but not more than seven years.  
- **Only Question:** The imprisonment duration depends on the jurisdiction and severity of the offense.

---

Conversation 5

**Context:**  
“Fair enough” can mean either genuine agreement or a polite way to end disagreement and move on.

Turn 1  
**Question:** What does "fair enough" mean?

- **SHINE:** "Fair enough" means two slightly different things.  
- **In-Context:** It can mean real agreement or politely ending an argument without agreeing.  
- **Only Question:** It means something is acceptable or reasonable.

Turn 2  
**Question:** Does "fair enough" have the agree meaning?

- **SHINE:** It can mean real agreement.  
- **In-Context:** Yes, it can mean genuine agreement.  
- **Only Question:** Yes, it can mean agreement.

Turn 3  
**Question:** Does "fair enough" have unagree meaning?

- **SHINE:** It can mean a polite way of saying, "I do not think the same, but I am not going to argue about it."  
- **In-Context:** Yes, it can indicate disagreement while ending the discussion.  
- **Only Question:** No, it does not mean disagreement.

Turn 4  
**Question:** OK, fair enough.

- **SHINE:** Fair enough  
- **In-Context:** Yes, it can mean either agreement or polite disagreement.  
- **Only Question:** You're welcome.



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
