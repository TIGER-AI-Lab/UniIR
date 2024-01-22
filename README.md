# UniIR
### This repo is under construction. Please stay tuned.

[**🌐 Homepage**](https://tiger-ai-lab.github.io/UniIR/) | [**🤗 Dataset**](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) | [**📖 arXiv**](https://arxiv.org/pdf/2311.17136.pdf) | [**GitHub**](https://github.com/TIGER-AI-Lab/UniIR)

This repo contains the codebase for the paper "[UniIR: Training and Benchmarking Universal Multimodal
Information Retrievers](https://arxiv.org/pdf/2311.17136.pdf)"

## 🔔News
- **[2024-01-21]: Refactor Codebase and Release the Preprocessing Scripts for all the datasets.**
- **🔥[2023-12-21]: Our [M-BEIR Benchmark](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) is now available for use.**


## Introduction
We propose the **UniIR**(Universal multimodal Information Retrieval) **framework** to learn a single retriever to accomplish (possibly) any retrieval task. Unlike traditional IR systems, UniIR needs to follow the instructions to take a heterogeneous query to retrieve from a heterogeneous candidate pool with millions of candidates in diverse modalities.

<img src="docs/images/uniir_teaser.jpg" alt="UniIR Teaser" style="width:80%;">


## Content

1. [M-BEIR](#M-BEIR)
2. [Training](#Training)
3. [Evaluation](#Evaluation)
2. [Model Zoo](#Model-Zoo)
4. [Citations and Contact](#Citation-and-Contact)


# M-BEIR
To train and evaluate universal multimodal retrieval models, we build a large-scale retrieval benchmark named **M-BEIR** (Multimodal BEnchmark for Instructed Retrieval).

## M-BEIR Downloading
We provide the M-BEIR dataset in the [**🤗 Dataset**](https://huggingface.co/datasets/TIGER-Lab/M-BEIR).
Please follow the instructions to download the dataset and prepare the data for training and evaluation.

# UniIR Models
We provide the codebase for training and evaluating the UniIR CLIP-ScoreFusion, CLIP-FeatureFusion, BLIP-ScoreFusion, and BLIP-FeatureFusion models.

## Training
To train the UniIR models from pretrained CLIP and BLIP checkpoints, please follow the instructions below. 
The scripts will automatically download the pretrained checkpoints.
### 1. Environment
#### UniIR CLIP_SF and CLIP_FF
```bash
# From the root directory of the repo
cd src/models/uniir_clip/
conda env create -f clip_env.yml
```
#### UniIR BLIP_SF and BLIP_FF
```bash
cd src/models/uniir_blip/
conda env create -f blip_env.yml
```
### 2. Scripts
#### UniIR CLIP_SF
```bash
cd src/models/uniir_clip/clip_scorefusion/configs_scripts/large/train/inbatch/
```
Modify `inbatch.yaml` for hyperparameter tuning and `run_inbatch.sh` for your own environment and paths.
```bash
bash run_inbatch.sh
```

#### UniIR BLIP_FF
```bash
cd src/models/uniir_blip/blip_featurefusion/configs_scripts/large/train/inbatch/
```
Modify `inbatch.yaml` for hyperparameter tuning and `run_inbatch.sh` for your own environment and paths.
```bash
bash run_inbatch.sh
```

Similarly, you can train the UniIR CLIP_FF and BLIP_SF models by modifying the corresponding scripts.

## Evaluation
We provide the evaluation pipline for the UniIR models on the M-BEIR benchmark.
### 1. Environment
```bash
# From the root directory of the repo
conda env create -f faiss_env.yml
```
### 2. Scripts
#### UniIR CLIP_SF
```bash
cd src/models/unii_clip/clip_scorefusion/configs_scripts/large/eval/inbatch/
```
Modify `embed.yaml` and `run_eval_pipeline_inbatch.sh` for your own environment and paths.
```bash
bash run_eval_pipeline_inbatch.sh
```
Similarly, you can evaluate the UniIR CLIP_FF, BLIP_SF, and BLIP_FF models by modifying the corresponding scripts.

## Model Zoo
TODO

## Citation and Contact
- Cong Wei: c58wei@uwaterloo.ca
- Yang Chen: yangc@gatech.edu
- Alan Ritter: alan.ritter@cc.gatech.edu
- Wenhu Chen: wenhuchen@uwaterloo.ca


**BibTeX:**
```bibtex
@article{wei2023uniir,
  title={UniIR: Training and Benchmarking Universal Multimodal Information Retrievers},
  author={Wei, Cong and Chen, Yang and Chen, Haonan and Hu, Hexiang and Zhang, Ge and Fu, Jie and Ritter, Alan and Chen, Wenhu},
  journal={arXiv preprint arXiv:2311.17136},
  year={2023}
}
```
