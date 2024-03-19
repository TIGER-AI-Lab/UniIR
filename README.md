# UniIR
### This repo is under construction. Please stay tuned.

[**🌐 Homepage**](https://tiger-ai-lab.github.io/UniIR/) | [**🤗 Dataset(M-BEIR Benchmark)**](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) | [**🤗 Checkpoints(UniIR models)**](https://huggingface.co/TIGER-Lab/UniIR) | [**📖 arXiv**](https://arxiv.org/pdf/2311.17136.pdf) | [**GitHub**](https://github.com/TIGER-AI-Lab/UniIR)

This repo contains the codebase for the paper "[UniIR: Training and Benchmarking Universal Multimodal
Information Retrievers](https://arxiv.org/pdf/2311.17136.pdf)"

## 🔔News
- **🔥[2024-03-18]: Release the UniIR(CLIP_SF) large and UniIR(BLIP_FF) large checkpoints [**🤗 Checkpoints**](https://huggingface.co/TIGER-Lab/UniIR)**
- **[2024-01-21]: Release the Preprocessing Scripts for all the datasets.**
- **🔥[2023-12-21]: Our [🤗 M-BEIR Benchmark](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) is now available for use.**


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
The scripts will automatically download the pretrained CLIP and BLIP checkpoints.
### 1. Environment
#### UniIR Env
```bash
cd src/models/
conda env create -f uniir_env.yml
```
### 2. Scripts
#### To train UniIR CLIP_SF Large with the default configuration:
```bash
cd src/models/uniir_clip/clip_scorefusion/configs_scripts/large/train/inbatch/
```
Modify `inbatch.yaml` for hyperparameter tuning and `run_inbatch.sh` for your own environment and paths.
```bash
bash run_inbatch.sh
```

#### To train UniIR BLIP_FF Large with the default configuration:
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
You need to create an environment for the FAISS library.
```bash
# From the root directory of the project
cd src/common/
conda env create -f faiss_env.yml
```

### 2. Download the UniIR Checkpoints
You can train the UniIR models from scratch or download the pretrained UniIR checkpoints by following the instructions in the [**Model Zoo**](#Model-Zoo) section.

### 3. Scripts
#### UniIR CLIP_SF
```bash
cd src/models/unii_clip/clip_scorefusion/configs_scripts/large/eval/inbatch/
```
Modify `embed.yaml`, `index.yaml`, `retrieval.yaml` and `run_eval_pipeline_inbatch.sh` for your own environment, paths and evaluation settings.

If you download our pretrained UniIR model, you first need to modify the ```UNIIR_DIR``` in the `run_eval_pipeline_inbatch.sh` to the directory where you
want to store large files including the checkpoints, embeddings, index and retrieval results.
Then you can place the ```clip_sf_large.pth``` file in the following path:
```bash
$UNIIR_DIR/checkpoint/CLIP_SF/Large/Instruct/InBatch/clip_sf_large.pth
```

The default configuration will evaluate the UniIR CLIP_SF Large model on  both the M-BEIR (5.6M heterogeneous candidate pool) and the M-BEIR_local (homogeneous candidate pool) benchmarks.
```UNION``` in the yaml files refers to the M-BEIR (5.6M heterogeneous candidate pool).
You can follow the comments in the yaml files and modify the configurations to evaluate the model on the M-BEIR_local benchmark only.
```bash
bash run_eval_pipeline_inbatch.sh
```
```embed```, ```index```, ```logger``` and ```retrieval_results``` will be saved in the ```$UNIIR_DIR``` directory.

#### UniIR BLIP_FF
```bash
cd src/models/unii_blip/blip_featurefusion/configs_scripts/large/eval/inbatch/
```
Similarly, if you download our pretrained UniIR model, you can place the ```blip_ff_large.pth``` file in the following path:
```bash
$UNIIR_DIR/checkpoint/BLIP_FF/Large/Instruct/InBatch/blip_ff_large.pth
```

The default configuration will evaluate the UniIR BLIP_FF Large model on both the M-BEIR and the M-BEIR_local benchmarks.
```bash
bash run_eval_pipeline_inbatch.sh
```
Similarly, you can evaluate the UniIR CLIP_FF, BLIP_SF, and BLIP_FF models by modifying the corresponding scripts.

## Model Zoo
We provide the UniIR model checkpoints in the [**🤗 Checkpoints**](https://huggingface.co/TIGER-Lab/UniIR).
You can directly use the checkpoints for retrieval tasks or fine-tune the models for your own retrieval tasks.

### Available Checkpoints

| Model Name     | Version | Model Size | Model Link                                                                                  |
|----------------|---------|------------|---------------------------------------------------------------------------------------------|
| UniIR(CLIP-SF) | Large   | 5.13 GB    | [Download Link](https://huggingface.co/TIGER-Lab/UniIR/blob/main/CLIP_SF/clip_sf_large.pth) |
| UniIR(BLIP-FF) | Large   | 7.49 GB    | [Download Link](https://huggingface.co/TIGER-Lab/UniIR/blob/main/BLIP_FF/blip_ff_large.pth) |

You can download them by 
```
git clone https://huggingface.co/TIGER-Lab/UniIR
```


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
