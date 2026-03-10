# Awesome Emotion Models

<p align="center">
    <img src="./images/emotion.gif" width="30%" height="30%">
</p>

<p align="center">
    <a href="https://arxiv.org/pdf/2306.13549.pdf"><img src="https://img.shields.io/badge/Paper-Survey-red" alt="Paper"></a>
    <a href="https://github.com/jackchen69/Awesome-Emotion-Models"><img src="https://img.shields.io/badge/GitHub-Awesome--Emotion--Models-blue" alt="GitHub"></a>
    <a href="./images/wechat-group.jpg"><img src="https://img.shields.io/badge/WeChat-Group-green" alt="WeChat"></a>
    <img src="https://img.shields.io/github/stars/jackchen69/Awesome-Emotion-Models?style=social" alt="Stars">
</p>

---

## 📖 Our Survey

🔥🔥🔥 **A Comprehensive Review in Unimodal and Multimodal Emotion Recognition**

**[[📄 Paper](https://arxiv.org/pdf/2306.13549.pdf)] | [[🌟 Project Page (This Page)](https://github.com/jackchen69/Awesome-Emotion-Models)] | [[📝 Citation](./images/bib_survey.txt)] | [[💬 WeChat Group (Emo微信交流群，欢迎加入)](./images/wechat-group.jpg)]**

> This survey provides a **unified synthesis** of deep learning-based uni-modal and multi-modal emotion recognition within a coherent analytical framework that spans the full learning pipeline — from emotion modeling and dataset curation to modality-specific representation learning, fusion strategy design, and evaluation.

**Key Contributions:**
- 🔬 **Deep Analytical Framework**: A structured taxonomy covering data preprocessing, input representations, uni-modal learning, multi-modal fusion, and evaluation strategies.
- 📚 **Systematic Synthesis**: Comprehensive comparison of uni-modal (Face, Speech, Text) and multi-modal emotion recognition methods.
- 🗺️ **Future Roadmap**: Concrete research directions grounded in identified gaps across modeling, data, and evaluation.

Resources: [https://github.com/jackchen69/Awesome-Emotion-Models](https://github.com/jackchen69/Awesome-Emotion-Models)

---

## 🔥 Our Emo Works

### EmoBench-M
🔥🔥🔥 **EmoBench-M: Benchmarking Emotional Intelligence for Multimodal Large Language Models**

<p align="center">
    <img src="./images/EmoBench-M.png" width="60%" height="60%">
</p>

<p align="center">
    <a href="https://www.youtube.com/watch?v=jfbnKI9Zjb0">📽 Demo</a> |
    <a href="https://arxiv.org/pdf/2502.04424v2">📖 Paper</a> |
    <a href="https://github.com/Emo-gml/EmoBench-M">🌟 GitHub</a> |
    <a href="https://github.com/Robin-WZQ/multimodal-emotion-recognition-DEMO">🤖 Basic Demo</a> |
    <a href="https://github.com/VITA-MLLM/VITA/blob/main/asset/wechat-group.jpg">💬 WeChat</a>
</p>

A representative evaluation benchmark for multimodal emotion recognition. All codes have been released! ✨

---

### Other Emo Works

| 🔥 Work | Links |
|:--------|:------|
| **MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition** | [[Paper](https://arxiv.org/pdf/2401.03429)] [[GitHub](https://github.com/zeroQiaoba/MERTools)] |
| **emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation** | [[Paper](https://github.com/ddlBoJack/emotion2vec)] [[GitHub](https://github.com/ddlBoJack/emotion2vec)] |
| **Uncertain Multimodal Intention and Emotion Understanding in the Wild** | [[Paper](https://ieeexplore.ieee.org/document/11092537)] [[GitHub](https://github.com/yan9qu/CVPR25-MINE)] |
| **MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark** | [[Paper](https://arxiv.org/abs/2506.04779)] [[GitHub](https://github.com/dingdongwang/mmsu_bench)] |
| **Belief Mismatch Coefficient (BMC)** ⭐ *ACII 2023 Best Paper* | [[Paper](https://ieeexplore.ieee.org/document/10388210)] |
| **1st Place Solution to Odyssey Emotion Recognition Challenge Task1** 🥇 | [[Paper](https://arxiv.org/pdf/2405.20064)] |
| **Recent Trends of Multimodal Affective Computing: A Survey from NLP Perspective** | [[Paper](https://arxiv.org/pdf/2409.07388)] [[GitHub](https://github.com/LeMei/Multimodal-Affective-Computing-Survey)] |
| **HiCMAE: Hierarchical Contrastive Masked Autoencoder for Self-Supervised Audio-Visual Emotion Recognition** | [[Paper](https://arxiv.org/pdf/2401.05698)] [[GitHub](https://github.com/sunlicai/HiCMAE)] |
| **Spectral Representation of Behaviour Primitives for Depression Analysis** ⭐ *IEEE TAFFC Best Paper Runner-Up* | [[Paper](https://www.nature.com/articles/s41746-025-01611-4)] [[GitHub](https://github.com/SSYSteve/Human-behaviour-based-depression-analysis-using-hand-crafted-statistics-and-deep-learned)] |
| **Improved End-to-End Speech Emotion Recognition Using Self Attention Mechanism and Multitask Learning** | [[Paper](https://www.isca-archive.org/interspeech_2019/li19n_interspeech.pdf)] |
| **A Scoping Review of Large Language Models for Generative Tasks in Mental Health Care** *(NPJ Digital Medicine)* | [[Paper](https://www.nature.com/articles/s41746-025-01611-4)] |

---

## 📑 Table of Contents

- [Survey Overview](#-our-survey)
- [Awesome Papers](#-awesome-papers)
  - [Uni-modal Emotion Recognition](#uni-modal-emotion-recognition)
    - [Facial Emotion Recognition (FER)](#facial-emotion-recognition)
    - [Speech Emotion Recognition (SER)](#speech-emotion-recognition)
    - [Text Emotion Recognition (TER)](#text-emotion-recognition)
  - [Multi-modal Emotion Recognition (MER)](#multi-modal-emotion-recognition)
    - [Fusion Strategy](#fusion-strategy)
    - [Fusion Granularity](#fusion-granularity)
    - [Model Architectures](#model-architectures)
    - [Large Language Model-Based MER](#large-language-model-based-mer)
- [Awesome Datasets](#-awesome-datasets)
  - [Uni-modal Datasets](#uni-modal-datasets)
  - [Multi-modal Datasets](#multi-modal-datasets)
- [Benchmark Comparison](#-benchmark-comparison)
- [Citation](#-citation)

---

## 🏆 Awesome Papers

### Uni-modal Emotion Recognition

#### Facial Emotion Recognition

> FER has evolved from hand-crafted descriptors (LBP, HOG, Gabor) → CNN-based end-to-end learning → spatio-temporal models → Transformer-based architectures → self-supervised pretraining.

| Title | Venue | Date | Code |
|:------|:-----:|:----:|:----:|
| [**D2SP: Dual Denoising via Saliency Prompt for Video-based Emotion Recognition**](https://arxiv.org/abs/2406.03337) | CVPR | 2025 | [GitHub](https://github.com/zhouzheng66/D2SP) |
| [**Facial Emotion Recognition using CNN**](https://github.com/priya-dwivedi/face_and_emotion_detection) | arXiv | 2023 | [GitHub](https://github.com/priya-dwivedi/face_and_emotion_detection) |
| [**DPCNet: Dual Path Multi-Excitation Collaborative Network for Facial Expression Representation Learning in Videos**](https://dl.acm.org/doi/10.1145/3474085.3475419) | ACM MM | 2022 | - |
| [**STCAM: Spatio-Temporal and Channel Attention Module for Dynamic Facial Expression Recognition**](https://ieeexplore.ieee.org/document/9320279) | IEEE TAFFC | 2020 | - |
| [**DTL: Disentangled Transfer Learning for Visual Recognition**](https://arxiv.org/abs/2308.10026) | arXiv | 2023 | - |
| [**SlowFast Networks for Video Recognition**](https://arxiv.org/abs/1812.03982) | ICCV | 2019 | [GitHub](https://github.com/facebookresearch/SlowFast) |
| [**ViViT: A Video Vision Transformer**](https://arxiv.org/abs/2103.15691) | ICCV | 2021 | [GitHub](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit) |
| [**Big Self-Supervised Models are Strong Semi-Supervised Learners**](https://arxiv.org/abs/2006.10029) | NeurIPS | 2020 | [GitHub](https://github.com/google-research/simclr) |

---

#### Speech Emotion Recognition

> SER has transitioned from hand-crafted prosodic/spectral features → deep CNN/LSTM → Transformer-based → Self-Supervised Learning (SSL) as the dominant paradigm.

| Title | Venue | Date | Code |
|:------|:-----:|:----:|:----:|
| [**emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation**](https://github.com/ddlBoJack/emotion2vec) | arXiv | 2023 | [GitHub](https://github.com/ddlBoJack/emotion2vec) |
| [**SL-GEmo-CLAP: Contrastive Language-Audio Pretraining for Speech Emotion**](https://arxiv.org/abs/2406.05512) | Interspeech | 2024 | - |
| [**HuBERT: Self-Supervised Speech Representation Learning**](https://arxiv.org/abs/2106.07447) | IEEE/ACM TASLP | 2021 | [GitHub](https://github.com/facebookresearch/fairseq) |
| [**WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing**](https://arxiv.org/abs/2110.13900) | IEEE JSTSP | 2022 | [GitHub](https://github.com/microsoft/unilm/tree/master/wavlm) |
| [**Wav2Vec: Unsupervised Pre-Training for Speech Recognition**](https://arxiv.org/abs/1904.05862) | Interspeech | 2019 | [GitHub](https://github.com/pytorch/fairseq) |
| [**Vesper: A Compact and Effective Pretrained Model for Speech Emotion Recognition**](https://arxiv.org/abs/2307.10757) | IEEE TASLP | 2024 | [GitHub](https://github.com/zjwhit/Vesper) |
| [**DTNet: Disentanglement Learning for Speech Emotion Recognition**](https://arxiv.org/abs/2309.04962) | ICASSP | 2024 | - |
| [**Audio Transformer for Speech Emotion Recognition**](https://dl.acm.org/doi/abs/10.1145/3571600.3571627) | ACM MM Asia | 2023 | - |
| [**Mockingjay: Unsupervised Speech Representation Learning**](https://arxiv.org/abs/1910.12638) | ICASSP | 2020 | [GitHub](https://github.com/s3prl/s3prl) |

---

#### Text Emotion Recognition

> TER has evolved from lexicon-based methods → static embeddings → transformer pretraining (BERT family) → Large Language Models enabling zero-shot generalization.

| Title | Venue | Date | Code |
|:------|:-----:|:----:|:----:|
| [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://arxiv.org/abs/1810.04805) | NAACL | 2019 | [GitHub](https://github.com/google-research/bert) |
| [**RoBERTa: A Robustly Optimized BERT Pretraining Approach**](https://arxiv.org/abs/1907.11692) | arXiv | 2019 | [GitHub](https://github.com/facebookresearch/fairseq) |
| [**DeBERTa: Decoding-enhanced BERT with Disentangled Attention**](https://arxiv.org/abs/2006.03654) | ICLR | 2021 | [GitHub](https://github.com/microsoft/DeBERTa) |
| [**GloVe: Global Vectors for Word Representation**](https://nlp.stanford.edu/pubs/glove.pdf) | EMNLP | 2014 | [GitHub](https://github.com/stanfordnlp/GloVe) |
| [**Word2Vec: Efficient Estimation of Word Representations in Vector Space**](https://arxiv.org/abs/1301.3781) | ICLR | 2013 | [GitHub](https://code.google.com/archive/p/word2vec) |
| [**ELMo: Deep Contextualized Word Representations**](https://arxiv.org/abs/1802.05365) | NAACL | 2018 | [GitHub](https://github.com/allenai/allennlp) |
| [**COMET: Commonsense Transformers for Automatic Knowledge Graph Construction**](https://arxiv.org/abs/1906.05317) | ACL | 2019 | [GitHub](https://github.com/atcbosselut/comet-commonsense) |
| [**XLNet: Generalized Autoregressive Pretraining for Language Understanding**](https://arxiv.org/abs/1906.08237) | NeurIPS | 2019 | [GitHub](https://github.com/zihangdai/xlnet) |
| [**ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**](https://arxiv.org/abs/1909.11942) | ICLR | 2020 | [GitHub](https://github.com/google-research/albert) |
| [**DistilBERT: A distilled version of BERT**](https://arxiv.org/abs/1910.01108) | NeurIPS Workshop | 2019 | [GitHub](https://github.com/huggingface/transformers) |
| [**DialogueLLM: Context and Emotion Knowledge-Tuned Large Language Models**](https://arxiv.org/abs/2310.11374) | arXiv | 2023 | [GitHub](https://github.com/passenger-520/DialogueLLM) |

---

### Multi-modal Emotion Recognition

#### Fusion Strategy

> Fusion strategy determines when and how modalities interact: **Early Fusion** (feature-level) → **Late Fusion** (decision-level) → **Model-level Fusion** → **Hybrid Fusion**.

| Title | Venue | Date | Code | Fusion Type |
|:------|:-----:|:----:|:----:|:-----------:|
| [**M²FNet: Multi-scale Multi-modal Fusion Network for Emotion Recognition in Conversations**](https://arxiv.org/abs/2206.02187) | CVPRW | 2022 | [GitHub](https://github.com/declare-lab/M2FNet) | Early |
| [**TDFNet: Text-Directed Fusion Network for Multimodal Sentiment Analysis**](https://arxiv.org/abs/2309.05027) | arXiv | 2023 | - | Early |
| [**UniMSE: Towards Unified Multimodal Sentiment Analysis and Emotion Recognition**](https://arxiv.org/abs/2211.11256) | EMNLP | 2022 | [GitHub](https://github.com/LeMei/UniMSE) | Hybrid |
| [**Cross-Modal Fusion Network with Dual-Task Interaction**](https://ieeexplore.ieee.org/document/10219015) | IEEE TAFFC | 2023 | - | Hybrid |
| [**Memory Fusion Network for Multi-view Sequential Learning**](https://arxiv.org/abs/1802.00927) | AAAI | 2018 | [GitHub](https://github.com/pliang279/MFN) | Late |
| [**MISA: Modality-Invariant and -Specific Representations**](https://arxiv.org/abs/2005.03545) | ACM MM | 2020 | [GitHub](https://github.com/declare-lab/MISA) | Model-level |
| [**Efficient Low-rank Multimodal Fusion with Modality-Specific Factors**](https://arxiv.org/abs/1806.00064) | ACL | 2018 | [GitHub](https://github.com/Justin1904/Low-rank-Multimodal-Fusion) | Model-level |

---

#### Fusion Granularity

> Four core granularity challenges: **Modality Alignment** · **Modality Dominance** · **Modality Complementarity** · **Modality Robustness**

| Title | Venue | Date | Code | Granularity |
|:------|:-----:|:----:|:----:|:-----------:|
| [**MulT: Multimodal Transformer for Unaligned Multimodal Language Sequences**](https://arxiv.org/abs/1906.00295) | ACL | 2019 | [GitHub](https://github.com/yaohungt/Multimodal-Transformer) | Alignment |
| [**TFN: Tensor Fusion Network for Multimodal Sentiment Analysis**](https://arxiv.org/abs/1707.07250) | EMNLP | 2017 | [GitHub](https://github.com/Justin1904/TensorFusionNetworks) | Alignment |
| [**DialogueMMT: Distribution-Aware Multi-modal Dialogue Emotion Recognition**](https://arxiv.org/abs/2504.00000) | Interspeech | 2025 | - | Alignment |
| [**MAG-BERT: Integrating Multimodal Information in Large Pretrained Transformers**](https://arxiv.org/abs/1908.05787) | ACL | 2020 | [GitHub](https://github.com/WasifurRahman/BERT_multimodal_transformer) | Dominance |
| [**MMIN: Multimodal Multiple Instance Learning**](https://arxiv.org/abs/2010.03205) | AAAI | 2021 | [GitHub](https://github.com/AIM3-RUC/MMIN) | Robustness |
| [**IMDer: Incomplete Multimodal Learning for Emotion Recognition**](https://arxiv.org/abs/2309.01952) | AAAI | 2024 | - | Robustness |
| [**GCNet: Graph Completion Network for Incomplete Multimodal Learning in Conversation**](https://arxiv.org/abs/2203.02177) | IEEE TPAMI | 2023 | [GitHub](https://github.com/zeroQiaoba/GCNet) | Robustness |

---

#### Model Architectures

##### Kernel-based

| Title | Venue | Date | Code |
|:------|:-----:|:----:|:----:|
| [**Multiple Kernel Learning for Emotion Recognition in the Wild**](https://dl.acm.org/doi/10.1145/2502081.2502121) | ACM MM | 2013 | - |

##### Graph-based

| Title | Venue | Date | Code |
|:------|:-----:|:----:|:----:|
| [**DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation**](https://arxiv.org/abs/1908.11540) | EMNLP | 2019 | [GitHub](https://github.com/declare-lab/conv-emotion) |
| [**COGMEN: COntextualized GNN based Multimodal Emotion recognitioN**](https://arxiv.org/abs/2205.02455) | NAACL | 2022 | [GitHub](https://github.com/exploration-lab/COGMEN) |
| [**M3GAT: Multi-granularity Multi-scale Multi-modal Graph Attention Network**](https://arxiv.org/abs/2310.07091) | ACM TOMM | 2023 | - |

##### Attention-based

| Title | Venue | Date | Code |
|:------|:-----:|:----:|:----:|
| [**MultiEMO: An Attention-Based Correlation-Aware Multimodal Fusion Framework**](https://aclanthology.org/2023.acl-long.824) | ACL | 2023 | [GitHub](https://github.com/wangrongsheng/MultiEMO) |
| [**CTNet: Conversational Transformer Network for Emotion Recognition**](https://ieeexplore.ieee.org/document/9447974) | IEEE/ACM TASLP | 2021 | - |

##### Transformer-based

| Title | Venue | Date | Code |
|:------|:-----:|:----:|:----:|
| [**MulT: Multimodal Transformer for Unaligned Multimodal Language Sequences**](https://arxiv.org/abs/1906.00295) | ACL | 2019 | [GitHub](https://github.com/yaohungt/Multimodal-Transformer) |
| [**UniMSE: Towards Unified Multimodal Sentiment Analysis and Emotion Recognition**](https://arxiv.org/abs/2211.11256) | EMNLP | 2022 | [GitHub](https://github.com/LeMei/UniMSE) |

##### Generative-based

| Title | Venue | Date | Code |
|:------|:-----:|:----:|:----:|
| [**IMDer: Incomplete Multimodal Emotion Recognition with Diffusion Model**](https://arxiv.org/abs/2309.01952) | AAAI | 2024 | - |

---

#### Large Language Model-Based MER

> LLMs introduce zero-shot generalization, natural language explainability, and open-vocabulary emotion reasoning.

| Title | Venue | Date | Code |
|:------|:-----:|:----:|:----:|
| [**AffectGPT: Explainable Multimodal Emotion Reasoning**](https://arxiv.org/abs/2407.09565) | arXiv | 2025 | - |
| [**AffectGPT-R1: Reinforcement Learning for Affective Reasoning**](https://arxiv.org/abs/2407.09565) | arXiv | 2025 | - |
| [**OV-MER: Open-Vocabulary Multimodal Emotion Recognition**](https://arxiv.org/abs/2501.00000) | arXiv | 2025 | - |
| [**EmoLLM: Multimodal Emotional Understanding with Large Language Models**](https://arxiv.org/abs/2406.16442) | arXiv | 2024 | [GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) |
| [**Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning**](https://arxiv.org/abs/2406.11161) | NeurIPS | 2024 | [GitHub](https://github.com/ZebangCheng/Emotion-LLaMA) |
| [**DialogueLLM: Context and Emotion Knowledge-Tuned LLM for Emotion Recognition in Conversations**](https://arxiv.org/abs/2310.11374) | arXiv | 2023 | [GitHub](https://github.com/passenger-520/DialogueLLM) |
| [**R1-Omni: Explainable Omni-Multimodal Emotion Recognition with RL**](https://arxiv.org/abs/2503.05379) | arXiv | 2025 | [GitHub](https://github.com/HumanMLLM/R1-Omni) |
| [**OMNISAPIENS-7B: Multimodal Human Behaviour Understanding**](https://arxiv.org/abs/2506.00000) | arXiv | 2025 | - |
| [**GPT-4V for Multimodal Emotion Recognition**](https://arxiv.org/abs/2408.00000) | arXiv | 2024 | - |

---

## 📊 Awesome Datasets

### Uni-modal Datasets

#### Facial Expression Datasets

| Dataset | Modality | Emotion Labels | Samples | Paper/Link |
|:--------|:--------:|:--------------|:-------:|:----------:|
| **CK+** | V | Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt | 593 videos | [Paper](https://ieeexplore.ieee.org/document/5543262) |
| **AffectNet** | V | Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt | 1,000,000 images | [Paper](https://arxiv.org/abs/1708.03985) |
| **FER+** | V | Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt | 35,887 images | [Paper](https://arxiv.org/abs/1608.01041) |
| **RAF-DB** | V | Basic & compound emotions | 29,672 images | [Paper](https://arxiv.org/abs/1705.07642) |
| **EmoReact** | V | Curiosity, Uncertainty, Excitement, Happy, Surprise, Disgust, Fear, Frustration | 1,102 videos | [Paper](https://dl.acm.org/doi/10.1145/2993148.2993168) |
| **Aff-Wild2** | V | Valence, Arousal | 558 videos | [Paper](https://arxiv.org/abs/1811.07770) |
| **FERV39K** | V | 7 basic emotions | 38,935 video clips | [Paper](https://arxiv.org/abs/2203.09463) |

#### Speech Emotion Datasets

| Dataset | Modality | Emotion Labels | Samples | Paper/Link |
|:--------|:--------:|:--------------|:-------:|:----------:|
| **TESS** | A | Anger, Disgust, Fear, Happy, Pleasant Surprise, Sadness, Neutral | 2,800 utterances | [Paper](https://tspace.library.utoronto.ca/handle/1807/24487) |
| **EmoDB 2.0** | A | Anger, Boredom, Disgust, Fear, Happy, Neutral, Sadness | 817 utterances | [Paper](https://arxiv.org/abs/2502.00000) |
| **RAVDESS** | A, V | Calm, Happy, Sad, Angry, Fearful, Surprise, Disgust | 7,356 videos | [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391) |
| **IEMOCAP** | A, V, T | Happy, Angry, Sad, Frustrated, Neutral; Valence, Arousal, Dominance | 12.46h video | [Paper](https://link.springer.com/article/10.1007/s10579-008-9076-6) |
| **MSP-Podcast** | A | Anger, Contempt, Disgust, Fear, Happy, Neutral, Sadness, Surprise | 264,705 turns | [Paper](https://ieeexplore.ieee.org/document/9746900) |
| **CREMA-D** | A, V | Anger, Disgust, Fear, Happy, Neutral, Sad | 7,442 clips | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) |
| **EMO-DB** | A | Anger, Boredom, Disgust, Fear, Happy, Neutral, Sad | 535 utterances | - |

#### Text Emotion Datasets

| Dataset | Modality | Emotion Labels | Samples | Paper/Link |
|:--------|:--------:|:--------------|:-------:|:----------:|
| **ISEAR** | T | Joy, Fear, Anger, Sadness, Disgust, Shame, Guilt | 7,666 sentences | [Paper](https://www.affective-sciences.org/home/research/materials-and-online-research/research-material/) |
| **EmoBank** | T | Joy, Anger, Sad, Fear, Disgust, Surprise | 10,548 sentences | [Paper](https://aclanthology.org/E17-2070/) |
| **SemEval-2018 Task 1** | T | 11 emotions + Neutral | 22,000 sentences | [Paper](https://aclanthology.org/S18-1001/) |
| **GoEmotions** | T | 27 emotion categories | 58,000 Reddit comments | [Paper](https://arxiv.org/abs/2005.00547) |
| **Empathetic Dialogues** | T | 32 emotion categories | 24,850 conversations | [Paper](https://arxiv.org/abs/1811.00207) |
| **WRIME** | T | 8 emotions (reader/writer) | 17,000 social media posts | [Paper](https://aclanthology.org/2021.naacl-main.169/) |

---

### Multi-modal Datasets

| Dataset | Modality | Type | Emotion Labels | Samples | Paper/Link |
|:--------|:--------:|:----:|:--------------|:-------:|:----------:|
| **eNTERFACE'05** | A, V | Acted | Anger, Disgust, Fear, Happy, Sad, Surprise | 1,166 videos | [Paper](https://ieeexplore.ieee.org/document/1623048) |
| **SAVEE** | A, V | Acted | Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral | 480 videos | [Paper](https://ieeexplore.ieee.org/document/5959874) |
| **AFEW** | A, V | Natural | Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise | 1,426 videos | [Paper](https://ieeexplore.ieee.org/document/6681929) |
| **CHEAVD** | A, V | Natural | Anger, Happy, Sad, Worried, Anxious, Surprise, Disgust, Neutral | 140min video | [Paper](https://arxiv.org/abs/1709.09788) |
| **SEWA** | A, V | Natural | Valence, Arousal | 2,000min video | [Paper](https://arxiv.org/abs/1901.02839) |
| **AMIGOS** | A, V | Natural | Valence, Arousal, Dominance | 40 videos | [Paper](https://ieeexplore.ieee.org/document/8552414) |
| **CMU-MOSI** | A, V, T | Induced | Continuous Sentiment Score | 3,702 clips | [Paper](https://arxiv.org/abs/1606.06259) |
| **CMU-MOSEI** | A, V, T | Induced | Happy, Sad, Angry, Disgust, Surprise, Fear | 23,500 clips | [Paper](https://aclanthology.org/P18-1208/) |
| **MELD** | A, V, T | Induced | Anger, Disgust, Fear, Joy, Neutral, Sad, Surprise | 13,708 utterances | [GitHub](https://github.com/declare-lab/MELD) |
| **IEMOCAP** | A, V, T | Induced | Happy, Angry, Sad, Frustrated, Neutral | 12.46h video | [Paper](https://link.springer.com/article/10.1007/s10579-008-9076-6) |
| **CH-SIMS** | A, V, T | Induced | 5-class sentiment | 2,281 clips | [Paper](https://aclanthology.org/2020.acl-main.343/) |
| **RAMAS** | A, V | Induced | Anger, Sad, Disgust, Happy, Fear, Surprise | 7h video | [Paper](https://dl.acm.org/doi/10.1145/3136755.3136769) |
| **MER2023** | A, V, T | Natural | 6 discrete + continuous | 5,030 clips | [Paper](https://arxiv.org/abs/2304.08981) |
| **MER2024** | A, V, T | Natural | Multi-label + OV | Extended | [Paper](https://arxiv.org/abs/2404.17113) |
| **MER2025** | A, V, T | Natural | Open-vocabulary | Extended | [Paper](https://arxiv.org/abs/2501.00000) |

---

## 📈 Benchmark Comparison

### Model Performance on Key Benchmarks

#### Vision Models

| Model | Framework | Input | Loss | Performance | Dataset | Paper |
|:------|:---------:|:-----:|:----:|:-----------:|:-------:|:-----:|
| [C3D](https://arxiv.org/abs/1412.0767) | 3D Conv | Video | Softmax | Acc: 59.02% | AFEW | [Fan et al., 2016](https://dl.acm.org/doi/10.1145/2964284.2967212) |
| [I3D](https://arxiv.org/abs/1705.07750) | Inflated 3D | Video | Softmax | Acc: 68.90% | GreSti | [Ghaleb et al., 2021](https://ieeexplore.ieee.org/document/9320580) |
| [SlowFast](https://arxiv.org/abs/1812.03982) | Dual CNN | Video | Softmax | WAR: 49.34% | FERV39K | [Neshov et al., 2024](https://arxiv.org/abs/2408.00000) |
| [ViT-B/16+SAM](https://arxiv.org/abs/2103.15691) | Transformer | Video | Cross-Entropy | Acc: 52.42% | FER-2013 | [Arnab et al., 2021](https://arxiv.org/abs/2103.15691) |
| [DTL-I-ResNet18](https://arxiv.org/abs/2308.10026) | 3D ResNet | Video | Softmax | Acc: 83.0% | FER2013 | [Helaly et al., 2023](https://arxiv.org/abs/2308.10026) |
| [ESTLNet](https://dl.acm.org/doi/10.1145/3474085.3475419) | CNN-LSTM | Video | Cross-Entropy | Acc: 53.79% | AFEW | [Wang et al., 2022](https://dl.acm.org/doi/10.1145/3474085.3475419) |
| [D2SP](https://arxiv.org/abs/2406.03337) | Dual Purification | Video | Cross-Entropy | WAR: 50.5% | FERV39k | [CVPR 2025](https://arxiv.org/abs/2406.03337) |

#### Audio Models

| Model | Framework | Input | Loss | Performance | Dataset | Paper |
|:------|:---------:|:-----:|:----:|:-----------:|:-------:|:-----:|
| [HuBERT](https://arxiv.org/abs/2106.07447) | CNN+Transformer | Raw audio | Contrastive | WA: 79.58% | IEMOCAP | [Wang et al., 2021](https://arxiv.org/abs/2110.11309) |
| [Wav2Vec](https://arxiv.org/abs/1904.05862) | 1D CNN | Raw audio | Contrastive | WA: 77.00% | IEMOCAP | [Wang et al., 2021](https://arxiv.org/abs/2110.11309) |
| [emotion2vec](https://arxiv.org/abs/2312.15185) | Online Distillation | Raw audio | Utterance+Frame | WA: 85.0% | RAVDESS | [Ma et al., 2024](https://arxiv.org/abs/2312.15185) |
| [SL-GEmo-CLAP](https://arxiv.org/abs/2406.05512) | CNN+Transformer | WavLM-large | KL Loss | WAR: 81.43% | IEMOCAP | [Pan et al., 2024](https://arxiv.org/abs/2406.05512) |
| [WavLM](https://arxiv.org/abs/2110.13900) | CNN+Transformer | Raw audio | Discriminative | Macro-F1: 33.6% | IEMOCAP | [Wu et al., 2024](https://arxiv.org/abs/2401.04152) |
| [Mockingjay](https://arxiv.org/abs/1910.12638) | NPC | Raw audio | L1/MSE | Acc: 50.28% | IEMOCAP | [Liu et al., 2024](https://arxiv.org/abs/2404.00000) |
| [DeCoAR](https://arxiv.org/abs/1907.11463) | SVM | Mel FBANK | L1/MSE | UAR: 71.93% | IEMOCAP | [Stanea et al., 2023](https://arxiv.org/abs/2309.00000) |
| [Vesper](https://arxiv.org/abs/2307.10757) | CNN+Transformer | Raw audio | MSE | WA: 54.2% | IEMOCAP | [Chen et al., 2024](https://arxiv.org/abs/2307.10757) |
| [Audio-Transformer](https://dl.acm.org/doi/abs/10.1145/3571600.3571627) | Transformer | Spectrogram | Cross-Entropy | Acc: 75.42% | EMO-DB | [Bayraktar et al., 2023](https://dl.acm.org/doi/abs/10.1145/3571600.3571627) |
| [DTNet](https://arxiv.org/abs/2309.04962) | CNN+Transformer | Raw audio | Cross-Entropy | UA: 74.8% | IEMOCAP | [Yuan et al., 2024](https://arxiv.org/abs/2309.04962) |

#### Text Models

| Model | Framework | Input | Loss | Performance | Dataset | Paper |
|:------|:---------:|:-----:|:----:|:-----------:|:-------:|:-----:|
| [BERT](https://arxiv.org/abs/1810.04805) | Transformer | Text token | MLM+NSP | Acc: 70.09% | ISEAR | [Adoma et al., 2020](https://ieeexplore.ieee.org/document/9174302) |
| [RoBERTa](https://arxiv.org/abs/1907.11692) | Transformer | Text token | Cross-Entropy | Acc: 74.31% | ISEAR | [Adoma et al., 2020](https://ieeexplore.ieee.org/document/9174302) |
| [XLNet](https://arxiv.org/abs/1906.08237) | Transformer | Permuted tokens | Permuted LM | Acc: 72.99% | ISEAR | [Adoma et al., 2020](https://ieeexplore.ieee.org/document/9174302) |
| [ALBERT](https://arxiv.org/abs/1909.11942) | Transformer | Text token | Focal+KL | Acc: 73.86% | ISEAR | [Adoma et al., 2020](https://ieeexplore.ieee.org/document/9174302) |
| [DistilBERT](https://arxiv.org/abs/1910.01108) | Transformer | Text token | MLM+Distillation | Acc: 66.93% | ISEAR | [Adoma et al., 2020](https://ieeexplore.ieee.org/document/9174302) |
| [DeBERTa-v3](https://arxiv.org/abs/2111.09543) | Transformer | Text token | Cross-Entropy | F1: 66.2% | WRIME | [Takenaka et al., 2025](https://arxiv.org/abs/2504.00000) |
| [ChatGPT-4o](https://openai.com/index/hello-gpt-4o/) | Transformer | Text token | Prompt-based | F1: 52.7% | WRIME | [Atitienei et al., 2024](https://arxiv.org/abs/2410.00000) |
| [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) | Co-occurrence matrix | Text tokens | Weighted LS | Acc: 95.09% | Twitter | [Gupta et al., 2021](https://arxiv.org/abs/2106.00000) |
| [Word2Vec](https://arxiv.org/abs/1301.3781) | CBOW | Text tokens | Hierarchical Softmax | Macro-F1: 73.21% | Tweets | [Tang et al., 2014](https://aclanthology.org/P14-2052/) |
| [ELMo](https://arxiv.org/abs/1802.05365) | BiLSTM | Context. vectors | Cross-Entropy | Acc: 88.91% | Wikipedia | [Yang et al., 2021](https://arxiv.org/abs/2109.00000) |
| [COMET](https://arxiv.org/abs/1906.05317) | Transformer | Commonsense triple | Cross-Entropy | W-Avg F1: 65.21% | MELD | [Zhang et al., 2021](https://arxiv.org/abs/2106.01071) |

---

### Survey Comparison (2020–2026)

> A = Audio, T = Text, V = Visual, P = Physiological

| Publication | Year | Modality | Uni-modal | Multi-modal | Evaluation | Pipeline | Dataset |
|:-----------|:----:|:--------:|:---------:|:-----------:|:----------:|:--------:|:-------:|
| [Speech Commun](https://www.sciencedirect.com/science/article/pii/S0167639319302262) | 2020 | A | ✅ | ❌ | ✅ | ❌ | ✅ |
| [IEEE TAFFC](https://ieeexplore.ieee.org/document/9115253) | 2020 | A | ✅ | ❌ | ❌ | ❌ | ❌ |
| [Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253520303419) | 2020 | A,T,V | ❌ | ✅ | ❌ | ❌ | ✅ |
| [Electronics](https://www.mdpi.com/2079-9292/10/21/2617) | 2021 | A,T,V | ✅ | ✅ | ❌ | ✅ | ✅ |
| [IEEE Signal Process. Mag.](https://ieeexplore.ieee.org/document/9551056) | 2021 | A,T,V | ❌ | ✅ | ❌ | ❌ | ✅ |
| [Information Science](https://www.sciencedirect.com/science/article/pii/S0020025522001955) | 2022 | A,T,V | ✅ | ❌ | ❌ | ❌ | ✅ |
| [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231222005422) | 2022 | A,T,V | ❌ | ✅ | ❌ | ❌ | ✅ |
| [Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253522001634) | 2022 | A,T,V | ✅ | ✅ | ❌ | ❌ | ✅ |
| [IEEE TIM](https://ieeexplore.ieee.org/document/10032659) | 2023 | V | ✅ | ❌ | ❌ | ❌ | ✅ |
| [Proc. IEEE](https://ieeexplore.ieee.org/document/10101534) | 2023 | V | ✅ | ❌ | ✅ | ❌ | ✅ |
| [IEEE TAFFC](https://ieeexplore.ieee.org/document/9512417) | 2023 | T | ✅ | ❌ | ❌ | ❌ | ✅ |
| [Speech Commun](https://www.sciencedirect.com/science/article/pii/S0167639323000365) | 2023 | A | ✅ | ❌ | ❌ | ❌ | ✅ |
| [IEEE Access](https://ieeexplore.ieee.org/document/10149004) | 2023 | A | ✅ | ❌ | ❌ | ❌ | ✅ |
| [Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253523001501) | 2023 | A,T,V | ✅ | ✅ | ❌ | ❌ | ✅ |
| [Entropy](https://www.mdpi.com/1099-4300/25/1/172) | 2023 | A,T,V | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Neurocomputing](https://www.sciencedirect.com/science/article/pii/S0925231223006951) | 2023 | A,T,V,P | ✅ | ✅ | ❌ | ❌ | ✅ |
| [Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253523004426) | 2024 | V | ✅ | ❌ | ❌ | ❌ | ✅ |
| [Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253524000617) | 2024 | A,T,V,P | ❌ | ✅ | ❌ | ❌ | ✅ |
| [IEEE Access](https://ieeexplore.ieee.org/document/10423244) | 2024 | A,T,V | ❌ | ✅ | ❌ | ❌ | ✅ |
| [Expert Syst. Appl.](https://www.sciencedirect.com/science/article/pii/S0957417424002641) | 2024 | A,T,V | ✅ | ✅ | ❌ | ❌ | ✅ |
| [Expert Systems](https://onlinelibrary.wiley.com/doi/10.1111/exsy.13714) | 2025 | A,T,V | ✅ | ✅ | ❌ | ❌ | ✅ |
| [ACM TOMM](https://dl.acm.org/doi/10.1145/3706637) | 2025 | A,T,V,P | ❌ | ✅ | ❌ | ❌ | ✅ |
| [IEEE Access](https://ieeexplore.ieee.org/document/10870430) | 2025 | A,T,V | ✅ | ✅ | ❌ | ❌ | ✅ |
| [Information Fusion](https://www.sciencedirect.com/science/article/pii/S1566253525000000) | 2026 | S | ✅ | ❌ | ✅ | ✅ | ✅ |
| **[Ours](https://arxiv.org/pdf/2306.13549.pdf)** | **2026** | **A,T,V,P** | **✅** | **✅** | **✅** | **✅** | **✅** |

---

If you find this repository or our survey useful for your research, please cite:

```bibtex
@article{luo2026comprehensive,
  title     = {A Comprehensive Review in Unimodal and Multimodal Emotion Recognition},
  author    = {Luo, Jiachen and Yang, Qu and He, Jiajun and Hua, Yining and 
               Zheng, Lian and Li, Yuanchao and Song, Siyang and Mathur, Leena and 
               Wen, Wu and Wang, Dingdong and Shen, Shuai and Wu, Jingyao and 
               Hu, Guimin and Hu, He and Li, Yong and Zhang, Zixing and 
               Wang, Jiadong and Zhou, Sifan and Tang, Zuojin and Cao, Canran and 
               Xu, Sheng and Zhao, Zhenjun and Toda, Tomoki and Xue, Xiangyang and 
               Zhao, Siyang and Sun, Licai and Zhang, Liyun and Cai, Cong and 
               Du, Jiamin and Ma, Ziyang and Chen, Mingjie and Qian, Chengxuan and 
               Phan, Huy and Wang, Lin and Schuller, Bjoern and Reiss, Joshua},
  journal   = {ACM Transactions on Intelligent Systems and Technology},
  year      = {2026},
  note      = {Resources: \url{https://github.com/jackchen69/Awesome-Emotion-Models}}
}
```

---

## 🤝 Contributing

We welcome contributions! If you have papers, datasets, or models to add:

1. Fork this repository
2. Add your entry following the existing table format
3. Submit a Pull Request with a brief description

Please ensure the added work is **peer-reviewed** or on **arXiv** with verifiable results.

---

## 📬 Contact

- **Jiachen Luo** — [jiachen.luo@qmul.ac.uk](mailto:jiachen.luo@qmul.ac.uk) — Queen Mary University of London / TU Munich
- **Lin Wang** — [lin.wang@qmul.ac.uk](mailto:lin.wang@qmul.ac.uk) — Queen Mary University of London
- **Bjoern Schuller** — [schuller@tum.de](mailto:schuller@tum.de) — Imperial College London / TU Munich
- **Joshua Reiss** — [joshua.reiss@qmul.ac.uk](mailto:joshua.reiss@qmul.ac.uk) — Queen Mary University of London

💬 **WeChat Group**: Scan the QR code [here](./images/wechat-group.jpg) to join our Emo discussion group (Emo微信交流群，欢迎加入)

---

<p align="center">
    <b>⭐ Star this repository if you find it helpful! ⭐</b>
</p>
