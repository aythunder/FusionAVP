# FusionAVP
The code and dataset of FusionAVP are available here for academic exchange and learning.
# Preparation of the Python Environment
- Python=3.9,numpy=1.26.4,pandas=2.2.2
- torch=1.12.0+cu116,torchaudio=0.12.0+cu116,torchvision=0.13.0+cu116
- transformers=4.41.2,scikit-learn=1.5.0
- All experiments were conducted on an NVIDIA A100 GPU with CUDA version 12.7.
# Installation:
1.Download the source code from this repository.
2.Download the pretrained protein language model from huggingface.For example: https://huggingface.co/facebook/esm1v_t33_650M_UR90S_1/tree/main
# Runing
- [AVP_LLM_feature.py]: We provide code here for extracting features from ESM and BERT-based large-scale models, which can be modified as needed.
- [AVP_manual_feature.py]: We provide code here for extracting conventional features, including AAindex, one-hot encoding, and BLOSUM62, which can be modified as needed.
- [AVP_LLM_feature.py]: We built the model architecture using PyTorch and used PeptideDataset for data loading.
- [AVP_LLM_feature.py]: We provide code for evaluating the performance of the model.
