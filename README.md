# GenZSL



This repository contains the training code for the ICML'25 paper titled with  "***GenZSL: Generative Zero-Shot Learning Via Inductive Variational Autoencoder***".



## Requirements
The code implementation of **GenZSL** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run and test in Python 3.8.18. To install all required dependencies:
```
$ pip install -r requirements.txt
```



## Preparing Dataset

You can download the datasets, and organize them as follows: 
```
./dataset
├── data
│   ├── CUB/
│   ├── SUN/
│   └── AWA2/
└── ···
```



## Train
Runing following commands and training **GenZSL**:

Refer to scripts in `./scripts/usage.sh`



### Results
Results of our method using various evaluation protocols on three datasets, both in the conventional ZSL (CZSL) and generalized ZSL (GZSL) settings.

| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-------: | :-----: | :-----: | :-----: |
|   CUB   |   63.3    |  53.5   |  61.9   |  57.4   |
|   SUN   |   73.5    |  50.6   |  43.8   |  47.0   |
|  AWA2   |   92.2    |  86.1   |  88.7   |  87.4   |

**Note**: All of above results are run on a server with a **NVIDIA TITAN X** GPU. 
