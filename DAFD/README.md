# DAFD - Domain Adaptation via Feature Disentanglement

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.8%2B-orange)  
## Abstract

In many real-world scenarios, image classifiers are applied to domains that differ from the original training data. This poses the challenging problem of domain shift, which significantly reduces the classification accuracy. To tackle this issue, unsupervised domain adaptation (UDA) techniques have been developed to bridge the gap between source and target domains by transferring knowledge from a labeled source domain to an unlabeled target domain. We propose a novel and effective coarse-to-fine domain adaptation method called Domain Adaptation via Feature Disentanglement (DAFD). This approach incorporates two key components: first, our Class-Relevant Feature Selection module CRFS disentangles the features that are relevant for determining the correct class from the class-irrelevant features. As a result, the CRFS module prevents the network from overfitting to irrelevant data and enhances its focus on crucial information for accurate classification. This reduces the complexity of domain alignment, leading to an improved classification accuracy on the target domain. Second, our Dynamic Local Maximum Mean Discrepancy module DLMMD achieves a fine-grained feature alignment by minimizing the discrepancy among class-relevant features from different domains. The alignment process now becomes more adaptive and contextually sensitive, enhancing ability of the model to recognize domain-specific patterns and characteristics. The combination of the CRFS and DLMMD modules results in an effective alignment of class-relevant features. The knowledge is successfully transferred from the source to the target domain. We conduct comprehensive experiments on four standard datasets. Our results demonstrate that DAFD is very robust and effective in domain adaptive image classification tasks and superior compared to the state-of-the-art.


## Key Features

DAFD comprises two key components:

1. **Class-relevant Feature Selection (CRFS)**: This module distills the discriminative features essential for accurate classification. It focuses on foreground features while excluding irrelevant background features that may hinder the domain adaptation process.

2. **Dynamic Local Maximum Mean Discrepancy (DLMMD)**: This module reduces the discrepancy between different domains at a fine-grained level. It aligns the identified features, allowing the model to adapt to diverse contexts and domain-specific characteristics.

## Installation

You can either git clone this whole repo by:

```
https://github.com/anonymous102410/DAFD.git
cd DAFD
pip install -r requirements.txt
```

## Datasets
### Office-31
- [Azure (supports wget)](https://wjdcloud.blob.core.windows.net/dataset/OFFICE31.zip)
### Office-Home
- [Azure (Supports wget)](https://wjdcloud.blob.core.windows.net/dataset/OfficeHome.zip)
### ImageCLEF
- [Azure (Supports wget)](https://wjdcloud.blob.core.windows.net/dataset/image_CLEF.zip)
### VisDA-17
- [Download the VisDA-classification dataset](http://csr.bu.edu/ftp/visda17/clf/)

## Pretraining Parameter
https://pan.baidu.com/s/1BGw1MJSv436FDnVnodAJHw 提取码：oxtv

## Run
```
1. cd DAFD
2. cp 'modelparameters' folder into 'DAFD' folder
3. sh sh/xx_sh/DAFD.sh
```

## Function of each file
mian.py - Project entry<br />
transfer_Losses.py - choose loss funtions<br />
utils.py - some utils<br />
models.py - create model<br />
data_loader.py - load data<br />
backbones.py - create backbone<br />
loss_funcs - loss funcs<br />
sh - start-up project<br />
