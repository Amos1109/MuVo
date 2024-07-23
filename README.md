# A multi-view consistency framework with semi-supervised domain adaptation

The official Pytorch implementation of "A multi-view consistency framework with semi-supervised domain adaptation" accepted by *Engineering Applications of Artificial Intelligence*.

## Introduction

Semi-Supervised Domain Adaptation (SSDA) leverages knowledge from a fully labeled source domain to classify data in a partially labeled target domain. Due to the limited number of labeled samples in the target domain, there can be intrinsic similarity of classes in the feature space, which may result in biased predictions, even when the model is trained on a balanced dataset. To overcome this limitation, we introduce a multi-view consistency framework, which includes two views for training strongly augmented data. One is a debiasing strategy for correcting class-wise prediction probabilities according to the prediction performance of the model. The other involves leveraging pseudo-negative labels derived from the model predictions. Furthermore, we introduce a cross-domain affinity learning aimed at aligning features of the same class across different domains, thereby enhancing overall performance. Experimental results demonstrate that our method outperforms the competing methods on two standard domain adaptation datasets, DomainNet and Office–Home. Combining unsupervised domain adaptation and semi-supervised learning offers indispensable contributions to the industrial sector by enhancing model adaptability, reducing annotation costs, and improving performance.

## Data Preparation

### Supported Datasets

Currently, we support the following datasets:

- [DomainNet](http://ai.bu.edu/M3SDA/)
- [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)

### Dataset Architecture

The dataset is organized into directories, as shown below:

```
- dataset_dir
    - dataset_name
        - domain 1
        - ...
        - domain N
        - text
            - domain 1
                - all.txt
                - train_1.txt
                - train_3.txt
                - test_1.txt
                - test_3.txt
                - val.txt
            - ...
            - domain N
    - ...
```

### Download and Preparation

Before running the data preparation script, make sure to update the configuration file in `data_preparation/dataset.yaml` with the correct settings for your dataset. In particular, you will need to update the `dataset_dir` variable to point to the directory where your dataset is stored.

```yaml
dataset_dir: /path/to/dataset
```

To download and prepare one of these datasets, run the following commands:

```sh
cd data_preparation
python data_preparation.py --dataset <DATASET>
```

Replace <DATASET> with the name of the dataset you want to prepare (e.g. DomainNet, OfficeHome). This script will download the dataset (if necessary) and extract the text data which specify the way to split training, validation, and test sets. The resulting data will be saved in the format described above.

## Running the model

To apply our proposed MuVo method, append the prefix"muvo_" to the selected method.  For example:

```sh
python main.py --method muvo_SLA --dataset DomainNet --source 0 --target 3 --seed 991109 --num_iters 100000 --shot 3shot --alpha 0.3 --update_interval 500 --warmup 50000 --T 0.6
```

This command runs the MuVo+ SLA model on the 3-shot C -> S DomainNet dataset, with the specified hyperparameters. 

## Citation

If you find our work useful, please cite it using the following BibTeX entry:

```bibtex
@article{hong2024multi,
  title={A multi-view consistency framework with semi-supervised domain adaptation},
  author={Hong, Yuting and Dong, Li and Qiu, Xiaojie and Xiao, Hui and Yao, Baochen and Zheng, Siming and Peng, Chengbin},
  journal={Engineering Applications of Artificial Intelligence},
  volume={136},
  pages={108886},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgement

This code is partially based on [MME](https://github.com/VisionLearningGroup/SSDA_MME), [CDAC](https://github.com/lijichang/CVPR2021-SSDA) and [SLA](https://github.com/chu0802/SLA).
