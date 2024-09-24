# Improving Pseudo-labelling and Enhancing Robustness for Semi-Supervised Domain Generalization
Adnan Khan, Mai A. Shaaban, Muhammad Haris Khan

**Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE**

[![Static Badge](https://img.shields.io/badge/Paper-Link-yellowgreen?link=https%3A%2F%2Fzenodo.org%2Frecords%2F10104139)](https://arxiv.org/abs/2401.13965)
[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

## Abstract

Beyond attaining domain generalization (DG), visual recognition models should also be data-efficient during learning by leveraging limited labels. We study the problem of Semi-Supervised Domain Generalization (SSDG) which is crucial for real-world applications like automated healthcare. SSDG requires learning a cross-domain generalizable model when the given training data is only partially labelled. Empirical investigations reveal that the DG methods tend to underperform in SSDG settings, likely because they are unable to exploit the unlabelled data. Semi-supervised learning (SSL) shows improved but still inferior results compared to fully-supervised learning. A key challenge, faced by the best performing SSL-based SSDG methods, is selecting accurate pseudo-labels under multiple domain shifts and reducing overfitting to source domains under limited labels. In this work, we propose new SSDG approach, which utilizes a novel uncertainty-guided pseudo-labelling with model averaging (UPLM). Our uncertainty-guided pseudo-labelling (UPL) uses model uncertainty to improve pseudo-labelling selection, addressing poor model calibration under multi-source unlabelled data. The UPL technique, enhanced by our novel model averaging (MA) strategy, mitigates overfitting to source domains with limited labels. Extensive experiments on key representative DG datasets suggest that our method demonstrates effectiveness against existing methods.

## Results
Comparison of baseline model (FixMatch), our Uncertainty-Guided PL approach (UPL), our Model Averaging MA, and our Uncertainty-Guided PL with Model averaging (UPLM).

### PACS

| Target Domain | FixMatch | UPL | MA | UPLM |
| --- | --- | --- | --- | --- |
| Photo   | 82.67 | 89.76 | 90.40 | 88.09 |
| Art     | 70.79 | 72.75 | 76.53 | 76.84 |
| Cartoon | 70.39 | 66.87 | 75.78 | 74.05 |
| Sketch  | 70.19 | 74.63 | 71.43 | 76.79 |
| Average | 73.51 | 76.35 | 78.54 | **78.94** |

### OfficeHome

| Target Domain | FixMatch | UPL | MA | UPLM |
| --- | --- | --- | --- | --- |
| Art         | 38.64 | 39.37 | 43.52 | 42.47 |
| Clipart     | 39.28 | 41.69 | 41.76 | 40.58 |
| Product     | 58.73 | 58.10 | 59.41 | 58.00 |
| Real World  | 56.88 | 60.87 | 63.91 | 61.37 |
| Average | 48.38 | 50.00 | **52.15** | 50.61 |

### VLCS

| Target Domain | FixMatch | UPL | MA | UPLM |
| --- | --- | --- | --- | --- |
| Caltech101  | 43.37 | 74.08 | 36.42 | 85.68 |
| LabelMe     | 52.78 | 59.23 | 51.49 | 61.09 |
| SUN09       | 49.88 | 42.96 | 62.60 | 50.41 |
| VOC2007     | 27.26 | 41.02 | 41.87 | 53.68 |
| Average     | 43.32 | 54.33 | 48.10 | **62.72** |

### Terra

| Target Domain | FixMatch | UPL | MA | UPLM |
| --- | --- | --- | --- | --- |
| Location 38       | 15.00 | 22.14 | 28.59 | 32.32 |
| Location 43       | 14.07 | 14.07 | 17.88 | 25.82 |
| Location 46       | 19.04 | 21.15 | 21.77 | 24.22 |
| Location 100      | 22.14 | 25.23 | 40.97 | 38.38 |
| Average           | 17.56 | 20.07 | 27.30 | **30.19** |

## Usage

### Dependencies
Create an environment using the following command: ```conda env create -n uplm --file environment.yml```

### Datatset Preparation
- Download datasets along with their splits (train, test, and unlabeled) from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/mai_kassem_mbzuai_ac_ae/EsC7ID7TDMNDi9m3O7evTYsBDOjeQG-adN4BPaeSfdqiaQ?e=jNPqRu).
- Create a folder named ```datasets``` in the root directory of the project.
- Place the downloaded zip files in the ```datasets``` folder.
- Unzip the datasets.

### Reproducing Results
#### Available choices
- ```--dataset_name``` pacs, office_home, terra, vlcs 
- ```--seed``` 1, 2, 3 
- ```--train_mode``` base, upl, ma, uplm
- ```--un_thresh``` use 0.2, 0.5, 0.5 and 0.7 for PACS, TerraIncognita, OfficeHome and VLCS respectively.
- ```--out``` your_output_path
- ```--domain``` dependent on given dataset_name:
  |    pacs  | office_home|   vlcs    |  terra |
  |----------|------------|-----------|--------|
  | photo    | art        | caltech101| loc_38 |
  | art      | clipart    | label_me  | loc_43 |
  | cartoon  | product    | sun09     | loc_46 |
  | sketch   | real_world | voc2007   | loc_100|

 #### Example 
To train the model on "pacs" dataset, seed "1" and "photo" domain, use the following command:

 <code> python main.py --dataset_name=pacs --seed=1 --domain=photo --un_thresh 0.2 --train_mode uplm --out ./outputs </code>

 ## Acknowledgments 

This work was supported in part by Google unrestricted gift 2023. The authors are grateful for their generous support, which made this research possible.



