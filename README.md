<div align="center">

# Face recognition

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

</div>

## Description

______________________________________________________________________


## Installation and environment

#### Pip

```bash
# clone project
git clone https://github.com/vspavl99/face-recognition.git
cd face-recognition

# install requirements
pip install -r requirements.txt
```

## Dataset
The dataset contain image of people in difference angle and random background photos. 
Each image correspond to certain cluster, which specified in `cluster.csv` file
## Preparing data
Unzipping the data 
```bash
python3 src/data/process_raw_data.py --raw_data_path="<path to raw file.zip>"  --baked_data_dir="<destination folder>"
```

## Extracting embeddings
```bash
python3 src/features/extract.py --data_dir="<path folder with images>"  --output_path="<path for result file>" 
```
example:
```bash
python3 src/features/extract.py --data_dir="/home/vpavlishen/data_ssd/vpavlishen/test-task/clusters" --output_path="/home/vpavlishen/face-recognition/data/processed/test-task/embeddings.txt"
```

## Training and evaluating


## Results
Embeddings projected into 2d space via umap algorithm:
|                         Predictions                          |                        Targets                        |
|:------------------------------------------------------------:|:-----------------------------------------------------:|
| ![reports/figures/predictions.png](reports/figures/predictions.png) | ![reports/figures/targets.png](reports/figures/targets.png) |