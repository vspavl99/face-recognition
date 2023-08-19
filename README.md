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

```bash
# Unzipping the data 
python3 src/data/process_raw_data.py --raw_data_path=<path to raw file.zip>  --baked_data_dir="<destination folder>
```


## Training and evaluating


## Results
