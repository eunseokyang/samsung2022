# Depth estimation for SEM images

Samsung project, manufacturing AI, 2022 Spring

## Requirements

```sh
conda env create -f conda_env.yml
conda activate samsung
```


## Instructions
Dataset is required in `data/`.

### Train
To run training use the `train.py` script
```sh
python train.py
```
for 4-to-1 depth estimation, or
```sh
python train.py --single_image
```
for 1-to-1 depth estimation.

### Predict
```sh
python predict.py
```
for 4-to-1 depth estimation, or
```sh
python predict_one.py
```
for 1-to-1 depth estimation.