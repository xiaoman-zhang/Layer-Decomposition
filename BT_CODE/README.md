We provide the official <b>PyTorch</b> implementation of Our Layer Decomposition


Take the implementation on brain tumor data as example:
### 1. Install environment
```bash
$ cd BT_CODE
$ pip install -r requirements.txt
```
### 2. Preprocessing the pretraining dataset
All volumes are already co-registered, skull stripped and resampled into a 1-mm isotropic image in atlas space.

### 3. Train the layer decomposition model
```bash
python train.py 
```

