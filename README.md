# Master thesis
## How to install on new machibe
### 0. install require things

```
sudo apt install screen git vim wget
alias python=python3
```

### 1. Install conda
#### 1.1 Install the conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

#### 1.2 Add to path
First open `~/.bashrc`
```
vim ~/.bashrc
```
Write the following in the end of the file
```
export PATH=$PATH:/home/ubuntu/anaconda3/bin
```
reload bash
```
source ~/.bashrc
```


### 2. Clone the Repository

#### 2.1 Create public key

```
ssh-keygen
```

#### 2.2 Copy the key to github

```
cat .ssh/id_rsa.pub
```

#### 2.2 Clone the repo to the machine

```
git clone git@github.com:SamanFekri/MasterThesis.git
```

### 3. Make the directories

```
mkdir models
mkdir dataset
mkdir checkpoints
```

### 4. Make .env file

```
cd MasterThesis
vim .env
```

```
DATASET_PATH_RAW=../dataset/raw
DATASET_PATH_PROCESSED=../dataset/pix2pix_clipped
MODEL_CONTROL_NET_PATH=../models/control_sd15_ini.ckpt
CHECKPOINT_PATH=../checkpoints
```

### 5. Install requirements

```
cd fill50k/ControlNet
conda env create -f environment.yaml
cd ../..
```

#### 5.1 use control environment
```
source activate control
```

### 6. Download from hugging face and convert it

```
screen -S download
python download_hugging.py && python convert_to_controlnet.py
// Ctrl+A Ctrl+d for detaching from it
```

### 7. Download the SD model

```
cd ../models
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
```

### 8. Convert SD Model to ControlNet

```
cd ../MasterThesis/fill50k/ControlNet
python tool_add_control.py ../../../models/v1-5-pruned.ckpt ../../../models/control_sd15_ini.ckpt
```

### 9. login to wandb

```
wandb login
```

### 10. run the code

```
cd MasterThesis/fill50k
python train_pix2pix.py
```