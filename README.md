### ViCE: Self-Supervised Visual Concept Embeddings as Contextual and Pixel Appearance Invariant Semantic Representations

This repository contain code to reproduce results in the paper "ViCE: Self-Supervised Visual Concept Embeddings as Contextual and Pixel Appearance Invariant Semantic Representations".

### Requirements

Python 3.8
Pytorch 1.9.1
CUDA 11.1


### Setup

Initialize an environment of your choice

Ex: conda create -n vice python=3.8

#### Install VISSL (NOTE: Use the provided code)

pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu111_pyt191/download.html
cd vissl/
pip install --progress-bar off -r requirements.txt
pip install opencv-contrib-python
pip install classy-vision
pip install -e .[dev]

verify installation
python -c 'import vissl, apex, cv2'

back to root
cd ../

#### Install MMSegmentation (NOTE: Use the provided code)

pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
cd mmsegmentation/

pip install -e . 

back to root
cd ../

### Usage

#### Example 1: Demonstrate COCO model training w. 1 GPU

1. Download the COCO-Stuff164K dataset and create a symbolic link inside datasets/coco to the COCO images/ directory

coco_stuff164k
    images
       train2017
       val2017
       unlabeled2017
...

ln -s path-to-to-your-coco_stuff164k/images coco

2. Configure extra_scripts/datasets/dataset_config.yaml to read COCO


3. Create filelist_coco.npy

python extra_scripts/datasets/gen_dataset_filelist.py extra_scripts/datasets/dataset_config.yaml --out-filename filelist_coco.npy

4. Run training demo config

python tools/run_distributed_engines.py config=pretrain/vice/vice_1gpu_resnet_coco_demo.yaml





### Pretrained models


### Results