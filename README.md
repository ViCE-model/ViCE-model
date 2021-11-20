## ViCE: Self-Supervised Visual Concept Embeddings as Contextual and Pixel Appearance Invariant Semantic Representations

This repository contain code to reproduce results in the paper "ViCE: Self-Supervised Visual Concept Embeddings as Contextual and Pixel Appearance Invariant Semantic Representations". We provide instructions for running training and benchmark experiments.

## 1. Requirements

* Python 3.8
* Pytorch 1.9.1
* CUDA 11.1

## 2. Installation

The training and evaluation code relies on the frameworks VISSL and MMSegmentation, respectively. Follow the instructions bellow to install both frameworks with modifications for running ViCE **using the code provided in this repository**.

First initialize an environment of your choice using Python 3.8.

    conda create -n vice python=3.8

### 2.1: Install VISSL

Starting from the ViCE-model directory root, run the following commands in order to install Pytorch, Apex, and VISSL along with required packages.

    pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu111_pyt191/download.html
    cd vissl/
    pip install --progress-bar off -r requirements.txt
    pip install opencv-contrib-python
    pip install classy-vision
    pip install -e .[dev]
    
The installation is successful if running the bellow line generates no error messages.

    python -c 'import vissl, apex, cv2'

Return to the root directory

    cd ../

### 2.2: Install MMSegmentation

Starting from the ViCE-model directory root, run the following commands in order to install MMCV and MMSegmentation.

    pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    cd mmsegmentation/
    pip install -e . 

Return to the root directory

    cd ../

## 3: Dataset setup

In this repository we provide instructions for setting up the COCO-Stuff164K dataset for training and benchmark experiments, as well as the Cityscapes dataset for benchmark experiments. For the road scene training experiment, please download and arrange the five datasets listed in the paper and refer to how the COCO-Stuff164K is setup.

### 3.1: COCO-Stuff164K

1. Download the COCO-Stuff164K dataset and create a symbolic link inside datasets/coco to the COCO images/ directory. For download instructions, refer to the [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#coco-stuff-164k). The additional unlabeled training images are downloaded from the [official COCO dataset download site](https://cocodataset.org/#download).

Expected vanilla dataset directory structure

```
coco_stuff164k/
    annotations/
        train2017/
            .png
        val2017/
            .png
    images/
        train2017/
            .jpg
        val2017/
            .jpg
        unlabeled2017/
            .jpg
```

2. Inside the `VISSL/datasets` directory, symbolically link a path to the COCO dataset `images` directory.

```coco_symlink
ln -s PATH-TO-YOUR-coco_stuff164k/images vissl/datasets/coco
```

3. Make sure that `vissl/extra_scripts/datasets/dataset_config.yaml` is setup to parse the COCO dataset (default configuration), with the entry for `coco` as bellow and all other entries set to `False`.

```
coco:
    root_path: "datasets/coco"
    use: True
```

3. Create `filelist_coco.npy` which specifies paths to training images for VISSL. Used datasets are specified in the `dataset_config.yaml` file. Codes for parsing datasets are provided in the `extra_scripts/datasets/dataset_parsers/` directory.

```
python extra_scripts/datasets/gen_dataset_filelist.py extra_scripts/datasets/dataset_config.yaml --out-filename filelist_coco.npy
```

### 3.2: Coarse COCO-Stuff164K for benchmarking

1. Copy the directory named `curated` from the `ViCE-model/mmsegmentation/tools/convert_datasets/curated` directory into your COCO-Stuff164K directory root. The coarse label splits in `curated` are originally provided in the [IIC paper's GitHub repository](https://github.com/xu-ji/IIC/tree/master/datasets)

2. Run `coco_stuff164k_coarse.py` to create the coarse COCO-Stuff164K dataset corresponding to the samples specified in `curated/train2017/Coco164kFull_Stuff_Coarse_7.txt` and `curated/val2017/Coco164kFull_Stuff_Coarse_7.txt`. Set `--nproc N` to as many threads your CPU have.

```
cd mmsegmentation/
python tools/convert_datasets/coco_stuff164k_coarse.py PATH-TO-YOUR-coco_stuff164k --nproc 4
```

The above script first generates new coarse GT label maps with 27 classes and stored in the same annotation folder as the original GT label map. Finally two new directories `annotations_coarse` and `images_coarse` are created with symbolic links to the previously generated GT label maps and corresponding RGB images.

Upon completion the COCO-Stuff164K directory structure will look as follows.

```
coco_stuff164k/
    annotations/
    annotations_coarse/
        train2017/
            .png
        val2017/
            .png
    curated/
    images/
    images_coarse/
```

All COCO benchmark training runs are configured to read samples from `annotations_coarse` and `images_coarse`.

3. Finally create a symbolic link to the COCO-Stuff164K dataset in MMSegmentation's `data` directory.

```
mkdir data
cd data/
ln -s PATH-TO-YOUR-coco_stuff164k/ coco_stuff164k_coarse
```

4. Return back to the ViCE-model root directory

```
cd ../../
```

### 3.3 Cityscapes for benchmarking

1. Download and setup Cityscapes following instructions [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#cityscapes)

Expected vanilla dataset directory structure

```
cityscapes/
    gtFine/
        test/
        train/
        val/
    leftImg8bit/
        test/
        train/
        val/
```

2. Also generate label maps following the above instructions provided by MMSementation. Make sure you run the following command from the `ViCE-model/mmsegmentation/` directory.

```
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

## Pretrained models


## Usage

### Example 1: COCO model training demo w. 1 GPU

1. Set up COCO-Stuff164K dataset (see 2.3.1: COCO-Stuff164K).

2. Make sure your terminal is in the VISSL directory (i.e. ViCE-model/vissl/).

3. Run training demo config.

python tools/run_distributed_engines.py config=pretrain/vice/vice_1gpu_resnet_coco_demo.yaml

### Example 2: COCO representation quality linear model training

First make sure you are in the MMSegmentation directory

```
cd mmsegmentation/
```

Run the following command to reproduce the COCO linear model using the supplied pretrained ViCE model. Note that the following setup is configured to run on  a node with 8 A6000 GPUs.

```
./tools/dist_train.sh tools/train.py configs/fcn_linear_500x500x10k_coco-stuff164k_exp27.py 8 --work-dir EXPERIMENT_DIR_NAME
```

For systems with fewer GPUs and memory, please modify the configuration to match your system

* Reduce samples per GPU if your run out of memory

```
data = dict(
    samples_per_gpu=4, # <-- Lower
    workers_per_gpu=4,
```

* Command to train the linear model on using a single GPU

```
python tools/train.py configs/fcn_linear_500x500x10k_coco-stuff164k_exp27.py --work-dir EXPERIMENT_DIR_NAME
```

**TODO** Confirm configuration and run option for COCO and Cityscapes benchmark experiments !!!

### Example 3: Run benchmark evaluation with pretrained ViCE and linear models

Run the following command to reproduce the benchmark evaluation score using the supplied pretrained ViCE and linear model.

```
python tools/test.py configs/fcn_linear_500x500x10k_coco-stuff164k_exp27.py --eval mIoU
```

### Example 2: Model training w. 32 GPUs

We train our models on a supercomputer using a job scheduling system. Feel free to reference our setup while configuring the code to work on your system.

1. Modify job.sh to activate the environment (See 'Setup'). Also specify username and paths.

2. Similarly modify distributed_train.sh to activate the environment. Specify config file to run (COCO or road scene training).

3. Launch the job.

pjsub job.sh



## Results