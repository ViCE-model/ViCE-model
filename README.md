# ViCE: Self-Supervised Visual Concept Embeddings as Contextual and Pixel Appearance Invariant Semantic Representations

This repository contain code to reproduce results in the paper "ViCE: Self-Supervised Visual Concept Embeddings as Contextual and Pixel Appearance Invariant Semantic Representations". We provide instructions for installations and running experiments.

## Summary

* Installation
    * VISSL
    * MMSegmentation
* Dataset setup
    * COCO-Stuff164K (for training)
    * Coarse COCO-Stuff164K (for evaluation)
    * Cityscapes (for evaluation)
* Download pretrained ViCE models
* Run experiments
    * Example 1: COCO ViCE model training (1 GPU)
    * Example 2: COCO linear model training (w. pretrained ViCE model)
    * Example 3: COCO evaluation (w. pretrained ViCE, linear model)
    * Example 4: Cityscapes evaluation (w. pretrained ViCE, linear model)
    * Example 5: COCO ViCE model training (32 GPUs)

## 1. Requirements

Confirmed to work for Ubuntu 18.04 and 20.04.

* Python 3.8
* Pytorch 1.9.1
* CUDA 11.1

## 2. Installation

The training and evaluation code is implemented within two the frameworks VISSL and MMSegmentation. Follow the instructions bellow to install both frameworks with included modifications for running ViCE **using the code provided in this repository**.

First initialize an environment of your choice using Python 3.8.

```
$ conda create -n vice python=3.8
```

### 2.1: Install VISSL

Starting from the `ViCE-model/` directory root, run the following commands in order to install Pytorch, Apex, and VISSL along with required packages.

```
$ pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu111_pyt191/download.html
$ cd vissl/
$ pip install --progress-bar off -r requirements.txt
$ pip install opencv-contrib-python
$ pip install classy-vision
$ pip install -e .[dev]
```

The installation is successful if running the bellow line generates no error messages.

```
$ python -c 'import vissl, apex, cv2'
```

Return to the root directory

```
$ cd ../
```

### 2.2: Install MMSegmentation

Starting from the `ViCE-model/` directory root, run the following commands in order to install MMCV and MMSegmentation.

```
$ pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
$ cd mmsegmentation/
$ pip install -e . 
```

Return to the root directory

```
$ cd ../
```

## 3: Dataset setup

We provide instructions for setting up the COCO-Stuff164K dataset for training and benchmark experiments, as well as the Cityscapes dataset for benchmark experiments.

### 3.1: COCO-Stuff164K

1. Download the COCO-Stuff164K dataset and create a symbolic link inside `vissl/datasets/` to the COCO `images/` directory. For download instructions, refer to the [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#coco-stuff-164k). The additional unlabeled training images are downloaded from the [official COCO dataset download site](https://cocodataset.org/#download).

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
$ cd vissl/
$ mkdir datasets
$ cd datasets/
$ ln -s PATH-TO-YOUR-coco_stuff164k/images vissl/datasets/coco
```

3. Make sure that `vissl/extra_scripts/datasets/dataset_config.yaml` is setup to parse the COCO dataset (default configuration), with the entry for `coco` as bellow and all other entries set to `False`.

```
coco:
    root_path: "datasets/coco"
    use: True
```

4. Create `filelist_coco.npy` which specifies paths to training images for VISSL. Used datasets are specified in the `dataset_config.yaml` file. Codes for parsing datasets are provided in the `extra_scripts/datasets/dataset_parsers/` directory.

```
$ python extra_scripts/datasets/gen_dataset_filelist.py extra_scripts/datasets/dataset_config.yaml --out-filename filelist_coco.npy
```

### 3.2: Coarse COCO-Stuff164K for evaluation

1. Copy the directory named `curated` from the `ViCE-model/mmsegmentation/tools/convert_datasets/curated` directory into your COCO-Stuff164K directory root. The coarse label splits in `curated` were originally provided in the [IIC paper's GitHub repository](https://github.com/xu-ji/IIC/tree/master/datasets)

2. Run `coco_stuff164k_coarse.py` to create the coarse COCO-Stuff164K dataset corresponding to the samples specified in `curated/train2017/Coco164kFull_Stuff_Coarse_7.txt` and `curated/val2017/Coco164kFull_Stuff_Coarse_7.txt`. Set `--nproc N` to as many threads your CPU have.

```
$ cd mmsegmentation/
$ python tools/convert_datasets/coco_stuff164k_coarse.py PATH-TO-YOUR-coco_stuff164k --nproc 4
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
        train2017/
            .jpg
        val2017/
            .jpg
```

All COCO benchmark training runs are configured to read samples from `annotations_coarse` and `images_coarse`.

3. Finally create a symbolic link to the COCO-Stuff164K dataset inside the `mmsegmentation/data` directory.

```
$ mkdir data
$ cd data/
$ ln -s PATH-TO-YOUR-coco_stuff164k/ coco_stuff164k_coarse
```

### 3.3 Cityscapes for evaluation

1. Download and setup Cityscapes following instructions [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html#cityscapes)

2. Remember to create the symbolic link inside `mmsegmentation/data/` 

```
$ ln -s PATH-TO-YOUR-cityscapes/ cityscapes
```

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

3. Make sure to generate label maps following the above instructions provided by MMSementation. Make sure you run the following command from the `ViCE-model/mmsegmentation/` directory.

```
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

4. Return back to the `ViCE-model/` root directory.

```
$ cd ../../
```

## 4. Pretrained models

### 4.1 ViCE models

Download the following pretrained models and configuration files to the specfied directory.

1. ViCE model trained on COCO

* Download [model_final_checkpoint_phase11.torch](https://drive.google.com/file/d/1oe6fkDfIFvoczIEsoym6vWm_8MQkoTlO/view?usp=sharing) ==> `ViCE-model/vissl/pretrained/exp27_coco/`
* Training log: `log.txt`

2. ViCE model trained on road scenes

* Download [model_final_checkpoint_phase1.torch](https://drive.google.com/file/d/14PlRUVvtn8nePLfdVcZHE7HO2063Mnlq/view?usp=sharing) ==> `ViCE-model/vissl/pretrained/exp31_road/`
* Training log: `log.txt`

### 4.2 Linear evaluation models

The linear evaluation models for COCO and Cityscapes are **included as part of the repository**.

1. Linear COCO model (for use with ViCE model `exp27_coco`)

* `ViCE-model/mmsegmentation/pretrained/fcn_linear_coco-stuff164k_vissl_sc_exp27_ep12_max/iter_12000.pth`
* Training log: `20211110_214004.txt`

2. Linear Cityscapes model (for use with ViCE model `exp31_road`)

* `ViCE-model/mmsegmentation/pretrained/fcn_linear_cityscapes_vissl_sc_exp31_ep2_max/iter_12000.pth`
* Training log: `20211111_143137.txt`

## 5. Usage

### Example 1: COCO ViCE model training (1 GPU)

Run the following command from the `ViCE-model/vissl/` directory.

```
$ python tools/run_distributed_engines.py config=pretrain/vice/vice_1gpu_resnet_coco_demo.yaml
```

The single GPU experiment is provided for demonstration purposes only. The resulting model is not expected to result in a high-performance model. See the 32 GPU experiment example bellow for how to reproduce the provided pretrained ViCE models.

### Example 2: COCO linear model training (pretrained ViCE model)

Run the following command from the `ViCE-modelmmsegmentation/` directory to reproduce the COCO linear evaluation model using the provided pretrained ViCE model. Note that the following setup is configured to run on a node with 8 A6000 GPUs.

```
$ ./tools/dist_train.sh tools/train.py configs/fcn_linear_coco-stuff164k_exp27.py 8 --work-dir YOUR-EXPERIMENT_DIR_NAME
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
$ python tools/train.py configs/fcn_linear_coco-stuff164k_exp27.py --work-dir YOUR-EXPERIMENT_DIR_NAME
```

### Example 3: COCO evaluation (pretrained ViCE, linear model)

Run the following command to reproduce the benchmark evaluation score using the provided pretrained ViCE and linear model.

```
$ python tools/test.py configs/fcn_linear_coco-stuff164k_exp27.py pretrained/fcn_linear_coco-stuff164k_vissl_sc_exp27_ep12_max/iter_12000.pth --eval mIoU
```

If VISSL, MMSegmentation, the coarse COCO-Stuff164K dataset, and pretrained models are correctly setup, the following score will be outputted.

```
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 65.47 | 27.71 | 37.18 |
+-------+-------+-------+
```

### Example 4: Cityscapes evaluation (pretrained ViCE, linear model)

Run the following command to reproduce the benchmark evaluation score using the provided pretrained ViCE and linear model.

```
python tools/test.py configs/fcn_linear_cityscapes_vissl_sc_exp31.py pretrained/fcn_linear_cityscapes_vissl_sc_exp31_ep2_max/iter_12000.pth --eval mIoU
```

If VISSL, MMSegmentation, the coarse COCO-Stuff164K dataset, and pretrained models are correctly setup, the following score will be outputted.

```
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 85.51 | 28.83 | 34.72 |
+-------+-------+-------+
```

### Example 5: COCO ViCE model training (32 GPUs)

We train our models on a supercomputer using a job scheduling system. Please reference our setup while configuring the code to work on your system.

1. Modify job.sh to activate the environment (See 'Setup'). Also specify username and paths.

2. Similarly modify distributed_train.sh to activate the environment. Specify config file to run (COCO or road scene training).

3. Launch the job.

```
$ pjsub job.sh
```

## Results

Representation quality performance of our model on coarse COCO-Stuff164K (Table 1. in the paper)

| Embedding extractor | mIoU      | Acc.      |
| ------------------- | --------- | --------- |
| IIC                 | 13.26     | 51.49     |
| Mod. DeepCluster    | 13.76     | 50.79     |
| PiCIE               | 14.77     | 54.75     |
| ViCE (ours)         | **27.71** | **65.47** |

Representation quality performance of our model on Cityscapes (Table 2. in the paper)

| Embedding extractor | mIoU      | Acc.      |
| ------------------- | --------- | --------- |
| IIC                 | 6.35      | 47.88     |
| Mod. DeepCluster    | 7.06      | 40.67     |
| PiCIE               | 12.31     | 65.50     |
| ViCE (ours)         | **28.83** | **85.51** |
