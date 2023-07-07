# TransLO: A Window-Based Masked Point Transformer Framework for Large-Scale LiDAR Odometry
This is official implementation for our AAAI2023 paper "TransLO: A Window-Based Masked Point Transformer Framework for Large-Scale LiDAR Odometry" .
<img src="pipeline.png" alt="teaser" width="50%" />

## Installation
Our model only depends on the following commonly used packages.

| Package      | Version                          |
| ------------ | -------------------------------- |
| CUDA         |  1.11.3                          |
| PyTorch      |  1.10.0                          |
| h5py         | *not specified*                  |
| tqdm         | *not specified*                  |
| numpy        | *not specified* (we used 1.20.2) |
| scipy        | *not specified* (we used 1.6.2)  |

## Install the pointnet2 library
Compile the furthest point sampling, grouping and gathering operation for PyTorch with following commands. 
```bash
cd ops_pytorch
cd fused_conv_random_k
python setup.py install
cd ../
cd fused_conv_select_k
python setup.py install
cd ../
```
## Datasets

Datasets are available at KITTI Odometry benchmark website: [ https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

## Training
```bash
python train.py 
```
You may specify the value of arguments. Please find the available arguments in the configs.py. 

## Testing

```bash
python train.py
```

## Citation

```
@inproceedings{liu2023translo,
  title={TransLO: A Window-Based Masked Point Transformer Framework for Large-Scale LiDAR Odometry},
  author={Liu, Jiuming and Wang, Guangming and Jiang, Chaokang and Liu, Zhe and Wang, Hesheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={2},
  pages={1683--1691},
  year={2023}
}
```
