# FLGo-Bench
Produce results of federated algorithms on various benchmarks

## Citation

Please cite our paper in your publications if this code helps your research.

```
@misc{wang2023flgo,
      title={FLGo: A Fully Customizable Federated Learning Platform}, 
      author={Zheng Wang and Xiaoliang Fan and Zhaopeng Peng and Xueheng Li and Ziqi Yang and Mingkuan Feng and Zhicheng Yang and Xiao Liu and Cheng Wang},
      year={2023},
      eprint={2306.12079},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Usage
- **Tuning Command**
```python
python tune.py --task TASKNAME --algorithm ALGORITHM --config CONFIG_PATH --gpu GPUids 
```

- **Running Command**
```python
python run.py --task TASKNAME --algorithm ALGORITHM --config CONFIG_PATH --gpu GPUids 
```
- **Optional Args**

| **Name**           | **Type** | **Desc.**                                                                                                   |   
|--------------------|----------|-------------------------------------------------------------------------------------------------------------|
| model              | str      | the file name in the dictionary `model/` that denotes a legal model in FLGo                                 |
| load_mode          | str      | be one of ['', 'mmap', 'mem'], which respectively denotes DefaultIO, MemmapIO, and InMemory Dataset Loading |
| max_pdev           | int      | the maximum number of processes on each gpu device                                                          |
| available_interval | int      | the time interval (s) to check whether a device is available                                                |
| put_interval       | int      | the time interval (s) to put one process into device                                                        |
| seq                | bool     | whether to run each process in sequencial                                                                   |
| train_parallel     | int      | the number of parallel client local training processes, default is 0                                        |
| test_parallel      | bool     | whether to use data parallel when evaluating model                                                          |
| use_cache          | bool     | whether to use the disk to dynamically cache clients' states                                                |

- **Example**
```python
# Tuning FedAvg on MNIST-IID with GPU 0 and 1
python tune.py --task mnist_iid_c100 --algorithm fedavg --config ./config/general.yml --gpu 0 1

# Runing FedAvg on MNIST-IID with GPU 0 and 1
python run.py --task mnist_iid_c100 --algorithm fedavg --config ./config/general.yml --gpu 0 1
```

## Analysis 
```python
# Show tuning Result
python get_tune_res.py --task TASK --algorithm ALGORITHM --model MODEL --config CONFIG_PATH 

# Show Running Result
python get_run_res.py --task TASK --algorithm ALGORITHM --model MODEL --config CONFIG_PATH 
```

## Algorithmic Configuration
We search the algorithmic hyper-parameter for each algorihtm according to the table below

| **Algorithm** | **Hyper-Parameter**                 |
|---------------|-------------------------------------|
| fedavg        | -                                   |
| fedprox       | μ ∈ [0.0001, 0.001, 0.01, 0.1, 1.0] | 
| scaffold      | η = 1.0                             |
| feddyn        | α ∈ [0.001, 0.01, 0.03, 0.1]        |
| moon          | μ ∈ [0.1, 1.0, 5.0, 10.0], τ=0.5    | 


**Remark:** To specify the search space of the hyper-parameters, add `algo_para: [V1, V2, ...]` in the corresponding config file.
# Experimental Results
## Nevigation

- [CIFAR10](#CIFAR10)
- [CIFAR100](#CIFAR100)
- [TinyImageNet](#TinyImageNet)
- [MNIST](#MNIST)
- [FEMNIST](#FEMNIST)
- [AgNews](#AGNEWS)
- [Office-Caltech10](#Office-Caltech10)
- [DomainNet](#DomainNet)
- [PACS](#PACS)
- [Digits](#Digits)
- [ProstateMRI](#ProstateMRI)
- [Camelyon17](#Camelyon17)
- [Fundus](#Fundus)
- [EndoPolyp](#EndoPolyp)
- [SpeechCommand](#SpeechCommand)
- [Shakespeare](#Shakespeare)
## CIFAR10
### 100 Clients

| iid            |  dir5.0 |dir2.0 |dir1.0 |dir0.1 |
| :-------------------------:|:-------------------------: |:-------------------------: |:-------------------------: |:-------------------------: |
| ![iid_img](/task/cifar10_iid_c100/res.png)  |  ![d5_img](/task/cifar10_dir5.0_c100/res.png) |  ![d2_img](/task/cifar10_dir2.0_c100/res.png) |  ![d1_img](/task/cifar10_dir1.0_c100/res.png) |  ![d0_img](/task/cifar10_dir0.1_c100/res.png) |

#### Configuration
```
learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
proportion: 0.2
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 2000
num_epochs: [5]
clip_grad: 10
early_stop: 500
train_holdout: 0.2
local_test: True
no_log_console: True
```

| **Algorithm** | **model**   | **iid**          | **dir5.0**     | **dir2.0**      | **dir1.0**        | **dir0.1**       |    
|---------------|-------------|------------------|----------------|-----------------|-------------------|------------------|
| fedavg        | CNN         | lr=0.1           | lr=0.1         | lr=0.05         | lr=0.1            | lr=0.1           |
| fedprox       | CNN         | lr=0.05, μ=0.001 | lr=0.1, μ=0.01 | lr=0.1, μ=0.001 | lr=0.1, μ=0.01    | lr=0.05, μ=0.001 |
| scaffold      | CNN         | lr=0.1           | lr=0.1         | lr=0.1          | lr=0.1            | lr=0.1           |
| moon          | CNN         | lr=0.1, μ=0.1    | lr=0.1, μ=0.1  | lr=0.05, μ=0.1  | lr=0.1, μ=0.1     | lr=0.1, μ=0.1    |
| feddyn        | CNN         | lr=0.1, α=0.1    | lr=0.1, α=0.1  | lr=0.05, α=0.1  | lr=0.1, α=0.03    | lr=0.05, α=0.03  |
|               |             |                  |                |                 |                   |                  |
| fedavg        | ResNet18    | lr=0.1           | lr=0.1         | lr=0.1          | lr=0.05           | lr=0.1           |
| fedprox       | ResNet18    | lr=0.05, μ=0.001 | lr=0.1, μ=0.1  | lr=0.1, μ=0.001 | lr=0.05, μ=0.0001 | lr=0.05, μ=0.001 |
| scaffold      | ResNet18    | lr=0.1           | lr=0.1         | lr=0.1          | lr=0.1            |                  |
| moon          | ResNet18    | lr=0.1, μ=0.1    | lr=0.05, μ=0.1 |                 |                   |                  |
| feddyn        | ResNet18    | lr=0.1, α=0.1    | lr=0.1, α=0.1  | lr=0.1, α=0.1   | lr=0.1, α=0.1     |                  |
|               |             |                  |                |                 |                   |                  |
| fedavg        | ResNet18-GN | lr=0.1           | lr=0.1         | lr=0.1          | lr=0.1            | lr=0.1           |



#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
*Global Test*

| **Algorithm** | **model**   | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-------------|------------|------------|------------|------------|------------|
| fedavg        | CNN         | 81.54±0.14 | 79.95±0.22 | 78.00±0.24 | 76.54±0.39 | 69.87±0.50 |
| fedprox       | CNN         | 80.83±0.20 | 79.86±0.30 | 78.39±0.26 | 76.58±0.50 | 69.41±0.33 |
| scaffold      | CNN         | 85.08±0.21 | 83.84±0.23 | 82.15±0.29 | 80.42±0.21 | 65.06±0.51 |
| moon          | CNN         | 80.88±0.27 | 79.63±0.20 | 77.21±0.33 | 75.67±0.26 | 62.44±1.10 |
| feddyn        | CNN         | 85.09±0.22 | 83.65±0.09 | 81.54±0.30 | 80.26±0.40 | 70.82±0.50 |
|               |             |            |            |            |            |            |
| fedavg        | ResNet18    | 94.07±0.12 | 93.56±0.18 | 92.59±0.09 | 91.53±0.11 | 78.79±0.56 |
| fedavg        | ResNet18-GN | 91.25±0.23 | 89.93±0.24 | 88.21±0.40 | 86.42±0.73 | 66.39±1.66 |

*Local Test*

| **Algorithm** | **model**   | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-------------|------------|------------|------------|------------|------------|
| fedavg        | CNN         | 80.98±0.39 | 79.97±0.12 | 77.43±0.12 | 77.03±0.29 | 70.02±0.70 |
| fedprox       | CNN         | 80.68±0.18 | 79.81±0.41 | 77.80±0.13 | 77.05±0.20 | 69.72±0.79 |
| scaffold      | CNN         | 85.22±0.33 | 84.23±0.44 | 81.90±0.28 | 80.23±0.19 | 65.00±0.66 |
| moon          | CNN         | 80.26±0.38 | 79.29±0.29 | 76.81±0.58 | 76.12±0.45 | 62.26±1.09 |
| feddyn        | CNN         | 85.25±0.26 | 83.99±0.17 | 81.76±0.17 | 80.48±0.42 | 71.69±0.29 |
|               |             |            |            |            |            |            |
| fedavg        | ResNet18    | 94.58±0.08 | 93.54±0.11 | 93.12±0.23 | 91.67±0.25 | 79.46±0.77 |
| fedavg        | ResNet18-GN | 91.69±0.19 | 90.04±0.26 | 87.84±0.48 | 86.52±0.69 | 66.98±1.59 |

#### Impact of Sampling Ratio

| **Task** | **Algorithm** | **model** | **p=0.1**  | **p=0.2**  | **p=0.5**  | **p=1.0**  |  
|----------|---------------|-----------|------------|------------|------------|------------|
| iid      | fedavg        | CNN       | 81.70±0.30 | 81.54±0.14 | 81.34±0.23 | 81.87±0.17 |

#### Impact of Local Epoch

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## CIFAR100
### 100 Clients

| iid      |dir1.0 |dir0.1 |
|:-------------------------:|:-------------------------: |:-------------------------: |
| ![cifar100_iid_img](/task/cifar100_iid_c100/res.png)  |  ![cifar100_d1_img](/task/cifar100_dir1.0_c100/res.png) |  ![cifar100_d0_img](/task/cifar100_dir0.1_c100/res.png) |
#### Configuration
```
learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
proportion: 0.2
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 2000
num_epochs: [5]
clip_grad: 10
early_stop: 500
train_holdout: 0.2
local_test: True
no_log_console: True
```

| **Algorithm** | **model**   | **iid**          | **dir1.0**      | **dir0.1**        | 
|---------------|-------------|------------------|-----------------|-------------------|
| fedavg        | CNN         | lr=0.1           | lr=0.1          | lr=0.1            | 
| fedprox       | CNN         | lr=0.1, μ=0.001  | lr=0.1, μ=0.001 | lr=0.05, μ=0.0001 |
| scaffold      | CNN         | lr=0.1           | lr=0.1          | lr=0.1            |
| feddyn        | CNN         | lr=0.001, α=0.1  | lr=0.1, α=0.1   | lr=0.1, α=0.03    | 
| moon          | CNN         | lr=0.1, μ=0.1    | lr=0.1, μ=0.1   | lr=0.05, μ=0.1    | 
|               |             |                  |                 |                   |    
| fedavg        | ResNet18    | lr=0.1           | lr=0.05         | lr=0.05           |
| fedprox       | ResNet18    | lr=0.1, μ=0.0001 | lr=0.05, μ=0.01 | lr=0.05, μ=0.1    |
| scaffold      | ResNet18    | lr=0.1           | lr=0.1          | lr=0.1            |
| feddyn        | ResNet18    | lr=0.1, α=0.1    | lr=0.05, α=0.1  | lr=0.05, α=0.1    |
| moon          | ResNet18    | lr=0.1, μ=10.0   | lr=0.05, μ=0.1  | lr=0.05, μ=0.1    |
|               |             |                  |                 |                   |
| fedavg        | ResNet18-GN | lr=0.1           | lr=0.1          | lr=0.01           | 

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
*Global Test*

| **Algorithm** | **model**   | **iid**    | **dir1.0** | **dir0.1** | 
|---------------|-------------|------------|------------|------------|
| fedavg        | CNN         | 41.33±0.30 | 37.93±0.64 | 22.50±0.52 | 
| fedprox       | CNN         | 41.27±0.31 | 37.79±0.27 | 22.20±0.16 |
| scaffold      | CNN         | 49.85±0.22 | 41.08±0.39 | 18.56±0.50 | 
| feddyn        | CNN         | 52.64±0.15 | 40.19±0.36 | 26.20±0.44 | 
| moon          | CNN         | 41.49±0.40 | 37.28±0.37 | 21.09±0.32 |
|               |             |            |            |            |    
| fedavg        | ResNet18    | 72.91±0.33 | 49.11±0.56 | 16.63±0.63 | 
| fedprox       | ResNet18    | 73.26±0.29 |            |            |
| scaffold      | ResNet18    | 76.35±0.22 |            |            |
| feddyn        | ResNet18    | 75.43±0.17 |            |            |
| moon          | ResNet18    | 74.95±0.22 |            |            |
|               |             |            |            |            |
| fedavg        | ResNet18-GN | 52.71±0.68 | 34.20±0.83 | 19.45±0.53 | 

*Local Test*


| **Algorithm** | **model**   | **iid**    | **dir1.0** | **dir0.1** | 
|---------------|-------------|------------|------------|------------|
| fedavg        | CNN         | 41.04±0.34 | 36.94±0.52 | 22.21±0.51 | 
| fedprox       | CNN         | 40.62±0.35 | 37.00±0.50 | 21.77±0.21 |
| scaffold      | CNN         | 49.94±0.22 | 39.70±0.31 | 18.58±0.61 |
| feddyn        | CNN         | 52.48±0.49 | 38.92±0.37 | 26.08±0.15 | 
| moon          | CNN         | 40.91±0.30 | 36.22±0.34 | 20.67±0.26 |
|               |             |            |            |            |    
| fedavg        | ResNet18    | 73.72±0.33 | 48.75±1.08 | 16.56±0.48 |     
| fedprox       | ResNet18    | 73.70±0.27 |            |            |
| scaffold      | ResNet18    | 76.60±0.28 |            |            |
| feddyn        | ResNet18    | 75.53±0.40 |            |            |
| moon          | ResNet18    | 75.20±0.24 |            |            |
|               |             |            |            |            |
| fedavg        | ResNet18-GN | 51.87±0.62 | 33.02±1.11 | 19.20±0.22 | 

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## TinyImageNet
### 100 Clients

|                             iid                              |                          dir1.0                           |dir0.1 |
|:------------------------------------------------------------:|:---------------------------------------------------------:|:-------------------------: |
| ![tinyimagenet_iid_img](/task/tinyimagenet_iid_c100/res.png) | ![tinyimagenet_d_img](/task/tinyimagenet_dir1.0_c100/res.png) |  ![tinyimagenet_d0_img](/task/tinyimagenet_dir0.1_c100/res.png) |
#### Configuration
```
learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
proportion: 0.2
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 2000
num_epochs: [5]
clip_grad: 10
early_stop: 500
train_holdout: 0.2
local_test: True
no_log_console: True
```

| **Algorithm** | **model** | **iid**        | **dir1.0**      | **dir0.1**      | 
|---------------|-----------|----------------|-----------------|-----------------|
| fedavg        | ResNet18  | lr=0.1         | lr=0.05         | lr=0.05         | 
| fedprox       | ResNet18  | lr=0.1, μ=0.01 | lr=0.05, μ=1.0  | lr=0.05, μ=1.0  | 
| scaffold      | ResNet18  | lr=0.05        | lr=0.1          | lr=0.1          | 
| feddyn        | ResNet18  | lr=0.1, α=0.1  | lr=0.05, α=0.03 | lr=0.05, α=0.03 |
| moon          | ResNet18  | lr=0.1, μ=1.0  | lr=0.05, μ=0.1  | lr=0.05, μ=0.1  |

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
*Global Test*

| **Algorithm** | **model** | **iid**    | **dir1.0** | **dir0.1**  | 
|---------------|-----------|------------|------------|-------------|
| fedavg        | ResNet18  | 58.89±0.34 | 19.10±0.51 | 6.42±0.17   | 
| fedprox       | ResNet18  | 58.61±0.26 | 27.77±0.65 | 8.84±0.19   | 
| scaffold      | ResNet18  | 60.02±0.33 | 24.89±0.74 |             | 
| feddyn        | ResNet18  | 61.22±0.41 | 27.52±0.61 |             | 
| moon          | ResNet18  | 58.70±0.26 |            |             |



*Local Test*

| **Algorithm** | **model** | **iid**    | **dir1.0** | **dir0.1** | 
|---------------|-----------|------------|------------|------------|
| fedavg        | ResNet18  | 59.16±0.19 | 19.06±0.26 | 5.99±0.34  | 
| fedprox       | ResNet18  | 59.33±0.20 | 28.03±0.14 | 8.38±0.16  | 
| scaffold      | ResNet18  | 60.91±0.31 | 25.37±0.81 |            | 
| feddyn        | ResNet18  | 62.07±0.20 | 27.25±0.68 |            | 
| moon          | ResNet18  | 59.30±0.23 |            |            |

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>


## MNIST
### 100 Clients

| iid            |  dir5.0 |dir2.0 |dir1.0 |dir0.1 |
| :-------------------------:|:-------------------------: |:-------------------------: |:-------------------------: |:-------------------------: |
| ![mnistiid_img](/task/mnist_iid_c100/res.png)  |  ![mnistd5_img](/task/mnist_dir5.0_c100/res.png) |  ![mnistd2_img](/task/mnist_dir2.0_c100/res.png) |  ![mnistd1_img](/task/mnist_dir1.0_c100/res.png) |  ![mnistd0_img](/task/mnist_dir0.1_c100/res.png) |

```
learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
proportion: 0.2
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 1000
num_epochs: [5]
clip_grad: 10
early_stop: 250
train_holdout: 0.2
local_test: True
no_log_console: True
```

| **Algorithm** | **model** | **iid**          | **dir5.0**       | **dir2.0**     | **dir1.0**     | **dir0.1**       |    
|---------------|-----------|------------------|------------------|----------------|----------------|------------------|
| fedavg        | CNN       | lr=0.1           | lr=0.05          | lr=0.05        | lr=0.05        | lr=0.05          |
| fedprox       | CNN       | lr=0.1, μ=0.0001 | lr=0.05, μ=0.001 | lr=0.1, μ=1.0  | lr=0.1, μ=1.0  | lr=0.05, μ=0.01  |
| scaffold      | CNN       | lr=0.01          | lr=0.01          | lr=0.01        | lr=0.01        | lr=0.01          |
| feddyn        | CNN       | lr=0.1, α=0.1    | lr=0.1, α=0.1    | lr=0.1, α=0.1  | lr=0.05, α=0.1 | lr=0.05, α=0.1   |
| moon          | CNN       | lr=0.01, μ=1.0   | lr=0.05, μ=0.1   | lr=0.05, μ=1.0 | lr=0.05, μ=0.1 | lr=0.1, μ=0.1    |


#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
*Global Test*

| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 99.20±0.00 | 99.05±0.01 | 99.01±0.05 | 98.87±0.07 | 98.31±0.06 |
| fedprox       | CNN       | 99.22±0.03 | 99.05±0.03 | 99.06±0.04 | 98.90±0.02 | 98.36±0.07 |
| scaffold      | CNN       | 99.11±0.02 | 99.13±0.02 | 99.15±0.02 | 99.20±0.03 | 99.11±0.02 |
| feddyn        | CNN       | 99.37±0.01 | 99.23±0.02 | 99.25±0.03 | 99.20±0.04 | 98.89±0.04 |
| moon          | CNN       | 99.07±0.04 | 99.05±0.04 | 99.10±0.08 | 99.00±0.05 | 98.43±0.07 |

*Local Test*

| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 98.87±0.03 | 98.97±0.03 | 99.07±0.07 | 98.67±0.06 | 97.96±0.09 |
| fedprox       | CNN       | 98.89±0.05 | 98.91±0.07 | 99.02±0.04 | 98.75±0.03 | 97.97±0.06 |
| scaffold      | CNN       | 98.97±0.02 | 98.96±0.02 | 99.15±0.02 | 98.87±0.05 | 98.90±0.05 |
| feddyn        | CNN       | 99.10±0.02 | 99.11±0.02 | 99.35±0.03 | 98.99±0.02 | 98.66±0.06 |
| moon          | CNN       | 98.79±0.04 | 98.89±0.02 | 99.12±0.01 | 98.73±0.02 | 98.08±0.13 |

#### Impact of Sampling Ratio

| **Task** | **Algorithm** | **model** | **p=0.1**  | **p=0.2**  | **p=0.5**  | **p=1.0**  |  
|----------|---------------|-----------|------------|------------|------------|------------|
| iid      | fedavg        | CNN       | 99.20±0.03 | 99.20±0.00 | 99.21±0.02 | 99.22±0.00 |
| dir5.0   | fedavg        | CNN       | 99.00±0.03 | 99.05±0.01 | 99.04±0.01 | 99.05±0.01 |
| dir2.0   | fedavg        | CNN       | 99.01±0.04 | 99.01±0.05 | 99.04±0.02 | 99.05±0.01 |
| dir1.0   | fedavg        | CNN       | 98.94±0.01 | 98.87±0.07 | 98.90±0.03 | 98.94±0.00 |
| dir0.1   | fedavg        | CNN       | 98.27±0.08 | 98.31±0.06 | 98.32±0.04 | 98.33±0.02 |

#### Impact of Local Epoch

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## FEMNIST
### 3597 Clients

| client-id            |
| :-------------------------:|
| <img src="/task/femnist_c3597/res.png" alt="Alt text" width="500" height="300"> |

```
learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 2000
num_epochs: 5
clip_grad: 10
proportion: 0.2
early_stop: 400
train_holdout: 0.2
local_test: True
no_log_console: True
log_file: True
```

| **Algorithm** | **model** | **client-id**    |   
|---------------|-----------|------------------|
| fedavg        | CNN       | lr=0.1           |
| fedprox       | CNN       | lr=0.1, μ=0.0001 |
| scaffold      | CNN       | lr=0.05          |
| feddyn        | CNN       | lr=0.05, α=0.03  |



#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
```
*Global Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | CNN       | 86.25±0.04    |
| fedprox       | CNN       | 86.24±0.02    |
| scaffold      | CNN       | 86.92±0.07    |
| feddyn        | CNN       | 86.90±0.05    |
| moon          | CNN       |               |


*Local Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | CNN       | 84.91±0.05    |
| fedprox       | CNN       | 84.76±0.06    |
| scaffold      | CNN       | 87.39±0.08    |
| feddyn        | CNN       | 87.30±0.08    |
| moon          | CNN       |               |



<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## AGNEWS
### 100 Clients

|                                        iid                                        | dir1.0            |                                        dir0.1                                        |
|:---------------------------------------------------------------------------------:| :-------------------------:|:------------------------------------------------------------------------------------:|
| <img src="/task/agnews_iid_c100/res.png" alt="Alt text" width="500" height="300"> |  <img src="/task/agnews_dir1.0_c100/res.png" alt="Alt text" width="500" height="300"> | <img src="/task/agnews_dir0.1_c100/res.png" alt="Alt text" width="500" height="300"> |


```
learning_rate: [0.1, 0.5, 1.0, 5.0, 10.0]
batch_size: 50
weight_decay: 1e-4
momentum: 0.9
lr_scheduler: 0
learning_rate_decay: 0.998
num_rounds: 600
num_epochs: 1
clip_grad: 10
proportion: 0.2
early_stop: 125
train_holdout: 0.2
local_test: True
no_log_console: True
log_file: True
```

| **Algorithm** | **model**           | **iid**        | **dir1.0**     | **dir0.1**       |     
|---------------|---------------------|----------------|----------------|------------------|
| fedavg        | EmbeddingBag+Linear | lr=1.0         | lr=1.0         | lr=0.5           |
| fedprox       | EmbeddingBag+Linear | lr=1.0, μ=0.01 | lr=1.0, μ=0.01 | lr=0.5, μ=0.0001 |
| scaffold      | EmbeddingBag+Linear | lr=1.0         | lr=1.0         | lr=0.5           |
| feddyn        | EmbeddingBag+Linear | lr=0.1, α=0.01 | lr=0.5, α=0.01 | lr=0.5, α=0.01   |
| moon          | EmbeddingBag+Linear | lr=1.0, μ=10.0 | lr=0.5, μ=10.0 | lr=0.5, μ=0.1    |


#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
*Global Test*

| **Algorithm** | **model**           | **iid**    | **dir1.0** | **dir0.1** |   
|---------------|---------------------|------------|------------|------------|
| fedavg        | EmbeddingBag+Linear | 90.93±0.06 | 89.37±0.14 | 87.87±0.15 |
| fedprox       | EmbeddingBag+Linear | 90.94±0.11 | 89.39±0.12 | 87.97±0.11 |
| scaffold      | EmbeddingBag+Linear | 90.28±0.28 | 87.98±0.65 | 85.96±0.45 |
| feddyn        | EmbeddingBag+Linear | 91.04±0.06 | 91.02±0.02 | 91.14±0.04 |
| moon          | EmbeddingBag+Linear | 91.54±0.11 | 90.28±0.07 | 87.61±0.13 |

*Local Test*

| **Algorithm** | **model**           | **iid**    | **dir1.0** | **dir0.1** |  
|---------------|---------------------|------------|------------|------------|
| fedavg        | EmbeddingBag+Linear | 91.53±0.03 | 89.84±0.08 | 88.35±0.16 |
| fedprox       | EmbeddingBag+Linear | 91.50±0.01 | 89.87±0.06 | 88.30±0.20 |
| scaffold      | EmbeddingBag+Linear | 90.58±0.50 | 88.31±0.69 | 86.50±0.49 |
| feddyn        | EmbeddingBag+Linear | 91.59±0.04 | 91.11±0.01 | 91.04±0.01 |
| moon          | EmbeddingBag+Linear | 92.03±0.05 | 90.64±0.04 | 88.08±0.19 |

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## Office-Caltech10
### 4 Clients

| domain           |
| :-------------------------:|
|  <img src="/task/office_caltech10_c4/res.png" alt="Alt text" width="400" height="500"> |

```
learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 500
num_epochs: 1
clip_grad: 10
sample: full
proportion: 1.0
early_stop: 100
train_holdout: 0.2
local_test: True
no_log_console: True
log_file: True
```

| **Algorithm** | **model** | **domain**     |   
|---------------|-----------|----------------|
| fedavg        | AlexNet   | lr=0.1         |
| fedprox       | AlexNet   | lr=0.1, μ=0.1  |
| scaffold      | AlexNet   | lr=0.1         |
| feddyn        | AlexNet   | lr=0.01, α=0.1 |
| moon          | AlexNet   | lr=0.05, μ=1.0 |

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
```

*Local Test*

| **Algorithm** | **model** | **$\textbf{Client}_{Caltech}$** | **$\textbf{Client}_{Amazon}$** | **$\textbf{Client}_{Dslr}$** | **$\textbf{Client}_{Webcam}$** | **Mean**   | **Weighted-Mean** |      
|---------------|-----------|---------------------------------|--------------------------------|------------------------------|--------------------------------|------------|-------------------|
| fedavg        | AlexNet   | 71.25±1.99                      | 88.00±0.84                     | 80.00±0.00                   | 84.14±4.14                     | 80.85±1.37 | 79.63±1.11        |
| fedprox       | AlexNet   | 72.68±2.37                      | 72.68±2.37                     | 82.67±3.27                   | 93.10±3.08                     | 83.90±1.66 | 81.15±1.56        |
| scaffold      | AlexNet   | 70.54±2.33                      | 86.32±1.76                     | 88.00±4.99                   | 93.79±3.38                     | 84.66±2.23 | 80.30±1.79        |
| feddyn        | AlexNet   | 73.75±2.91                      | 84.00±1.68                     | 88.00±4.99                   | 95.17±1.69                     | 85.23±1.96 | 81.00±1.79        |
| moon          | AlexNet   | 71.61±2.55                      | 87.16±1.23                     | 78.67±2.67                   | 88.97±3.38                     | 81.60±1.06 | 79.95±1.27        |


<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## DomainNet
### 6 Clients

| domain           |
| :-------------------------:|
|  <img src="/task/domainnet_c6/res.png" alt="Alt text" width="500" height="300"> |

```
learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 500
num_epochs: 1
clip_grad: 10
sample: full
proportion: 1.0
early_stop: 100
train_holdout: 0.2
local_test: True
no_log_console: True
log_file: True
```

| **Algorithm** | **model** | **domain**     |   
|---------------|-----------|----------------|
| fedavg        | AlexNet   | lr=0.1         |
| fedprox       | AlexNet   | lr=0.1, μ=0.01 |
| scaffold      | AlexNet   | lr=0.1         |
| feddyn        | AlexNet   | lr=0.05, α=0.1 |
| moon          | AlexNet   | lr=0.1, μ=0.1  |

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
```


*Local Test*

| **Algorithm** | **model** | **Clipart** | **Infograph** | **Painting** | **Quickdraw** | **Real**   | **Sketch**  | **Mean**    | **Weighted-Mean** |   
|---------------|-----------|-------------|---------------|--------------|---------------|------------|-------------|-------------|-------------------|
| fedavg        | AlexNet   | 74.54±0.96  | 38.42±0.63    | 64.56±0.60   | 78.44±0.92    | 74.67±0.32 | 72.76±0.92  | 67.23±0.20  | 68.90±0.20        |
| fedprox       | AlexNet   | 75.46±1.34  | 38.51±0.88    | 64.56±1.26   | 77.39±1.82    | 74.04±0.57 | 73.28±1.16  | 67.20±0.63  | 68.67±0.66        |
| scaffold      | AlexNet   | 78.25±0.36  | 40.77±0.41    | 66.70±1.39   | 78.77±0.90    | 75.12±0.92 | 77.09±0.69  | 69.45±0.41  | 70.65±0.44        |
| feddyn        | AlexNet   | 78.55±1.04  | 39.82±0.71    | 67.18±1.20   | 78.23±0.30    | 74.34±0.67 | 75.40±1.32  | 68.92±0.22  | 70.09±0.17        |
| moon          | AlexNet   | 72.77±0.58  | 38.46±1.04    | 63.38±0.74   | 79.75±1.40    | 73.83±0.48 | 70.66±0.45  | 66.48±0.54  | 68.35±0.58        |


<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## PACS
### 4 Clients

|                                  domain                                   |
|:-------------------------------------------------------------------------:|
| <img src="/task/PACS_c4/res.png" alt="Alt text" width="500" height="400"> |

```
learning_rate: [0.001, 0.01, 0.05, 0.1, 0.5]
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 500
num_epochs: 5
clip_grad: 10
proportion: 1.0
early_stop: 100
train_holdout: 0.2
local_test: True
no_log_console: True
log_file: True
```


| **Algorithm** | **model** | **domain**        |   
|---------------|-----------|-------------------|
| fedavg        | AlexNet   | lr=0.05           |
| fedprox       | AlexNet   | lr=0.05, μ=0.0001 |
| scaffold      | AlexNet   | lr=0.1            |
| feddyn        | AlexNet   | lr=0.1, α=0.03    |
| moon          | AlexNet   | lr=0.05, μ=1.0    |

*Local Test*

| **Algorithm** | **model** | **$\textbf{Client}_{ArtPainting}$** | **$\textbf{Client}_{Cartoon}$** | **$\textbf{Client}_{Photo}$** | **$\textbf{Client}_{Sketch}$** | **Mean**   | **Weighted-Mean** |      
|---------------|-----------|-------------------------------------|---------------------------------|-------------------------------|--------------------------------|------------|-------------------|
| fedavg        | AlexNet   | 61.47±1.47                          | 84.53±2.07                      | 71.14±3.54                    | 82.65±1.56                     | 74.95±0.73 | 76.83±0.78        |
| fedprox       | AlexNet   | 65.78±2.55                          | 81.88±0.96                      | 76.05±2.18                    | 80.97±1.03                     | 76.17±0.78 | 77.25±0.46        |
| scaffold      | AlexNet   |                                     |                                 |                               |                                |            |                   |
| feddyn        | AlexNet   |                                     |                                 |                               |                                |            |                   |
| moon          | AlexNet   |                                     |                                 |                               |                                |            |                   |


<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## Digits
### 5 Clients

|                                   domain                                    |
|:---------------------------------------------------------------------------:|
| <img src="/task/digits_c5/res.png" alt="Alt text" width="800" height="200"> |

```
```

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>


## ProstateMRI
### 6 Clients

|                                      domain                                      |
|:--------------------------------------------------------------------------------:|
| <img src="/task/prostateMRI_c6/res.png" alt="Alt text" width="600" height="300"> |

```
learning_rate: [0.00005, 0.0001, 0.0005, 0.001, 0.005]
batch_size: 16
weight_decay: 1e-4
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 500
num_epochs: 1
clip_grad: 10
proportion: 1.0
early_stop: 100
train_holdout: 0.2
local_test: True
optimizer: Adam
no_log_console: True
log_file: True
```

| **Algorithm** | **model** | **domain**          |   
|---------------|-----------|---------------------|
| fedavg        | UNet      | lr=0.0001           |
| fedprox       | UNet      | lr=0.0001, μ=0.0001 |
| scaffold      | UNet      | lr=0.0001           |
| feddyn        | UNet      | lr=0.00005, α=0.1   |
| moon          | UNet      | lr=0.0001, μ=0.1    |

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

| **Algorithm** | **model** | **BMC**    | **UCL**    | **BIDMC**  | **RUNMC**  | **HK**     | **I2CVB**  | **Mean**    | **Weighted-Mean** |   
|---------------|-----------|------------|------------|------------|------------|------------|------------|-------------|-------------------|
| fedavg        | UNet      | 91.09±0.69 | 90.74±0.99 | 93.02±0.69 | 94.32±0.39 | 94.84±0.54 | 95.98±0.10 | 93.33±0.12  | 93.59±0.13        |
| fedprox       | UNet      | 91.94±0.67 | 90.83±0.70 | 93.27±0.29 | 94.87±0.19 | 94.84±0.29 | 95.67±0.32 | 93.57±0.23  | 93.86±0.24        |
| scaffold      | UNet      | 56.16±0.09 | 49.23±0.03 | 51.56±0.02 | 54.69±0.06 | 53.50±0.05 | 46.92±0.07 | 52.01±0.02  | 51.98±0.02        |
| feddyn        | UNet      | 91.41±1.14 | 90.96±1.49 | 91.89±1.49 | 94.28±0.38 | 93.93±0.85 | 93.83±1.00 | 92.72±0.28  | 92.90±0.25        |
| moon          | UNet      | 91.93±0.36 | 89.85±1.10 | 92.39±0.63 | 94.30±0.43 | 94.06±0.29 | 96.17±0.28 | 93.12±0.23  | 93.57±0.20        |

## Camelyon17
### 5 Clients

|                                    hospital                                     |
|:-------------------------------------------------------------------------------:|
| <img src="/task/camelyon17_c5/res.png" alt="Alt text" width="800" height="160"> |

```
```

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>


## Fundus
### 4 Clients

|                                  hospital                                   |
|:---------------------------------------------------------------------------:|
| <img src="/task/fundus_c4/res.png" alt="Alt text" width="500" height="500"> |

```
```

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>


## EndoPolyp
### 5 Clients

|                                    hospital                                    |
|:------------------------------------------------------------------------------:|
| <img src="/task/endopolyp_c5/res.png" alt="Alt text" width="800" height="160"> |

```
```

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>


## SpeechCommand
### 2112 Clients

| client-id            |
| :-------------------------:|
|  <img src="/task/speechcommand_c2112/res.png" alt="Alt text" width="500" height="300">  |


```
learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.998
num_rounds: 2000
num_epochs: 1
clip_grad: 10
proportion: 0.05
early_stop: 250
train_holdout: 0.0
no_log_console: True
log_file: True
```

| **Algorithm** | **model** | **client-id**    |   
|---------------|-----------|------------------|
| fedavg        | M5        | lr=1.0           |
| fedprox       | M5        | lr=1.0, mu=0.01  |
| scaffold      | M5        | lr=1.0           |
| feddyn        | M5        | lr=0.1, α=0.001  |
| moon          | M5        | lr=1.0, mu=0.1   |

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
```

*Global Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | M5        | 69.11±0.91    |
| fedprox       | M5        | 69.30±0.87    |
| scaffold      | M5        | 64.40±0.42    |
| feddyn        | M5        | 60.65±0.76    |
| moon          | M5        | 69.08±0.86    |

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## Shakespeare
### 1012 Clients

| client-id            |
| :-------------------------:|
| <img src="/task/shakespeare_c1012/res.png" alt="Alt text" width="500" height="300">  |


```
learning_rate: [0.1, 0.5, 1.0, 5.0, 10.0]
batch_size: 50
weight_decay: 5e-4
lr_scheduler: 0
learning_rate_decay: 0.9998
num_rounds: 200
num_epochs: 5
clip_grad: 10
proportion: 0.1
early_stop: 100
train_holdout: 0.2
local_test: True
no_log_console: True
log_file: True
```

| **Algorithm** | **model** | **client-id**    |   
|---------------|-----------|------------------|
| fedavg        | LSTM      | lr=0.1           |
| fedprox       | LSTM      | lr=0.1, μ=0.0001 |
| scaffold      | LSTM      | lr=0.5           |


#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
```

*Global Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | LSTM      | 52.85±0.06    |
| fedprox       | LSTM      | 53.09±0.06    |
| scaffold      | LSTM      | 49.93±0.09    |



*Local Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | LSTM      | 52.76±0.17    |
| fedprox       | LSTM      | 53.31±0.04    |
| scaffold      | LSTM      | 50.01±0.14    |


<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

