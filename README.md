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
```
python tune.py --task TASKNAME --algorithm ALGORITHM --config CONFIG_PATH --gpu GPUids 
```

- **Running Command**
```
python run.py --task TASKNAME --algorithm ALGORITHM --config CONFIG_PATH --gpu GPUids 
```
- **Optional Args**

| **Name**            | **Type** | **Desc.**                                                                                                   |   
|---------------------|----------|-------------------------------------------------------------------------------------------------------------|
| model               | str      | the file name in the dictionary `model/` that denotes a legal model in FLGo                                 |
| load_mode           | str      | be one of ['', 'mmap', 'mem'], which respectively denotes DefaultIO, MemmapIO, and InMemory Dataset Loading |
| max_pdev            | int      | the maximum number of processes on each gpu device                                                          |
| available_interval  | int      | the time interval (s) to check whether a device is available                                                |
| put_interval        | int      | the time interval (s) to put one process into device                                                        |
| seq                 | bool     | whether to run each process in sequencial                                                                   |
| num_client_parallel | int      | the number of parallel client local training processes, default is 0                                        |
| test_parallel       | bool     | whether to use data parallel when evaluating model                                                          |

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
python performance_analyze.py --task TASK --algorithm ALGORITHM --model MODEL --config CONFIG_PATH 

# Show Running Result
python show_result.py --task TASK --algorithm ALGORITHM --model MODEL --config CONFIG_PATH 
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
- [MNIST](#MNIST)
- [FEMNIST](#FEMNIST)
- [AgNews](#AGNEWS)
- [Office-Caltech10](#Office-Caltech10)
- [DomainNet](#DomainNet)
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

| **Algorithm** | **model**   | **iid**          | **dir5.0**     | **dir2.0**      | **dir1.0**     | **dir0.1**       |    
|---------------|-------------|------------------|----------------|-----------------|----------------|------------------|
| fedavg        | CNN         | lr=0.1           | lr=0.1         | lr=0.05         | lr=0.1         | lr=0.1           |
| fedprox       | CNN         | lr=0.05, μ=0.001 | lr=0.1, μ=0.01 | lr=0.1, μ=0.001 | lr=0.1, μ=0.01 | lr=0.05, μ=0.001 |
| scaffold      | CNN         | lr=0.1           | lr=0.1         | lr=0.1          | lr=0.1         | lr=0.1           |
| moon          | CNN         | lr=0.1, μ=0.1    | lr=0.1, μ=0.1  | lr=0.05, μ=0.1  | lr=0.1, μ=0.1  | lr=0.1, μ=0.1    |
| feddyn        | CNN         | lr=0.1, α=0.1    | lr=0.1, α=0.1  | lr=0.05, α=0.1  | lr=0.1, α=0.03 | lr=0.05, α=0.03  |
|               |             |                  |                |                 |                |                  |
| fedavg        | ResNet18    |                  |                |                 |                |                  |
| fedavg        | ResNet18-GN | lr=0.1           | lr=0.1         | lr=0.1          | lr=0.1         |                  |



#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
*Global Test*

| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 81.54±0.14 | 79.95±0.22 | 78.00±0.24 | 76.54±0.39 | 69.87±0.50 |
| fedprox       | CNN       | 80.83±0.20 | 79.86±0.30 | 78.39±0.26 | 76.58±0.50 | 69.41±0.33 |
| scaffold      | CNN       | 85.08±0.21 | 83.84±0.23 | 82.15±0.29 | 80.42±0.21 | 65.06±0.51 |
| moon          | CNN       | 80.88±0.27 | 79.63±0.20 | 77.21±0.33 | 75.67±0.26 | 62.44±1.10 |
| feddyn        | CNN       | 85.09±0.22 | 83.65±0.09 | 81.54±0.30 | 80.26±0.40 | 70.82±0.50 |

*Local Test*

| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 80.98±0.39 | 79.97±0.12 | 77.43±0.12 | 77.03±0.29 | 70.02±0.70 |
| fedprox       | CNN       | 80.68±0.18 | 79.81±0.41 | 77.80±0.13 | 77.05±0.20 | 69.72±0.79 |
| scaffold      | CNN       | 85.22±0.33 | 84.23±0.44 | 81.90±0.28 | 80.23±0.19 | 65.00±0.66 |
| moon          | CNN       | 80.26±0.38 | 79.29±0.29 | 76.81±0.58 | 76.12±0.45 | 62.26±1.09 |
| feddyn        | CNN       | 85.25±0.26 | 83.99±0.17 | 81.76±0.17 | 80.48±0.42 | 71.69±0.29 |

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
| ![cifar100_iid_img](/task/cifar100_iid_c100/res.png)  |  ![cifar100_d1_img](/task/cifar100_dir1.0_c100/res.png) |  ![cifar100_d0_img](/task/cifar10_dir0.1_c100/res.png) |
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

| **Algorithm** | **model**   | **iid**         | **dir1.0**      | **dir0.1**        | 
|---------------|-------------|-----------------|-----------------|-------------------|
| fedavg        | CNN         | lr=0.1          | lr=0.1          | lr=0.1            | 
| fedprox       | CNN         | lr=0.1, μ=0.001 | lr=0.1, μ=0.001 | lr=0.05, μ=0.0001 |
| scaffold      | CNN         | lr=0.1          | lr=0.1          | lr=0.1            |
| feddyn        | CNN         | lr=0.001, α=0.1 | lr=0.1, α=0.1   | lr=0.1, α=0.03    | 
| moon          | CNN         | lr=0.1, μ=0.1   | lr=0.1, μ=0.1   | lr=0.05, μ=0.1    | 
|               |             |                 |                 |                   |    
| fedavg        | ResNet18    | lr=0.1          |                 |                   |     
| fedavg        | ResNet18-GN | lr=0.1          |                 |                   | 

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
*Global Test*

| **Algorithm** | **model** | **iid**    | **dir1.0** | **dir0.1** | 
|---------------|-----------|------------|------------|------------|
| fedavg        | CNN       | 41.33±0.30 | 37.93±0.64 | 22.50±0.52 | 
| fedprox       | CNN       | 41.27±0.31 | 37.79±0.27 | 22.20±0.16 |
| scaffold      | CNN       | 49.85±0.22 | 41.08±0.39 | 18.56±0.50 | 
| feddyn        | CNN       | 52.64±0.15 | 40.19±0.36 | 26.20±0.44 | 
| moon          | CNN       | 41.49±0.40 | 37.28±0.37 | 21.09±0.32 |


*Local Test*


| **Algorithm** | **model** | **iid**    | **dir1.0** | **dir0.1** | 
|---------------|-----------|------------|------------|------------|
| fedavg        | CNN       | 41.04±0.34 | 36.94±0.52 | 22.21±0.51 | 
| fedprox       | CNN       | 40.62±0.35 | 37.00±0.50 | 21.77±0.21 |
| scaffold      | CNN       | 49.94±0.22 | 39.70±0.31 | 18.58±0.61 |
| feddyn        | CNN       | 52.48±0.49 | 38.92±0.37 | 26.08±0.15 | 
| moon          | CNN       | 40.91±0.30 | 36.22±0.34 | 20.67±0.26 |

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
| scaffold      | CNN       |                  |
| feddyn        | CNN       |                  |
| moon          | CNN       |                  |



#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
*Global Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | CNN       | lr=0.1        |
| fedprox       | CNN       |               |
| scaffold      | CNN       |               |
| feddyn        | CNN       |               |
| moon          | CNN       |               |


*Local Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | CNN       | lr=0.1        |
| fedprox       | CNN       |               |
| scaffold      | CNN       |               |
| feddyn        | CNN       |               |
| moon          | CNN       |               |



<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>


## AGNEWS
### 100 Clients

| dir1.0            |
| :-------------------------:|
|  <img src="/task/agnews_dir1.0_c100/res.png" alt="Alt text" width="500" height="300"> |


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

| **Algorithm** | **model**           | **dir1.0**     |   
|---------------|---------------------|----------------|
| fedavg        | EmbeddingBag+Linear | lr=1.0         |
| fedprox       | EmbeddingBag+Linear | lr=1.0, μ=0.01 |
| scaffold      | EmbeddingBag+Linear | lr=1.0         |
| feddyn        | EmbeddingBag+Linear | lr=0.5, α=0.01 |
| moon          | EmbeddingBag+Linear | lr=0.5, μ=10.0 |


#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
*Global Test*

| **Algorithm** | **model**           | **dir1.0** |   
|---------------|---------------------|------------|
| fedavg        | EmbeddingBag+Linear | 89.37±0.14 |
| fedprox       | EmbeddingBag+Linear | 89.39±0.12 |
| scaffold      | EmbeddingBag+Linear | 87.98±0.65 |
| feddyn        | EmbeddingBag+Linear | 91.02±0.02 |
| moon          | EmbeddingBag+Linear | 90.28±0.07 |

*Local Test*

| **Algorithm** | **model**           | **dir1.0** |   
|---------------|---------------------|------------|
| fedavg        | EmbeddingBag+Linear | 89.84±0.08 |
| fedprox       | EmbeddingBag+Linear | 89.87±0.06 |
| scaffold      | EmbeddingBag+Linear | 88.31±0.69 |
| feddyn        | EmbeddingBag+Linear | 91.11±0.01 |
| moon          | EmbeddingBag+Linear | 90.64±0.04 |

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

## Office-Caltech10
### 4 Clients

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

| **Algorithm** | **model** | **domain** |   
|---------------|-----------|------------|
| fedavg        | AlexNet   | 81.34±0.75 |
| fedprox       | AlexNet   | 83.21±0.78 |
| scaffold      | AlexNet   | 82.82±1.31 |
| feddyn        | AlexNet   | 83.61±1.66 |
| moon          | AlexNet   | 80.52±1.48 |

Size-Weighted *Local Test*

| **Algorithm** | **model** | **domain** |   
|---------------|-----------|------------|
| fedavg        | AlexNet   | 78.67±0.46 |
| fedprox       | AlexNet   | 78.06±1.01 |
| scaffold      | AlexNet   | 75.93±1.55 |
| feddyn        | AlexNet   | 76.18±1.76 |
| moon          | AlexNet   | 76.46±0.68 |

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
| fedprox       | AlexNet   | lr=0.1, μ=0.1  |

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
```


*Global Test*

| **Algorithm** | **model** | **domain** |   
|---------------|-----------|------------|
| fedavg        | AlexNet   | 71.44±0.35 |
| fedprox       | AlexNet   | 71.41±0.29 |
| scaffold      | AlexNet   | 72.31±0.64 |
| feddyn        | AlexNet   | 71.65±0.27 |
| moon          | AlexNet   | 71.37±0.18 |

*Local Test*

| **Algorithm** | **model** | **domain** |   
|---------------|-----------|------------|
| fedavg        | AlexNet   | 71.06±0.73 |
| fedprox       | AlexNet   | 71.04±0.44 |
| scaffold      | AlexNet   | 72.14±0.52 |
| feddyn        | AlexNet   | 70.79±0.47 |
| moon          | AlexNet   | 70.33±0.36 |

*Sized-weighted Local Test*

| **Algorithm** | **model** | **domain** |   
|---------------|-----------|------------|
| fedavg        | AlexNet   | 72.68±0.65 |
| fedprox       | AlexNet   | 72.40±0.49 |
| scaffold      | AlexNet   | 73.35±0.50 |
| feddyn        | AlexNet   | 72.20±0.30 |
| moon          | AlexNet   | 71.88±0.37 |

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

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | LSTM      | lr=0.1        |
| scaffold      | LSTM      | lr=0.5        |


#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
```

*Global Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | LSTM      | 52.85±0.06    |


*Local Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | LSTM      | 52.76±0.17    |

<div style="text-align: right;">
<a href="#Nevigation" style="text-decoration: none; background-color: #0366d6; color: white; padding: 5px 10px; border-radius: 5px;">Back</a>
</div>

