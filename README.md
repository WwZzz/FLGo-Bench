# FLGo-Bench
Produce results of federated algorithms on various benchmarks

# Result
## CIFAR10
### 100 Clients
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

| **Algorithm** | **model** | **iid**          | **dir5.0**     | **dir2.0**      | **dir1.0**     | **dir0.1**       |    
|---------------|-----------|------------------|----------------|-----------------|----------------|------------------|
| fedavg        | CNN       | lr=0.1           | lr=0.1         | lr=0.05         | lr=0.1         | lr=0.1           |
| fedprox       | CNN       | lr=0.05, μ=0.001 | lr=0.1, μ=0.01 | lr=0.1, μ=0.001 | lr=0.1, μ=0.01 | lr=0.05, μ=0.001 |
| scaffold      | CNN       | lr=0.1           | lr=0.1         | lr=0.1          | lr=0.1         | lr=0.1           |
| moon          | CNN       | lr=0.1, μ=0.1    | lr=0.1, μ=0.1  | lr=0.05, μ=0.1  | lr=0.1, μ=0.1  | lr=0.1, μ=0.1    |
| feddyn        | CNN       | lr=0.1, α=0.1    | lr=0.1, α=0.1  | lr=0.05, α=0.1  | lr=0.1, α=0.03 | lr=0.05, α=0.03  |

```
fedprox: μ ∈ [0.0001, 0.001, 0.01, 0.1, 1.0]
moon: μ ∈ [0.1, 1.0, 5.0, 10.0]
```

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

## CIFAR100
### 100 Clients
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

| **Algorithm** | **model** | **iid**         | **dir1.0**      | **dir0.1**        | 
|---------------|-----------|-----------------|-----------------|-------------------|
| fedavg        | CNN       | lr=0.1          | lr=0.1          | lr=0.1            | 
| fedprox       | CNN       | lr=0.1, μ=0.001 | lr=0.1, μ=0.001 | lr=0.05, μ=0.0001 |
| scaffold      | CNN       | lr=0.1          | lr=0.1          | lr=0.1            |
| feddyn        | CNN       | lr=0.001, α=0.1 | lr=0.1, α=0.1   | lr=0.1, α=0.03    | 
| moon          | CNN       | lr=0.1, μ=0.1   | lr=0.1, μ=0.1   | lr=0.05, μ=0.1    | 


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

## MNIST
### 100 Clients
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

## AGNEWS
### 100 Clients
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

## DomainNet
### 6 Clients
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

*Local Test*

| **Algorithm** | **model** | **domain** |   
|---------------|-----------|------------|
| fedavg        | AlexNet   | 71.06±0.73 |
| fedprox       | AlexNet   | 71.04±0.44 |

*Sized-weighted Local Test*

| **Algorithm** | **model** | **domain** |   
|---------------|-----------|------------|
| fedavg        | AlexNet   | 72.68±0.65 |
| fedprox       | AlexNet   | 72.40±0.49 |
'

## SpeechCommand
### 2112 Clients
```
learning_rate: [0.001, 0.005, 0.01, 0.05, 0.1]
batch_size: 50
weight_decay: 1e-3
lr_scheduler: 0
learning_rate_decay: 0.998
num_rounds: 1000
num_epochs: 1
clip_grad: 10
proportion: 0.05
early_stop: 250
train_holdout: 0.0
no_log_console: True
log_file: True
pin_memory: True
```

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | M5        | lr=1.0        |
| scaffold      | M5        | lr=1.0        |

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
```

*Global Test*

| **Algorithm** | **model** | **client-id** |   
|---------------|-----------|---------------|
| fedavg        | M5        | 69.11±0.91    |
| scaffold      | M5        | 64.40±0.42    |

## Shakespeare
### 1012 Clients
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
