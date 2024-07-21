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

| **Algorithm** | **model** | **iid**          | **dir5.0** | **dir2.0** | **dir1.0**     | **dir0.1** |    
|---------------|-----------|------------------|------------|------------|----------------|------------|
| fedavg        | CNN       | lr=0.1           | lr=0.1     | lr=0.05    | lr=0.1         | lr=0.1     |
| fedprox       | CNN       | lr=0.05, μ=0.001 |            |            | lr=0.1, μ=0.01 |            |

```
fedprox: μ ∈ [0.0001, 0.001, 0.01, 0.1, 1.0]
```

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
Global Test

| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 81.54±0.14 | 79.95±0.22 | 78.00±0.24 | 76.54±0.39 | 69.87±0.50 |
| fedprox       | CNN       | 80.83±0.20 |            |            |            |            |

Local Test

| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 80.98±0.39 | 79.97±0.12 | 77.43±0.12 | 77.03±0.29 | 70.02±0.70 |
| fedprox       | CNN       | 80.68±0.18 |            |            |            |            |

#### Impact of Sampling Ratio

| **Task** | **Algorithm** | **model** | **p=0.1**  | **p=0.2**  | **p=0.5**  | **p=1.0**  |  
|----------|---------------|-----------|------------|------------|------------|------------|
| iid      | fedavg        | CNN       | 81.70±0.30 | 81.54±0.14 | 81.34±0.23 | 81.87±0.17 |

#### Impact of Local Epoch
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

| **Algorithm** | **model** | **iid**          | **dir5.0**       | **dir2.0**    | **dir1.0**    | **dir0.1**      |    
|---------------|-----------|------------------|------------------|---------------|---------------|-----------------|
| fedavg        | CNN       | lr=0.1           | lr=0.05          | lr=0.05       | lr=0.05       | lr=0.05         |
| fedprox       | CNN       | lr=0.1, μ=0.0001 | lr=0.05, μ=0.001 | lr=0.1, μ=1.0 | lr=0.1, μ=1.0 | lr=0.05, μ=0.01 |

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
Global Test

| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 99.20±0.00 | 99.05±0.01 | 99.01±0.05 | 98.87±0.07 | 98.31±0.06 |

Local Test

| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 98.87±0.03 | 98.97±0.03 | 99.07±0.07 | 98.67±0.06 | 97.96±0.09 |

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


#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
Global Test

| **Algorithm** | **model**           | **dir1.0** |   
|---------------|---------------------|------------|
| fedavg        | EmbeddingBag+Linear | 89.37±0.14 |

Local Test

| **Algorithm** | **model**           | **dir1.0** |   
|---------------|---------------------|------------|
| fedavg        | EmbeddingBag+Linear | 89.84±0.08 |