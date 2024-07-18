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

| **Algorithm** | **model** | **iid**     | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|-------------|------------|------------|------------|------------|
| fedavg        | CNN       | lr=0.1      |            |            |            |            |

#### Main Results
#### Impact of Sampling Ratio
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

| **Algorithm** | **model** | **iid**     | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|-------------|------------|------------|------------|------------|
| fedavg        | CNN       | lr=0.1      | lr=0.05    | lr=0.05    | lr=0.05    | lr=0.05    |

#### Main Results
```
seed: [2,4388,15,333,967] # results are averaged over five random seeds
proportion: 0.2
```
Global Test

| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 99.20±0.00 |            |            |            |            |

Local Test


| **Algorithm** | **model** | **iid**    | **dir5.0** | **dir2.0** | **dir1.0** | **dir0.1** |    
|---------------|-----------|------------|------------|------------|------------|------------|
| fedavg        | CNN       | 98.87±0.03 |            |            |            |            |

#### Impact of Sampling Ratio

| **Task** | **Algorithm** | **model** | **p=0.1**  | **p=0.2**  | **p=0.5**  | **p=1.0** |  
|----------|---------------|-----------|------------|------------|------------|-----------|
| iid      | fedavg        | CNN       | 99.20±0.03 | 99.20±0.00 | 99.21±0.02 | 99.22±0.0 |

#### Impact of Local Epoch