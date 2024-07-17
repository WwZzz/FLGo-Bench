# FLGo-Bench
Produce results of federated algorithms on various benchmarks

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
| fedavg        | CNN       | lr=0.1      |            |            |            |            |