import flgo.benchmark.partition as fbp
import flgo.benchmark.cifar10_classification as cifar10
import flgo
import os
#######################################################CIFAR10
if not os.path.exists('CIFAR100_Dirichlet1.0_Clients20_IMB0.0'):
    flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(alpha=1.0, num_clients=20), 'CIFAR100_Dirichlet1.0_Clients20_IMB0.0')