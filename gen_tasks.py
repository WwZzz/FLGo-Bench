import flgo.benchmark.partition as fbp
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.cifar100_classification as cifar100
import flgo

# flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(alpha=1.0, num_clients=20), 'cifar10_dir1_c20')
# flgo.gen_task_by_(cifar100, fbp.DirichletPartitioner(alpha=1.0, num_clients=20, error_bar=1e-9), 'cifar100_dir1_c20')