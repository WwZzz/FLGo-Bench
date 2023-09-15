import flgo.benchmark.partition as fbp
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.cifar100_classification as cifar100
import flgo

# # 10.24.81.135
# flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(alpha=1.0, num_clients=20), 'cifar10_dir1_c20')

# # 10.24.80.246
# flgo.gen_task_by_(cifar100, fbp.DirichletPartitioner(alpha=1.0, num_clients=20, error_bar=1e-9), 'cifar100_dir1_c20')

# # 10.24.116.58
# flgo.gen_task_by_(cifar100, fbp.DirichletPartitioner(alpha=0.1, num_clients=20, error_bar=1e-9), 'cifar100_dir.1_c20')
#
# # 10.24.116.59
# flgo.gen_task_by_(cifar100, fbp.IIDPartitioner(num_clients=20), 'cifar100_iid_c20')
#
# # 10.24.116.60
# flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(alpha=0.1, num_clients=20, error_bar=1e-9), 'cifar10_dir.1_c20')
#
# # 10.24.81.135
# flgo.gen_task_by_(cifar10, fbp.IIDPartitioner(num_clients=20), 'cifar10_iid_c20')

############################################ 2023 - 09 - 14
# # 10.24.116.60
# flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(alpha=0.1, num_clients=100, error_bar=1e-9), 'cifar10_dir.1_c100')
#
# # 10.24.116.60
# flgo.gen_task_by_(cifar10, fbp.IIDPartitioner(num_clients=100), 'cifar10_iid_c100')
#
# # 10.24.116.58
# flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(alpha=1.0, num_clients=100), 'cifar10_dir1_c100')

# # 10.24.116.59
# flgo.gen_task_by_(cifar100, fbp.DirichletPartitioner(alpha=1.0, num_clients=100, error_bar=1e-9), 'cifar100_dir1_c100')

# # 10.24.116.58
# flgo.gen_task_by_(cifar100, fbp.DirichletPartitioner(alpha=0.1, num_clients=100, error_bar=1e-9), 'cifar100_dir.1_c100')
#
# # 10.24.116.59
# flgo.gen_task_by_(cifar100, fbp.IIDPartitioner(num_clients=100), 'cifar100_iid_c100')
#



