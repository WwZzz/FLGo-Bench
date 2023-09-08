import flgo.benchmark.partition as fbp
import flgo.benchmark.cifar100_classification as cifar100
import flgo

flgo.gen_task_by_(cifar100, fbp.DirichletPartitioner(alpha=1.0, num_clients=20, error_bar=1e-8), 'CIFAR100_Dirichlet1.0_Clients20_IMB0.0')