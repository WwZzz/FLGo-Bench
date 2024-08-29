from .model import vit, default_model, resnet18_gn, resnet18
import flgo.benchmark.toolkits.visualization
import flgo.benchmark.toolkits.partition

default_model = resnet18
default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
visualize = flgo.benchmark.toolkits.visualization.visualize_by_class

