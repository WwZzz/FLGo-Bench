from .model import vit, default_model
import flgo.benchmark.toolkits.visualization
import flgo.benchmark.toolkits.partition

default_model = vit
default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
visualize = flgo.benchmark.toolkits.visualization.visualize_by_class

