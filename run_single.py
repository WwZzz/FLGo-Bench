import argparse
import warnings
import logger
import flgo
from flgo.experiment.logger import BasicLogger
import numpy as np
import yaml
import importlib
import os

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='name of task', type=str, default='')
    parser.add_argument('--algorithm', help='name of method', type=str, default='fedavg')
    parser.add_argument('--model', help = 'model name', type=str, default='')
    parser.add_argument('--gpu', help='GPU IDs and empty input is equal to using CPU', type=int, default=[0])
    parser.add_argument('--config', type=str, help='configuration of hypara', default='')
    parser.add_argument('--logger', help='test parallel', type=str, default='SimpleLogger')
    parser.add_argument('--check_interval', help='interval (s) to save checkpoints', type=int, default=-1)
    parser.add_argument('--load_mode', help = 'load_mode', type=str, default='')
    parser.add_argument('--num_client_parallel', help = 'num of client processes', type=int, default=0)
    parser.add_argument('--test_parallel', help='test parallel',  action="store_true", default=False)
    parser.add_argument('--data_root', help = 'the root of dataset', type=str, default='')
    return parser.parse_known_args()

args = read_args()[0]
task = args.task
gpus = args.gpu
if args.data_root!='':
    if os.path.exists(args.data_root) and os.path.isdir(args.data_root):
        flgo.set_data_root(args.data_root)
    else:
        warnings.warn("Failed to change data root.")
config = args.config
if config!='' and os.path.exists(config):
    with open(config, 'r') as inf:
        option = yaml.load(inf, Loader=yaml.FullLoader)
else:
    option = {}
option['gpu'] = gpus
optimal_option = option

class FullLogger(BasicLogger):
    def log_once(self, *args, **kwargs):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        val_metric = self.server.test(flag='val')
        for met_name, met_val in val_metric.items():
            self.output['val_' + met_name].append(met_val)
        # calculate weighted averaging of metrics on training datasets across participants
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        local_val_metrics = self.server.global_test(flag='val')
        for met_name, met_val in local_val_metrics.items():
            self.output['local_val_'+met_name+'_dist'].append(met_val)
            self.output['local_val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_val_' + met_name].append(np.mean(met_val))
            self.output['std_local_val_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()

if __name__=='__main__':
    algo = None
    acce = False
    modules = [".".join(["algorithm", args.algorithm]), ".".join(["develop",  args.algorithm]),".".join(["flgo", "algorithm",  args.algorithm])]
    if args.num_client_parallel>0:
        try:
            algo = importlib.import_module(".".join(["algorithm", "accelerate", args.algorithm]))
            acce = True
        except:
            algo = None
            warnings.warn(f"There is no acceleration support for {args.algorithm}")
    if algo is None:
        for m in modules:
            try:
                algo = importlib.import_module(m)
                break
            except ModuleNotFoundError:
                continue
    if algo is None: raise ModuleNotFoundError("{} was not found".format(algo))
    model  = None
    if args.model != '':
        try:
            model = getattr(algo, args.model)
        except:
            model = None
        if model is None:
            try:
                model = importlib.import_module(args.model)
            except:
                print("using default model")
                model = None
    optimal_option['load_mode'] = args.load_mode
    if acce and args.num_client_parallel>0:
        optimal_option['num_parallels'] =args.num_client_parallel
        optimal_option['parallel_type'] = 'else'
    if args.check_interval>0:
        optimal_option['check_interval'] = args.check_interval
        optimal_option['save_checkpoint'] = algo.__name__
        optimal_option['load_checkpoint'] = algo.__name__
    if args.test_parallel: optimal_option['test_parallel'] = True
    Logger = getattr(logger, args.logger)
    flgo.init(os.path.join('task', task), algo, optimal_option, model=model, Logger=Logger).run()
