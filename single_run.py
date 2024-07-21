import argparse
import flgo
from flgo.experiment.logger import BasicLogger
import flgo.experiment.device_scheduler as ds
import numpy as np
import torch.multiprocessing
import yaml
import importlib
import os
import time

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='name of task', type=str, default='')
    parser.add_argument('--algorithm', help='name of method', type=str, default='fedavg')
    parser.add_argument('--model', help = 'model name', type=str, default='')
    parser.add_argument('--gpu', help='GPU IDs and empty input is equal to using CPU', type=int, default=[0])
    parser.add_argument('--config', type=str, help='configuration of hypara', default='')
    return parser.parse_known_args()

args = read_args()[0]
task = args.task
gpus = args.gpu
config = args.config
with open(config, 'r') as inf:
    option = yaml.load(inf, Loader=yaml.FullLoader)
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


def fedrun(task, algo, optimal_option={}, seeds=[0], Logger=None, model=None, put_interval=10, available_interval=10, max_processes_per_device=10):
    runner_dict = []
    asc = ds.AutoScheduler(optimal_option['gpu'], put_interval=put_interval, available_interval=available_interval, max_processes_per_device=max_processes_per_device)
    for seed in seeds:
        opi = optimal_option.copy()
        opi.update({'seed': seed})
        runner_dict.append({'task': task, 'algorithm': algo, 'option': opi, 'model':model, 'Logger':Logger})
    res = flgo.multi_init_and_run(runner_dict, scheduler=asc)
    return res

if __name__=='__main__':
    algo = None
    modules = [".".join(["algorithm", args.algorithm]), ".".join(["develop",  args.algorithm]),".".join(["flgo", "algorithm",  args.algorithm])]
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
    flgo.init(os.path.join('task', task), algo, optimal_option, model=model, Logger=FullLogger).run()
