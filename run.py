import argparse
import warnings

import flgo
from flgo.experiment.logger import BasicLogger
import flgo.experiment.device_scheduler as ds
import numpy as np
import torch.multiprocessing
import yaml
import importlib
import os
import flgo.utils.fflow as fuf
import time

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='name of task', type=str, default='')
    parser.add_argument('--algorithm', help='name of method', type=str, default='fedavg')
    parser.add_argument('--model', help = 'model name', type=str, default='')
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int, default=[0])
    parser.add_argument('--seeds', nargs='+', help='seeds', type=int, default=[2,4388,15,333,967])
    parser.add_argument('--config', type=str, help='configuration of hypara', default='')
    parser.add_argument('--put_interval', help='interval (s) to put command into devices', type=int, default=10)
    parser.add_argument('--max_pdev', help='interval (s) to put command into devices', type=int, default=7)
    parser.add_argument('--available_interval', help='check availability of devices every x seconds', type=int, default=10)
    parser.add_argument('--mmap', help='mmap',  action="store_true", default=False)
    parser.add_argument('--load_mode', help = 'load_mode', type=str, default='')
    parser.add_argument('--seq', help='run sequencially',  action="store_true", default=False)
    parser.add_argument('--num_client_parallel', help = 'number of parallel processing', type=int, default=0)
    return parser.parse_known_args()

args = read_args()[0]
task = args.task
seeds = args.seeds
gpus = args.gpu
config = args.config
with open(config, 'r') as inf:
    option = yaml.load(inf, Loader=yaml.FullLoader)
option['gpu'] = gpus
optimal_option = option
optimal_option['load_mode'] = args.load_mode
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
        train_metrics = self.server.global_test(flag='train')
        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name + '_dist'].append(met_val)
            self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        local_val_metrics = self.server.global_test(flag='val')
        for met_name, met_val in local_val_metrics.items():
            self.output['local_val_'+met_name+'_dist'].append(met_val)
            self.output['local_val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_val_' + met_name].append(np.mean(met_val))
            self.output['std_local_val_' + met_name].append(np.std(met_val))
        local_test_metrics = self.server.global_test(flag='test')
        for met_name, met_val in local_test_metrics.items():
            self.output['local_test_'+met_name+'_dist'].append(met_val)
            self.output['local_test_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_test_' + met_name].append(np.mean(met_val))
            self.output['std_local_test_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()

def fedrun(task, algo, optimal_option={}, seeds=[0], Logger=None, model=None, put_interval=10, available_interval=10, max_processes_per_device=10, mmap=False, seq=False):
    runner_dict = []
    asc = ds.AutoScheduler(optimal_option['gpu'], put_interval=put_interval, available_interval=available_interval, max_processes_per_device=max_processes_per_device)
    for seed in seeds:
        opi = optimal_option.copy()
        opi.update({'seed': seed})
        runner_dict.append({'task': task, 'algorithm': algo, 'option': opi, 'model':model, 'Logger':Logger})
    if not seq:
        res = flgo.multi_init_and_run(runner_dict, scheduler=asc, mmap=mmap)
    else:
        res = flgo.run_in_sequencial(task, algo, [r['option'] for r in runner_dict], model, Logger=Logger, mmap=mmap)
    return res

if __name__=='__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
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
    if acce and args.num_client_parallel>0:
        optimal_option['num_parallels'] =args.num_client_parallel
        optimal_option['parallel_type'] = 'obj'
    fedrun(os.path.join('task', task), algo, optimal_option=optimal_option, seeds=seeds, Logger=FullLogger, model=model, put_interval=args.put_interval, available_interval=args.available_interval, max_processes_per_device=args.max_pdev, mmap=args.mmap, seq=args.seq)