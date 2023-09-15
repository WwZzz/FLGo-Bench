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
    parser.add_argument('--method', help='name of method', type=str, default='fedavg')
    parser.add_argument('--tune', help='whether to tune', action="store_true", default=False)
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int, default=[])
    parser.add_argument('--seeds', nargs='+', help='seeds', type=int, default=[1,15,47,967])
    parser.add_argument('--config', type=str, help='configuration of hypara', default='')
    return parser.parse_known_args()

args = read_args()[0]
task = args.task
seeds = args.seeds
gpus = args.gpu
config = args.config
with open(config, 'r') as inf:
    option = yaml.load(inf, Loader=yaml.FullLoader)
option['gpu'] = gpus
tune_option = option
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

class TuneLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self._es_key = 'val_accuracy'
        self.turn_es_direction()

    def log_once(self, *args, **kwargs):
        val_metrics = self.coordinator.global_test(flag='val')
        local_data_vols = [c.datavol for c in self.participants]
        total_data_vol = sum(local_data_vols)
        for met_name, met_val in val_metrics.items():
            self.output['val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        self.show_current_output()

class LimitedAutoScheduler(ds.AutoScheduler):
    def __init__(self, devices: list, put_interval=5, mean_memory_occupated=1000, available_interval=5,
                 dynamic_memory_occupated=True, dynamic_condition='mean', left_ratio=0.6):
        super(LimitedAutoScheduler, self).__init__(devices = devices, put_interval=put_interval, mean_memory_occupated=mean_memory_occupated, available_interval=available_interval,
                 dynamic_memory_occupated=dynamic_memory_occupated, dynamic_condition=dynamic_condition)
        self.left_ratio = min(max(left_ratio,0.0),1.0)

    def check_available(self, dev):
        if dev=='-1':return True
        crt_time = time.time()
        crt_free_memory = self.dev_state[dev]['free_memory']
        target_memory = self.mean_memory_occupated
        total_memory = self.dev_state[dev]['total_memory']
        crt_avl = (crt_free_memory>target_memory) and (crt_free_memory>=total_memory*self.left_ratio)
        if crt_avl:
            if self.dev_state[dev]['avl']:
                if crt_time - self.dev_state[dev]['time']>=self.available_interval:
                    if self.dev_state[dev]['time_put'] is None or crt_time-self.dev_state[dev]['time_put']>=self.put_interval:
                        self.dev_state[dev]['time_put'] = crt_time
                        return True
        if crt_avl!=self.dev_state[dev]['avl']:
            self.dev_state[dev]['avl'] = True
            self.dev_state[dev]['time'] = crt_time
        return False

def fedrun(task, algo, tune_option={}, optimal_option={}, seeds=[0], tune=True, Logger=None):
    if tune:
        return flgo.tune(task, algo, tune_option, Logger=Logger)
    else:
        runner_dict = []
        asc = LimitedAutoScheduler(optimal_option['gpu'], put_interval=10, available_interval=10, left_ratio=0.7)
        for seed in seeds:
            opi = optimal_option.copy()
            opi.update({'seed': seed})
            runner_dict.append({'task': task, 'algorithm': algo, 'option': opi, 'Logger':Logger})
        res = flgo.multi_init_and_run(runner_dict, scheduler=asc)
        return res

if __name__=='__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    algo = None
    while algo is None:
        try:
            algo = importlib.import_module(args.method)
            break
        except:
            algo = flgo.download_resource('.', args.method, 'algorithm')
    optimal_option = fedrun(task, algo, tune_option, optimal_option=optimal_option, seeds=seeds, tune=args.tune, Logger=TuneLogger if args.tune else FullLogger)
    if args.tune:
        with open(os.path.join(os.path.dirname(args.config), '_'.join(['op', 'config', args.method, task])+'.yml'), 'w') as outf:
            yaml.dump(optimal_option, outf)