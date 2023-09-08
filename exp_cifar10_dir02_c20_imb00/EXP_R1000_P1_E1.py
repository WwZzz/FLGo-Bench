import argparse
import flgo
from flgo.experiment.logger import BasicLogger
import flgo.benchmark.partition as fbp
import flgo.experiment.device_scheduler as ds
import numpy as np
import torch.multiprocessing
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help='name of method', type=str, default='fedavg')
    parser.add_argument('--tune', help='whether to tune', action="store_true", default=False)
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int, default=[])
    parser.add_argument('--seeds', nargs='+', help='seeds', type=int, default=[1,15,47,967])
    return parser.parse_known_args()

args = read_args()[0]
seeds = args.seeds
gpus = args.gpu

import flgo.benchmark.cifar100_classification as cifar
task = './CIFAR10_Dirichlet1.0_Clients20_IMB0.0'
flgo.gen_task_by_(cifar, fbp.DirichletPartitioner(alpha=1.0, num_clients=20), task)

local_training_option = {
    'learning_rate':[0.01, 0.05, 0.1,],
    'batch_size':[10, 32, 50],
    'weight_decay': [1e-2, 1e-3],
    'lr_scheduler': 0,
    'learning_rate_decay': 0.998,
}
server_option = {
    'num_rounds': 1000,
    'num_epochs': [1,5],
    'sample_option': 'full',
    'proportion': 1.0,
    'early_stop':200,
}

data_option = {
    'local_test': True,
}
other_option = {
    'gpu':gpus,
    'no_log_console': True,
    'log_file': True,
}

common_options = [local_training_option, server_option, data_option, other_option]
tune_option = {}
for op in common_options: tune_option.update(op)

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
        self.turn_early_stop_direction()

    def log_once(self, *args, **kwargs):
        val_metrics = self.coordinator.global_test(flag='val')
        local_data_vols = [c.datavol for c in self.participants]
        total_data_vol = sum(local_data_vols)
        for met_name, met_val in val_metrics.items():
            self.output['val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        self.show_current_output()

def fedrun(task, algo, tune_option={}, optimal_option={}, seeds=[0], tune=True, Logger=None):
    if tune:
        return flgo.tune(task, algo, tune_option, Logger=Logger)
    else:
        runner_dict = []
        asc = ds.AutoScheduler(optimal_option['gpu'])
        for seed in seeds:
            opi = optimal_option.copy()
            opi.update({'seed': seed})
            runner_dict.append({'task': task, 'algorithm': algo, 'option': opi, 'Logger':Logger})
        res = flgo.multi_init_and_run(runner_dict, scheduler=asc)
        return res

if __name__=='__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    if args.method=='fedavg':
        import flgo.algorithm.fedavg as algo
        """
        """
        optimal_option = {'gpu':gpus,'learning_rate':0.1, 'batch_size':16, 'weight_decay':0.001, "lr_scheduler":0, "learning_rate_decay":0.998, 'num_rounds':1000, "num_epochs":1, "sample_option":'full', "proportion":1.0, "local_test":True, "no_log_console":True, "log_file":True }
        fedrun(task, algo, tune_option, optimal_option=optimal_option, seeds=seeds, tune=args.tune, Logger=TuneLogger if args.tune else FullLogger)
    elif args.method=='fedprox':
        import flgo.algorithm.fedprox as algo
        """
        """
        tune_option.update({'mu':[0.001, 0.01, 0.1]})
        optimal_option = {}
        fedrun(task, algo, tune_option, optimal_option=optimal_option, seeds=seeds, tune=args.tune, Logger=TuneLogger if args.tune else FullLogger)
    elif args.method=='feddyn':
        import flgo.algorithm.feddyn as algo
        """
        """
        tune_option.update({'mu':[0.0001, 0.001, 0.01], 'alpha':[0.001, 0.01, 0.1]})
        optimal_option = {}
        fedrun(task, algo, tune_option, optimal_option=optimal_option, seeds=seeds, tune=args.tune, Logger=TuneLogger if args.tune else FullLogger)




