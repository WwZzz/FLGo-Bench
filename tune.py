import sys
import os.path
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
import argparse
import os.path
import torch.utils.data as tud
import flgo
from flgo.experiment.logger import BasicLogger
import torch.multiprocessing as mlp
import yaml
import collections
import numpy as np

class TuneLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        if self.server.val_data is not None and len(self.server.val_data) > 0:
            self.set_es_key("val_accuracy")
        else:
            self.set_es_key("local_val_accuracy")
        self.set_es_direction(1)
    def log_once(self, *args, **kwargs):
        if self.server.val_data is not None and len(self.server.val_data)>0:
            sval = self.server.test(self.server.model, 'val')
            for met_name in sval.keys():
                self.output['val_'+met_name].append(sval[met_name])
        else:
            cvals = [c.test(self.server.model, 'val') for c in self.clients]
            cdatavols = np.array([len(c.val_data) for c in self.clients])
            cdatavols = cdatavols/cdatavols.sum()
            cval_dict = {}
            if len(cvals) > 0:
                for met_name in cvals[0].keys():
                    if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                    for cid in range(len(cvals)):
                        cval_dict[met_name].append(cvals[cid][met_name])
                    self.output['local_val_' + met_name].append(float((np.array(cval_dict[met_name])*cdatavols).sum()))
                    self.output['min_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).min()))
                    self.output['max_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).max()))
        self.show_current_output()


def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', help='algorithm name', type=str, default='fedavg')
    parser.add_argument('--task', help='task name', type=str, default='cifar10_iid_c100')
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int, default=[0])
    parser.add_argument('--config', help='congiguration', type=str, default='')
    parser.add_argument('--model', help = 'model name', type=str, default='')
    parser.add_argument('--put_interval', help='interval (s) to put command into devices', type=int, default=5)
    parser.add_argument('--max_pdev', help='interval (s) to put command into devices', type=int, default=7)
    parser.add_argument('--available_interval', help='check availability of devices every x seconds', type=int, default=5)
    parser.add_argument('--memory', help='mean memory occupation', type=float, default=1000)
    parser.add_argument('--no_dynmem', help='no_dynmem',  action="store_true", default=False)
    parser.add_argument('--mmap', help='mmap',  action="store_true", default=False)
    parser.add_argument('--load_mode', help = 'load_mode', type=str, default='')
    parser.add_argument('--seq', help='tune sequencially',  action="store_true", default=False)
    parser.add_argument('--num_client_parallel', help='number of parallel processing',   type=int, default=0)
    try:
        option = vars(parser.parse_known_args()[0])
    except IOError as msg:
        parser.error(str(msg))
    return option

if __name__=='__main__':
    mlp.set_sharing_strategy("file_system")
    mlp.set_start_method("spawn", force=True)
    option = read_option()
    config_file = option['config']
    mmap = option['mmap']
    if config_file=='': config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task', option['task'], 'config.yml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as inf:
            config = yaml.load(inf, yaml.Loader)
    else:
        config = {}
    paras = config
    paras['load_mode'] = option['load_mode']
    import flgo.experiment.device_scheduler as fed
    scheduler = None if option['gpu'] is None else fed.AutoScheduler(option['gpu'], put_interval=option['put_interval'], available_interval=option['available_interval'], mean_memory_occupated=option['memory'], dynamic_memory_occupated=not option['no_dynmem'], max_processes_per_device=option['max_pdev'])
    method = None
    acce = False
    modules = [".".join(["algorithm", option['algorithm']]), ".".join(["develop", option['algorithm']]),".".join(["flgo", "algorithm", option['algorithm']])]
    if option['num_client_parallel']>0:
        try:
            method = importlib.import_module(".".join(["algorithm", "accelerate", option['algorithm']]))
            acce = True
        except:
            method = None
            warnings.warn(f"There is no acceleration support for {option['algorithm']}")
    if method is None:
        for m in modules:
            try:
                method = importlib.import_module(m)
                break
            except ModuleNotFoundError:
                continue
    if method is None: raise ModuleNotFoundError("{} was not found".format(method))
    task = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task', option['task'])

    model  = None
    if option['model'] != '':
        try:
            model = getattr(method, option['model'])
        except:
            model = None
        if model is None:
            try:
                model = importlib.import_module(option['model'])
            except:
                print("using default model")
                model = None
    if acce and option['num_client_parallel']>0:
        paras['num_parallels'] = option['num_client_parallel']
        paras['parallel_type'] = 'obj'
    if option['seq']:
        res = flgo.tune_sequencially(task, method, paras, model=model, Logger=TuneLogger, mmap=mmap)
    else:
        res = flgo.tune(task, method, paras, model=model, Logger=TuneLogger, scheduler=scheduler, mmap=mmap)