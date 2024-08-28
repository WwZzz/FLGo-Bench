import argparse
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import flgo
import flgo.experiment.device_scheduler as ds
import torch.multiprocessing
import yaml
import importlib
import logger

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='name of task', type=str, default='')
    parser.add_argument('--algorithm', help='name of method', type=str, nargs='*', default='fedavg')
    parser.add_argument('--model', help = 'model name', type=str, default='')
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int, default=[0])
    parser.add_argument('--seeds', nargs='+', help='seeds', type=int, default=[2,4388,15,333,967])
    parser.add_argument('--config', type=str, help='configuration of hypara', default='', nargs='*')
    parser.add_argument('--put_interval', help='interval (s) to put command into devices', type=int, default=10)
    parser.add_argument('--max_pdev', help='interval (s) to put command into devices', type=int, default=7)
    parser.add_argument('--available_interval', help='check availability of devices every x seconds', type=int, default=10)
    parser.add_argument('--mmap', help='mmap',  action="store_true", default=False)
    parser.add_argument('--load_mode', help = 'load_mode', type=str, default='')
    parser.add_argument('--seq', help='run sequencially',  action="store_true", default=False)
    parser.add_argument('--num_client_parallel', help = 'number of parallel processing', type=int, default=0)
    parser.add_argument('--test_parallel', help='test parallel',  action="store_true", default=False)
    parser.add_argument('--logger', help='test parallel', type=str, default='FullLogger')
    return parser.parse_known_args()

args = read_args()[0]
task = args.task
seeds = args.seeds
gpus = args.gpu
assert len(args.config)==len(args.algorithm)
assert len(args.config)>0
optimal_options = []
for config in args.config:
    with open(config, 'r') as inf:
        option = yaml.load(inf, Loader=yaml.FullLoader)
    option['load_mode'] = args.load_mode
    if 'early_stop' in option.keys(): option.pop('early_stop')
    optimal_options.append(option)

def fedrun(task, algos=[], optimal_options=[], seeds=[0], Logger=None, model=None, put_interval=10, available_interval=10, max_processes_per_device=10, mmap=False, seq=False):
    assert len(algos)==len(optimal_options)
    runner_dict = []
    asc = ds.AutoScheduler(args.gpu, put_interval=put_interval, available_interval=available_interval, max_processes_per_device=max_processes_per_device)
    if not seq:
        for algo, optimal_option in zip(algos, optimal_options):
            for seed in seeds:
                opi = optimal_option.copy()
                opi.update({'seed': seed})
                runner_dict.append({'task': task, 'algorithm': algo, 'option': opi, 'model':model, 'Logger':Logger})
        res = flgo.multi_init_and_run(runner_dict, scheduler=asc, mmap=mmap)
    else:
        for algo, optimal_option in zip(algos, optimal_options):
            options = []
            for seed in seeds:
                opi = optimal_option.copy()
                opi.update({'seed': seed})
                options.append(opi)
            res = flgo.run_in_sequencial(task, algo, options, model, Logger=Logger, mmap=mmap)
    return res

if __name__=='__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    algos = []
    acce = False
    for algoname in args.algorithm:
        modules = [".".join(["algorithm", algoname]), ".".join(["develop", algoname]),
                   ".".join(["flgo", "algorithm", algoname])]
        algo = None
        if args.num_client_parallel > 0:
            try:
                algo = importlib.import_module(".".join(["algorithm", "accelerate", algoname]))
                acce = True
            except:
                algo = None
                warnings.warn(f"There is no acceleration support for {algoname}")
        if algo is None:
            for m in modules:
                try:
                    algo = importlib.import_module(m)
                    break
                except ModuleNotFoundError:
                    continue
        if algo is None: raise ModuleNotFoundError("{} was not found".format(algo))
        algos.append(algo)
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
        for optimal_option in optimal_options:
            optimal_option['num_parallels'] =args.num_client_parallel
            optimal_option['parallel_type'] = 'else'
    if args.test_parallel:
        for optimal_option in optimal_options:
            optimal_option['test_parallel'] = True
    Logger = getattr(logger, args.logger)
    fedrun(os.path.join('task', task), algos, optimal_options=optimal_options, seeds=seeds, Logger=Logger, model=model, put_interval=args.put_interval, available_interval=args.available_interval, max_processes_per_device=args.max_pdev, mmap=args.mmap, seq=args.seq)