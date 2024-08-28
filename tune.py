import sys
import os.path
import warnings
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib
import argparse
import os.path
import flgo
import torch.multiprocessing as mlp
import yaml
import logger

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', help='algorithm name', type=str, default=['fedavg'], nargs='*')
    parser.add_argument('--task', help='task name', type=str, default='cifar10_iid_c100')
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int, default=[0])
    parser.add_argument('--config', help='congiguration', type=str, default=[], nargs='*')
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
    parser.add_argument('--test_parallel', help='test parallel',  action="store_true", default=False)
    parser.add_argument('--logger', help='test parallel', type=str, default='TuneLogger')
    return parser.parse_known_args()

if __name__=='__main__':
    mlp.set_sharing_strategy("file_system")
    mlp.set_start_method("spawn", force=True)
    args = read_args()[0]
    config_file = args.config
    mmap = args.mmap
    configs = []
    if len(config_file)>0:
        assert len(args.algorithm)==len(config_file)
        for config_i in config_file:
            with open(config_i, 'r') as inf:
                config_data_i = yaml.load(inf, yaml.Loader)
            configs.append(config_data_i)
    else:
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task', args.task, 'config.yml')
        if os.path.exists(config_file):
            with open(config_file, 'r') as inf:
                config = yaml.load(inf, yaml.Loader)
        else:
            config = {}
        algo_para_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'algo_para.yml')
        with open(algo_para_config, 'r') as inf:
            algo_para_config = yaml.load(inf, yaml.Loader)
        for algo in args.algorithm:
            config_tmp = config.copy()
            algo_para = algo_para_config.get(algo, None)
            if algo_para is not None: config_tmp['algo_para'] = algo_para
            configs.append(config_tmp)
    for config in configs: config['load_mode'] = args.load_mode
    import flgo.experiment.device_scheduler as fed
    scheduler = None if args.gpu is None else fed.AutoScheduler(args.gpu, put_interval=args.put_interval, available_interval=args.available_interval, mean_memory_occupated=args.memory, dynamic_memory_occupated=not args.no_dynmem, max_processes_per_device=args.max_pdev)
    #
    # paras = config
    # paras['load_mode'] = args.load_mode
    algos = []
    for algo_id, algo_name in enumerate(args.algorithm):
        algo = None
        acce = False
        modules = [".".join(["algorithm", algo_name]), ".".join(["develop", algo_name]),".".join(["flgo", "algorithm", algo_name])]
        if args.num_client_parallel>0:
            try:
                algo = importlib.import_module(".".join(["algorithm", "accelerate", algo_name]))
                acce = True
            except:
                algo = None
                warnings.warn(f"There is no acceleration support for {algo_name}")
        if algo is None:
            for m in modules:
                try:
                    algo = importlib.import_module(m)
                    break
                except ModuleNotFoundError:
                    continue
        if algo is None: raise ModuleNotFoundError("{} was not found".format(algo))
        algos.append(algo)
        if acce and args.num_client_parallel>0:
            configs[algo_id]['num_parallels'] = args.num_client_parallel
            configs[algo_id]['parallel_type'] = 'else'
    task = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task', args.task)
    models = []
    if args.model != '':
        for algo in algos:
            model = None
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
            models.append(model)
    if args.test_parallel:
        for config in configs:
            config['test_parallel'] = True
    Logger = getattr(logger, args.logger)
    if args.seq:
        for algo, para, model in zip(algos, configs, models):
            res = flgo.tune_sequencially(task, algo, paras, model=model, Logger=Logger, mmap=mmap, target_path=os.path.join(os.path.dirname(__file__), 'config'))
    else:
        if len(algos)==1:
            algo = algos[0]
            paras = configs[0]
            model = models[0] if len(models)>0 else None
            res = flgo.tune(task, algo, paras, model=model, Logger=Logger, scheduler=scheduler, mmap=mmap, target_path=os.path.join(os.path.dirname(__file__), 'config'))
        else:
            task_dict = {'task':task, 'algorithm':algos, 'option': configs, 'Logger':Logger, 'model':models}
            flgo.multi_tune(task_dict, scheduler=scheduler, target_path=os.path.join(os.path.dirname(__file__), 'config'))