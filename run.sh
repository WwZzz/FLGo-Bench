nohup python run.py --task cifar10_iid_c100 --algorithm fedavg --gpu 0 1 2 3 --config ./config/tmp.yml &
nohup python run.py --task cifar10_dir0.1_c100 --algorithm fedavg --gpu 0 1 2 3 --config ./config/tmp.yml &
nohup python run.py --task cifar10_dir1.0_c100 --algorithm fedavg --gpu 0 1 2 3 --config ./config/tmp.yml &
nohup python run.py --task cifar10_dir2.0_c100 --algorithm fedavg --gpu 0 1 2 3 --config ./config/tmp.yml &
nohup python run.py --task cifar10_dir5.0_c100 --algorithm fedavg --gpu 0 1 2 3 --config ./config/tmp.yml &

nohup python run.py --task mnist_iid_c100 --algorithm fedavg --gpu 0 1 2 3 --config ./config/tmp.yml &
nohup python run.py --task mnist_dir0.1_c100 --algorithm fedavg --gpu 0 1 2 3 --config ./config/tmp.yml &
nohup python run.py --task mnist_dir1.0_c100 --algorithm fedavg --gpu 1 2 3 0 --config ./config/tmp.yml &
nohup python run.py --task mnist_dir2.0_c100 --algorithm fedavg --gpu 2 3 0 1 --config ./config/tmp.yml &
nohup python run.py --task mnist_dir5.0_c100 --algorithm fedavg --gpu 3 0 1 2 --config ./config/tmp.yml &

nohup python run.py --task agnews_dir1.0_c100 --algorithm fedavg --gpu 1 2 3 0 --config ./config/tmp.yml --max_pdev 1 --put_interval 30 --available_interval 30 &

nohup python run.py --task cifar10_iid_c100 --algorithm fedprox --gpu 1 2 3 0 --config ./config/run_prox.yml &
nohup python run.py --task mnist_iid_c100 --algorithm fedprox --gpu 0 1 2 3 --config ./config/prox_iid.yml &
nohup python run.py --task mnist_dir0.1_c100 --algorithm fedprox --gpu 0 1 2 3 --config ./config/prox_dir01.yml &
nohup python run.py --task mnist_dir1.0_c100 --algorithm fedprox --gpu 1 2 3 0 --config ./config/prox_dir10.yml &
nohup python run.py --task mnist_dir2.0_c100 --algorithm fedprox --gpu 2 3 0 1 --config ./config/prox_dir20.yml &
nohup python run.py --task mnist_dir5.0_c100 --algorithm fedprox --gpu 3 0 1 2 --config ./config/prox_dir50.yml &

nohup python run.py --task mnist_iid_c100 --algorithm feddyn --gpu 3 0 1 2 --config ./config/dyn_iid.yml &
nohup python run.py --task mnist_dir0.1_c100 --algorithm feddyn --gpu 0 1 2 3 --config ./config/dyn_dir.yml &
nohup python run.py --task mnist_dir1.0_c100 --algorithm feddyn --gpu 2 3 0 1 --config ./config/dyn_dir.yml &

nohup python run.py --task mnist_iid_c100 --algorithm moon --gpu 0 1 2 3 --config ./config/moon_iid.yml &
nohup python run.py --task mnist_dir0.1_c100 --algorithm moon --gpu 0 1 2 3 --config ./config/moon_d0.yml &
nohup python run.py --task mnist_dir1.0_c100 --algorithm moon --gpu 1 2 3 0 --config ./config/moon_d1.yml &
nohup python run.py --task mnist_dir2.0_c100 --algorithm moon --gpu 2 3 0 1 --config ./config/moon_d2.yml &
nohup python run.py --task mnist_dir5.0_c100 --algorithm moon --gpu 3 0 1 2 --config ./config/moon_d5.yml &
