nohup python tune.py --task cifar10_iid_c100 --gpu 0 1 2 3 --method fedavg
nohup python tune.py --task cifar10_dir5.0_c100 --gpu 0 1 2 3 --method fedavg
nohup python tune.py --task cifar10_dir2.0_c100 --gpu 0 1 2 3 --method fedavg
nohup python tune.py --task cifar10_dir1.0_c100 --gpu 0 1 2 3 --method fedavg
nohup python tune.py --task cifar10_dir0.1_c100 --gpu 0 1 2 3 --method fedavg
nohup python tune.py --task mnist_iid_c100 --gpu 0 1 2 3 --method fedavg
nohup python tune.py --task mnist_dir5.0_c100 --gpu 0 1 2 3 --method fedavg
nohup python tune.py --task mnist_dir2.0_c100 --gpu 0 1 2 3 --method fedavg
nohup python tune.py --task mnist_dir1.0_c100 --gpu 0 1 2 3 --method fedavg
nohup python tune.py --task mnist_dir0.1_c100 --gpu 0 1 2 3 --method fedavg

nohup python tune.py --task agnews_dir1.0_c100 --gpu 0 1 2 3 --method fedavg --max_pdev 1 &

nohup python tune.py --task mnist_iid_c100 --gpu 0 1 2 3 --method fedprox --config ./config/tune_prox.yml --put_interval 10 --available_interval 10 --max_pdev 10 &
nohup python tune.py --task mnist_dir0.1_c100 --gpu 0 1 2 3 --method fedprox --config ./config/tune_prox.yml --put_interval 10 --available_interval 10 --max_pdev 10 &
nohup python tune.py --task mnist_dir1.0_c100 --gpu 0 1 2 3 --method fedprox --config ./config/tune_prox.yml --put_interval 10 --available_interval 10 --max_pdev 10 &
nohup python tune.py --task mnist_dir2.0_c100 --gpu 0 1 2 3 --method fedprox --config ./config/tune_prox.yml --put_interval 10 --available_interval 10 --max_pdev 10 &
nohup python tune.py --task mnist_dir5.0_c100 --gpu 0 1 2 3 --method fedprox --config ./config/tune_prox.yml --put_interval 10 --available_interval 10 --max_pdev 10 &


nohup python tune.py --task cifar10_iid_c100 --gpu 0 1 2 3 --method fedprox --config ./config/prox.yml &
nohup python tune.py --task cifar10_dir1.0_c100 --gpu 3 1 2 0 --method fedprox --config ./config/prox.yml &

nohup python tune.py --task agnews_dir1.0_c100 --gpu 0 1 2 3 --method fedprox --config ./config/tune_prox.yml --put_interval 30 --available_interval 30 --max_pdev 1 &


nohup python tune.py --task mnist_iid_c100 --gpu 0 1 2 3 --method scaffold &
nohup python tune.py --task mnist_dir5.0_c100 --gpu 1 2 3 0 --method scaffold &
nohup python tune.py --task mnist_dir2.0_c100 --gpu 2 3 0 1 --method scaffold &
nohup python tune.py --task mnist_dir1.0_c100 --gpu 3 0 1 2 --method scaffold &
nohup python tune.py --task mnist_dir0.1_c100 --gpu 0 1 2 3 --method scaffold &


nohup python tune.py --task mnist_iid_c100 --gpu 0 1 2 3 --method feddyn --config ./config/tune_dyn.yml &
nohup python tune.py --task mnist_dir5.0_c100 --gpu 1 2 3 0 --method feddyn --config ./config/tune_dyn.yml &
nohup python tune.py --task mnist_dir2.0_c100 --gpu 2 3 0 1 --method feddyn --config ./config/tune_dyn.yml &
nohup python tune.py --task mnist_dir1.0_c100 --gpu 3 0 1 2 --method feddyn --config ./config/tune_dyn.yml &
nohup python tune.py --task mnist_dir0.1_c100 --gpu 0 1 2 3 --method feddyn --config ./config/tune_dyn.yml &

nohup python tune.py --task mnist_iid_c100 --gpu 0 1 2 3 --method moon --config ./config/tune_moon.yml &
nohup python tune.py --task mnist_dir5.0_c100 --gpu 1 2 3 0 --method moon --config ./config/tune_moon.yml &
nohup python tune.py --task mnist_dir2.0_c100 --gpu 2 3 0 1 --method moon --config ./config/tune_moon.yml &
nohup python tune.py --task mnist_dir1.0_c100 --gpu 3 0 1 2 --method moon --config ./config/tune_moon.yml &
nohup python tune.py --task mnist_dir0.1_c100 --gpu 0 1 2 3 --method moon --config ./config/tune_moon.yml &

nohup python tune.py --task agnews_dir1.0_c100 --gpu 0 1 2 3 --method scaffold &
