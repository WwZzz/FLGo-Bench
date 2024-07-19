python tune.py --task cifar10_iid_c100 --gpu 0 1 2 3 --method fedavg
python tune.py --task cifar10_dir5.0_c100 --gpu 0 1 2 3 --method fedavg
python tune.py --task cifar10_dir2.0_c100 --gpu 0 1 2 3 --method fedavg
python tune.py --task cifar10_dir1.0_c100 --gpu 0 1 2 3 --method fedavg
python tune.py --task cifar10_dir0.1_c100 --gpu 0 1 2 3 --method fedavg

nohup python tune.py --task agnews_dir1.0_c100 --gpu 0 1 2 3 --method fedavg --max_pdev 1 &

nohup python tune.py --task cifar10_iid_c100 --gpu 0 1 2 3 --method fedprox --config ./config/prox.yml &
nohup python tune.py --task cifar10_dir1.0_c100 --gpu 3 1 2 0 --method fedprox --config ./config/prox.yml &


