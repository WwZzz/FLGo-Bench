python tune.py --task cifar10_iid_c100 --gpu 0 1 2 3 --method fedavg
python tune.py --task cifar10_dir5.0_c100 --gpu 0 1 2 3 --method fedavg
python tune.py --task cifar10_dir2.0_c100 --gpu 0 1 2 3 --method fedavg
python tune.py --task cifar10_dir1.0_c100 --gpu 0 1 2 3 --method fedavg
python tune.py --task cifar10_dir0.1_c100 --gpu 0 1 2 3 --method fedavg

nohup python tune.py --task cifar10_iid_c100 --gpu 0 1 2 3 --method fedprox --config ./config/prox.yml &

