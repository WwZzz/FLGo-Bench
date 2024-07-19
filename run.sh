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