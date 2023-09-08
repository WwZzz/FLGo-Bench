python run.py --task CIFAR100_Dirichlet1.0_Clients20_IMB0.0 --method fedavg --config config_fedavg_R1000_P1.yml --tune --gpu 0 1 2 3 4 5 6 7  > fedavg_r1000_p1.txt
python run.py --task CIFAR100_Dirichlet1.0_Clients20_IMB0.0 --method fedprox --config config_fedprox_R1000_P1.yml --tune --gpu 0 1 2 3 4 5 6 7 > fedprox_r1000_p1.txt
