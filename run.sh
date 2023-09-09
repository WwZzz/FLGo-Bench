############################Task cifar10_dir1_20
######################## P=1.0
# fedavg
python run.py --task cifar10_dir1_c20 --method fedavg --config ./EXP_R1000_P1._V.2/config_fedavg.yml --tune --gpu 0 1 2 3 4 5 6 7  > fedavg_r1000_p1_v2.txt
python run.py --task cifar10_dir1_c20 --method fedavg --config ./EXP_R1000_P1._V.2/op_config_fedavg_cifar10_dir1_c20.yml --gpu 0 1 2 3 4 > fedavg_r1000_p1_v2.txt
# fedprox
python run.py --task cifar10_dir1_c20 --method fedprox --config ./EXP_R1000_P1._V.2/config_fedprox.yml --tune --gpu 3 4 5 6 7 > fedprox_r1000_p1_v2.txt
python run.py --task cifar10_dir1_c20 --method fedprox --config ./EXP_R1000_P1._V.2/op_config_fedprox_cifar10_dir1_c20.yml --gpu 4 5 6 7 > fedprox_r1000_p1_v2.txt

####################### P=0.1
