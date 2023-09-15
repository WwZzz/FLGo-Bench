######################## R1000
# P=0.1
## fedavg
#python run.py --task cifar100_dir.1_c20 --method fedavg --config ./EXP_R2000_P.1_V.2/config_fedavg.yml --tune --gpu 0 1 2 3
#
####################### P=0.2
# fedavg
nohup python run.py --task cifar100_dir.1_c20 --method fedavg --config ./EXP_R1000_P.2_V.2/config_fedavg.yml --tune --gpu 0 1 2 3 &

######################## P=1.0
## fedavg
#python run.py --task cifar100_dir.1_c20 --method fedavg --config ./EXP_R2000_P1._V.2/config_fedavg.yml --tune --gpu 0 1 2 3

######################## R2000 P=0.1
# fedavg
nohup python run.py --task cifar100_dir.1_c20 --method fedavg --config ./EXP_R2000_P.1_V.2/config_fedavg.yml --tune --gpu 0 1 2 3 &

####################### P=0.2
# fedavg
nohup python run.py --task cifar100_dir.1_c20 --method fedavg --config ./EXP_R2000_P.2_V.2/config_fedavg.yml --tune --gpu 0 1 2 3 &

####################### P=1.0
# fedavg
nohup python run.py --task cifar100_dir.1_c20 --method fedavg --config ./EXP_R2000_P1._V.2/config_fedavg.yml --tune --gpu 0 1 2 3 &