import argparse
import yaml

def config2filter(option):
    res = {}
    key_map = {
        'batch_size':'B',
        'learning_rate':'LR',
        'num_epochs':'E',
        'num_steps':'K',
        'proportion':'P',
        'num_rounds':'R',
    }
    for key in key_map:
        if key in option.keys():
            res.update({key_map[key]: option[key]})
    return res


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='name of task', type=str, default='')
    parser.add_argument('--method', help='name of method', type=str, default='fedavg')
    parser.add_argument('--config', type=str, help='configuration of hypara', default='')
    return parser.parse_known_args()

args = read_args()[0]
task =args.task

with open(args.config, 'r') as inf:
    option = yaml.load(inf, Loader=yaml.FullLoader)
method = args.method
config = args.config

import flgo.experiment.analyzer as fea
selector = fea.Selector({'task': task, 'header':[method], 'filter':config2filter(option)})
records = selector.records[task]
records = [r for r in records if 'Tune' not in r.name]
selector.all_records = records
grouped_records, groups = selector.group_records(key=['seed', 'gpu'])
painter = fea.Painter(grouped_records)
# create figs
fig1 = {'args':{'x':'communication_round', 'y':'test_accuracy'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'test_accuracy','title': "{}_on_{}".format(method, task)}}
fig2 = {'args':{'x':'communication_round', 'y':'test_loss'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'test_loss','title': "{}_on_{}".format(method, task)}}
fig3 = {'args':{'x':'communication_round', 'y':'mean_local_test_accuracy'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'mean_local_test_accuracy','title': "{}_on_{}".format(method, task)}}
fig4 = {'args':{'x':'communication_round', 'y':'mean_local_test_loss'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'mean_local_test_loss','title': "{}_on_{}".format(method, task)}}
fig5 = {'args':{'x':'communication_round', 'y':'std_local_test_accuracy'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'std_local_test_accuracy','title': "{}_on_{}".format(method, task)}}
fig6 = {'args':{'x':'communication_round', 'y':'std_local_test_loss'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'std_local_test_loss','title': "{}_on_{}".format(method, task)}}
figs = [fig1, fig2, fig3, fig4, fig5, fig6]
for fig in figs:
    painter.create_figure('GroupCurve', fig)
tabulor = fea.Table(grouped_records)
tabulor.add_column('group_optimal_value', {'x':'test_accuracy', 'flag':'max', 'name':'Acc.'})
tabulor.add_column('group_optimal_value', {'x':'mean_local_test_accuracy', 'flag':'max', 'name':'Local Acc.'})
tabulor.add_column('group_optimal_value', {'x':'test_loss', 'flag':'min','name':'Loss'})
tabulor.add_column('group_optimal_value', {'x':'mean_local_test_loss', 'flag':'min', 'name':'Local Loss'})
tabulor.add_column('group_optimal_x_by_y', {'x':'std_local_test_accuracy', 'y':'test_accuracy', 'flag':'max', 'name':'Std. Local Acc.'})
tabulor.add_column('group_optimal_x_by_y', {'x':'std_local_test_loss', 'y':'test_accuracy', 'flag':'max', 'name':'Std. Local Loss'})
tabulor.add_column('group_optimal_x_by_y', {'x':'communication_round', 'y':'test_accuracy', 'flag':'max', 'name':'Optimal Round'})
tabulor.print()