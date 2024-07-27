import argparse
import os.path
import numpy as np
import yaml
import flgo.experiment.analyzer as fea

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='name of task', type=str, default='')
    parser.add_argument('--algorithm', help='name of method', type=str, default='fedavg')
    parser.add_argument('--config', type=str, help='configuration of hypara', default='')
    return parser.parse_known_args()

args = read_args()[0]
task =args.task

if args.config!='' and os.path.exists(args.config):
    with open(args.config, 'r') as inf:
        option = yaml.load(inf, Loader=yaml.FullLoader)
else: option = {}
algorithm = args.algorithm
config = args.config

records = fea.load_records(os.path.join('task', task), algorithm, option)
print(f"Number of Records: {len(records)}")
painter = fea.Painter(records)
painter.create_figure(fea.Curve, {'args':{'x':'communication_round', 'y':'val_accuracy'}})
tb = fea.Table(records)

def get_column(tb, name):
    idx = tb.tb.field_names.index(name)
    col_values = [r[idx] for r in tb.tb.rows]
    return col_values
def max_log(x, op={}):
    res = x.log.get(op['x'], None)
    return max(res) if res is not None else -np.inf

def max_local_val_acc(x, op={}):
    return max_log(x, {'x': 'local_val_accuracy'})

def max_global_val_acc(x, op={}):
    return max_log(x, {'x': 'val_accuracy'})

def lr(x, op={}):
    return x.option['learning_rate']

def optimal_round_by_val(x, op={}):
    return np.argmax(x.log['val_accuracy'])

tb.add_column(max_local_val_acc)
tb.add_column(max_global_val_acc)
tb.add_column(lr)
tb.add_column(optimal_round_by_val)
sort_key =  max_local_val_acc.__name__
gv = get_column(tb, max_global_val_acc.__name__)
if len(gv)>0 and gv[0] is not None:
    sort_key = max_global_val_acc.__name__
tb.tb.sortby = sort_key
tb.print()
# selector = fea.Selector({'task': task, 'header':[method], 'filter':config2filter(option)})
# records = selector.records[task]
# records = [r for r in records if 'Tune' not in r.name]
# selector.all_records = records
# grouped_records, groups = selector.group_records(key=['seed', 'gpu'])
# painter = fea.Painter(grouped_records)
# # create figs
# fig1 = {'args':{'x':'communication_round', 'y':'test_accuracy'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'test_accuracy','title': "{}_on_{}".format(method, task)}}
# fig2 = {'args':{'x':'communication_round', 'y':'test_loss'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'test_loss','title': "{}_on_{}".format(method, task)}}
# fig3 = {'args':{'x':'communication_round', 'y':'mean_local_test_accuracy'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'mean_local_test_accuracy','title': "{}_on_{}".format(method, task)}}
# fig4 = {'args':{'x':'communication_round', 'y':'mean_local_test_loss'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'mean_local_test_loss','title': "{}_on_{}".format(method, task)}}
# fig5 = {'args':{'x':'communication_round', 'y':'std_local_test_accuracy'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'std_local_test_accuracy','title': "{}_on_{}".format(method, task)}}
# fig6 = {'args':{'x':'communication_round', 'y':'std_local_test_loss'}, 'fig_option':{'xlabel':'communication_round', 'ylabel':'std_local_test_loss','title': "{}_on_{}".format(method, task)}}
# figs = [fig1, fig2, fig3, fig4, fig5, fig6]
# for fig in figs:
#     painter.create_figure('GroupCurve', fig)
# tabulor = fea.Table(grouped_records)
# tabulor.add_column('group_optimal_value', {'x':'test_accuracy', 'flag':'max', 'name':'Acc.'})
# tabulor.add_column('group_optimal_value', {'x':'mean_local_test_accuracy', 'flag':'max', 'name':'Local Acc.'})
# tabulor.add_column('group_optimal_value', {'x':'test_loss', 'flag':'min','name':'Loss'})
# tabulor.add_column('group_optimal_value', {'x':'mean_local_test_loss', 'flag':'min', 'name':'Local Loss'})
# tabulor.add_column('group_optimal_x_by_y', {'x':'std_local_test_accuracy', 'y':'test_accuracy', 'flag':'max', 'name':'Std. Local Acc.'})
# tabulor.add_column('group_optimal_x_by_y', {'x':'std_local_test_loss', 'y':'test_accuracy', 'flag':'max', 'name':'Std. Local Loss'})
# tabulor.add_column('group_optimal_x_by_y', {'x':'communication_round', 'y':'test_accuracy', 'flag':'max', 'name':'Optimal Round'})
# tabulor.print()