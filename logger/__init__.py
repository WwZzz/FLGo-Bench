from flgo.experiment.logger import BasicLogger
import numpy as np
class TuneLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        if self.server.val_data is not None and len(self.server.val_data) > 0:
            self.set_es_key("val_accuracy")
        else:
            self.set_es_key("local_val_accuracy")
        self.set_es_direction(1)
    def log_once(self, *args, **kwargs):
        if self.server.val_data is not None and len(self.server.val_data)>0:
            sval = self.server.test(self.server.model, 'val')
            for met_name in sval.keys():
                self.output['val_'+met_name].append(sval[met_name])
        else:
            cvals = [c.test(self.server.model, 'val') for c in self.clients]
            cdatavols = np.array([len(c.val_data) for c in self.clients])
            cdatavols = cdatavols/cdatavols.sum()
            cval_dict = {}
            if len(cvals) > 0:
                for met_name in cvals[0].keys():
                    if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                    for cid in range(len(cvals)):
                        cval_dict[met_name].append(cvals[cid][met_name])
                    self.output['local_val_' + met_name].append(float((np.array(cval_dict[met_name])*cdatavols).sum()))
                    self.output['min_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).min()))
                    self.output['max_local_val_' + met_name].append(float(np.array(cval_dict[met_name]).max()))
        self.show_current_output()

class FullLogger(BasicLogger):
    def log_once(self, *args, **kwargs):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        val_metric = self.server.test(flag='val')
        for met_name, met_val in val_metric.items():
            self.output['val_' + met_name].append(met_val)
        # calculate weighted averaging of metrics on training datasets across participants
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        train_metrics = self.server.global_test(flag='train')
        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name + '_dist'].append(met_val)
            self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        local_val_metrics = self.server.global_test(flag='val')
        for met_name, met_val in local_val_metrics.items():
            self.output['local_val_'+met_name+'_dist'].append(met_val)
            self.output['local_val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_val_' + met_name].append(np.mean(met_val))
            self.output['std_local_val_' + met_name].append(np.std(met_val))
        local_test_metrics = self.server.global_test(flag='test')
        for met_name, met_val in local_test_metrics.items():
            self.output['local_test_'+met_name+'_dist'].append(met_val)
            self.output['local_test_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_test_' + met_name].append(np.mean(met_val))
            self.output['std_local_test_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()

class SimpleLogger(BasicLogger):
    def log_once(self, *args, **kwargs):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        val_metric = self.server.test(flag='val')
        for met_name, met_val in val_metric.items():
            self.output['val_' + met_name].append(met_val)
        # calculate weighted averaging of metrics on training datasets across participants
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        local_val_metrics = self.server.global_test(flag='val')
        for met_name, met_val in local_val_metrics.items():
            self.output['local_val_'+met_name+'_dist'].append(met_val)
            self.output['local_val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_val_' + met_name].append(np.mean(met_val))
            self.output['std_local_val_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()