"""
This is a non-official implementation of MOON proposed in 'Model-Contrastive
Federated Learning (https://arxiv.org/abs/2103.16257)'. The official implementation is in https://github.com/QinbinLi/MOON.
********************************************Note***********************************************
The model used by this algorithm should be formulated by two submodules: encoder and head
"""
from flgo.algorithm.fedbase import BasicServer, BasicClient
import copy
import torch
import torch.nn.functional as F
import flgo.utils.fmodule as fmodule

def model_contrastive_loss(z, z_glob, z_prev, temperature=0.5):
    pos_sim = F.cosine_similarity(z, z_glob, dim=-1)
    logits = pos_sim.reshape(-1, 1)
    if z_prev is not None:
        neg_sim = F.cosine_similarity(z, z_prev, dim=-1)
        # neg_sim = self.cos(z, z_prev)
        logits = torch.cat((logits, neg_sim.reshape(-1, 1)), dim=1)
    logits /= temperature
    return F.cross_entropy(logits, torch.zeros(z.size(0)).long().to(logits.device))

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'mu': 0.1, 'tau':0.5})
        self.output_layer = "".join([f'[{m}]' if m.isdigit() else f'.{m}' for m in list(self.model.state_dict().keys())[-1].split('.')[:-1]])
        self.register_cache_var('local_model')
class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.local_model = None
        self.output_layer = self.server.output_layer

    @fmodule.with_multi_gpus
    def train(self, model):
        # init global model and local model
        global_model = copy.deepcopy(model)
        global_model.freeze_grad()
        if self.local_model is not None:
            self.local_model.to(model.get_device())

        feature_maps = []
        def hook(model, input, output):
            feature_maps.append(input)

        global_hook = eval('global_model{}'.format(self.output_layer)).register_forward_hook(hook)
        local_hook = eval("self.local_model{}".format(self.output_layer)).register_forward_hook(hook) if self.local_model is not None else None
        model_hook = eval('model{}'.format(self.output_layer)).register_forward_hook(hook)

        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            model.zero_grad()
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            _ = global_model(batch_data[0])
            if self.local_model is not None:
                _ = self.local_model(batch_data[0])
            else:
                feature_maps.append((None,))
            z_prev = feature_maps.pop()[0]
            z_glob = feature_maps.pop()[0]
            z = feature_maps.pop()[0]
            loss_con = model_contrastive_loss(z, z_glob, z_prev, self.tau)
            loss = loss + self.mu * loss_con
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        # remove hook
        global_hook.remove()
        if local_hook is not None:
            local_hook.remove()
        model_hook.remove()
        self.local_model = copy.deepcopy(model).to(torch.device('cpu'))
        self.local_model.freeze_grad()
        return