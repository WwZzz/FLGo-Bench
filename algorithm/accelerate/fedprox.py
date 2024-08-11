import flgo.algorithm.fedbase as fedbase
import torch
from tqdm import tqdm
import ray
import torch.utils.data as tud
import copy
import flgo.simulator.base as ss
import flgo.utils.shared_memory as fus
import warnings
warnings.filterwarnings('ignore')

class Server(fedbase.BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'mu':0.1})
        self.sample_option = 'md' if self.proportion < 1.0 else 'full'
        self.aggregation_option = 'uniform'

    @ss.with_clock
    def communicate(self, selected_clients, mtype=0, asynchronous=False):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.

        Args:
            selected_clients (list of int): the clients to communicate with
            mtype (anytype): type of message
            asynchronous (bool): asynchronous communciation or synchronous communcation

        Returns:
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        received_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        # prepare packages for clients
        for client_id in communicate_clients:
            received_package_buffer[client_id] = None
        # communicate with selected clients
        if self.num_parallels <= 1:
            # computing iteratively
            for client_id in tqdm(communicate_clients, desc="Local Training on {} Clients".format(len(communicate_clients)), leave=False):
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                response_from_client_id = self.communicate_with(self.clients[client_id].id, package=server_pkg)
                packages_received_from_clients.append(response_from_client_id)
        else:
            paralleltype = self.option.get('parallel_type', 'obj')
            @ray.remote(num_gpus=torch.cuda.device_count()*1.0/self.num_parallels)
            def client_train_fedprox(data_ref, model, config, calculator):
                model.train()
                if config['parallel_type'] != 'obj':
                    data = fus.sharable2dataset(data_ref)
                else:
                    data = data_ref
                device = calculator.device
                data_loader = tud.DataLoader(data, batch_size=config['batch_size'], shuffle=True)
                optimizer = calculator.get_optimizer(model, lr=config['lr'], weight_decay=config['weight_decay'], momentum=config['momentum'])
                model.to(device)
                mu = config['mu']
                src_model = copy.deepcopy(model)
                src_model.freeze_grad()
                for epoch in range(config['num_epochs']):
                    for i, batch in enumerate(data_loader):
                        batch = calculator.to_device(batch)
                        model.zero_grad()
                        loss = calculator.compute_loss(model, batch)['loss']
                        loss_proximal = 0
                        for pm, ps in zip(model.parameters(), src_model.parameters()):
                            loss_proximal += torch.sum(torch.pow(pm - ps, 2))
                        loss = loss + 0.5 * mu * loss_proximal
                        loss.backward()
                        if config.get('clip_grad', -1) > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.get('clip_grad', 0.))
                        optimizer.step()
                return {'model': model.to('cpu')}
            self.model = self.model.to(torch.device('cpu'))
            if not hasattr(self, '_client_data_sharable'):
                if paralleltype != 'obj':
                    self._client_data_sharable = [fus.dataset2sharable(c.train_data) for c in self.clients]
                else:
                    self._client_data_sharable = [ray.put(c.train_data) for c in self.clients]
            for client_id in communicate_clients:
                c = self.clients[client_id]
                model = self.model
                config = {'mu': c.mu, 'lr':c.learning_rate, 'weight_decay':c.weight_decay, 'momentum':c.momentum, 'batch_size':c.batch_size, 'num_epochs':c.num_epochs, 'clip_grad':c.clip_grad}
                config['parallel_type'] = paralleltype
                calculator = self.gv.TaskCalculator(device='cuda',  optimizer_name=self.option.get('optimizer', 'SGD'),)
                cpkg = client_train_fedprox.remote(self._client_data_sharable[client_id], model, config, calculator)
                packages_received_from_clients.append(cpkg)
            ready_refs, remaining_refs = ray.wait(packages_received_from_clients, num_returns=len(communicate_clients), timeout=None)
            packages_received_from_clients = ray.get(ready_refs)
            self.model = self.model.to(self.device)
            for pkg in packages_received_from_clients:
                for k, v in pkg.items():
                    if hasattr(v, 'to'):
                        try:
                            pkg[k] = v.to(self.device)
                        except:
                            continue
        for i, client_id in enumerate(communicate_clients): received_package_buffer[client_id] = \
        packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if
                                          received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

class Client(fedbase.BasicClient):
    pass