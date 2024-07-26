from torchaudio.datasets import SPEECHCOMMANDS
import torch
import torch.nn as nn
import torch.nn.functional as F
import flgo.benchmark
import torch.utils.data as tud
import numpy as np
import torchaudio

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
raw_data = SPEECHCOMMANDS(flgo.benchmark.data_root, download=True, subset='training')
sample_speaker_ids = [raw_data.get_metadata(i)[3] for i in range(len(raw_data))]
speaker_ids = set(sample_speaker_ids)
speaker_maps = {sid:i for i,sid in enumerate(speaker_ids)}
sample_speaker_ids = np.array(list(map(lambda x: speaker_maps[x], sample_speaker_ids)))
data_idxs = [np.where(sample_speaker_ids == i)[0].tolist() for i in range(len(speaker_ids))]

train_data = [tud.Subset(raw_data, didx) for didx in data_idxs]
test_data = SPEECHCOMMANDS(flgo.benchmark.data_root, download=True, subset='testing')
val_data = SPEECHCOMMANDS(flgo.benchmark.data_root, download=True, subset='validation')

transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)

loss_fn = nn.CrossEntropyLoss()

def label_to_index(word):
    return torch.tensor(labels.index(word))

def index_to_label(index):
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch[0] = nn.ConstantPad1d((0, 16000 - batch[0].shape[0]), 0)(batch[0].squeeze(-1)).unsqueeze(-1)
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return transform(tensors), targets

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze(1)

# class M5(nn.Module):
#     def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
#         super().__init__()
#         self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
#         self.pool1 = nn.MaxPool1d(4)
#         self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
#         self.pool2 = nn.MaxPool1d(4)
#         self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
#         self.pool3 = nn.MaxPool1d(4)
#         self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
#         self.pool4 = nn.MaxPool1d(4)
#         self.fc1 = nn.Linear(2 * n_channel, n_output)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool3(x)
#         x = F.relu(self.conv4(x))
#         x = self.pool4(x)
#         x = F.avg_pool1d(x, x.shape[-1])
#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=2)

def get_model():
    return M5(n_input=1, n_output=len(labels))

def data_to_device(batch_data, device):
    return batch_data[0].to(device), batch_data[1].to(device)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def eval(model, data_loader, device) -> dict:
    model.eval()
    model.to(device)
    loss_total = 0.0
    correct = 0
    num_total = 0
    for batch_id, batch_data in enumerate(data_loader):
        batch_data = data_to_device(batch_data, device)
        ypred = model(batch_data[0])
        pred = get_likely_index(ypred)
        correct += number_of_correct(pred, batch_data[-1])
        loss = loss_fn(ypred.squeeze(1), batch_data[1])
        num_total += len(batch_data[1])
        loss_total += loss.item()*len(batch_data[1])
    return {'loss': loss_total/num_total, 'accuracy': correct/num_total}

def compute_loss(model, batch_data, device) -> dict:
    batch_data = data_to_device(batch_data, device)
    model = model.to(device)
    ypred = model(batch_data[0])
    loss = loss_fn(ypred.squeeze(1), batch_data[1])
    return {'loss': loss}
