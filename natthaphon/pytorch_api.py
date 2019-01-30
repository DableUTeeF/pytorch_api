import torch
from torch import nn
import time
from .utils import Progbar


class Model:
    def __init__(self, model, optimizer=None, loss=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = 'cpu'

    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')

    def to(self, device):
        self.device = device
        self.model.to(self.device)
        self.loss.to(self.device)

    def compile(self, optimizer, loss, metric=None, device='cpu'):
        if optimizer in ['sgd', 'SGD']:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=0.1,
                                             momentum=0.9,
                                             weight_decay=1e-4)
        elif optimizer in ['adam', 'Adam']:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=0.01,
                                              weight_decay=1e-4)
        else:
            assert isinstance(optimizer, torch.optim.optimizer.Optimizer), 'Optimizer should be an Optimizer object'
            self.optimizer = optimizer
        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.BCELoss()
        if metric == 'acc' or metric == 'accuracy':
            if isinstance(loss, nn.BCELoss):
                self.metric = [self.bce_accuracy()]
            elif isinstance(loss, nn.CrossEntropyLoss):  # loss == nn.CrossEntropyLoss():
                self.metric = [self.categorical_accuracy()]
            else:
                raise ValueError('String metric should use with nn.BCELoss or nn.CrossEntropyLoss, found {}'.format(self.loss))
        else:
            self.metric = [metric]
        self.metric.append(self.loss)
        self.to(device)

    def fit_generator(self, generator, epoch, validation_data=None, lrstep=None):
        if self.loss is None:
            self.compile('sgd', None)
        if lrstep:
            schedule = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            lrstep)
        log = {}
        try:
            for e in range(epoch):
                print('Epoch:', e+1)
                self.lastext = ''
                self.start_epoch_time = time.time()
                self.last_print_time = self.start_epoch_time
                total = 0
                self.model.train()
                progbar = Progbar(len(generator))
                history_log = {}
                for idx, (inputs, targets) in enumerate(generator):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    # inputs = inputs.permute(0, 1, 4, 2, 3).float()
                    output = self.model(inputs)
                    printlog = []
                    for metric in self.metric:
                        m_out = metric(output, targets)
                        if metric.__str__()[:-2] not in history_log:
                            history_log[metric.__str__()[:-2]] = m_out.cpu().detach().numpy()
                            printlog.append([metric.__str__()[:-2], m_out.cpu().detach().numpy()])
                        else:
                            history_log[metric.__str__()[:-2]] += m_out.cpu().detach().numpy()
                            printlog.append([metric.__str__()[:-2], history_log[metric.__str__()[:-2]]/(idx+1)])

                    self.optimizer.zero_grad()
                    m_out.backward()
                    self.optimizer.step()
                    total += inputs.size(0)

                    progbar.update(idx-1, printlog)

                for h in history_log:
                    history_log[h] = history_log[h] / len(generator)
                metrics = []
                if validation_data:
                    val_metrics = self.evaluate_generator(validation_data)
                    for metric in val_metrics:
                        metrics.append(['val_'+metric, val_metrics[metric]])
                progbar.update(len(generator), metrics, force=True)
                if lrstep:
                    schedule.step()
                for key in history_log:
                    if key not in log:
                        log[key] = []
                    log[key].append(history_log[key])

        finally:
            return log

    def evaluate_generator(self, generator):
        if self.loss is None:
            self.compile('sgd', None)
        self.lastext = ''
        self.start_epoch_time = time.time()
        total = 0
        self.model.eval()
        history_log = {}
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(generator):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float()
                outputs = self.model(inputs)
                for metric in self.metric:
                    m_out = metric(outputs, targets)
                    if metric.__str__()[:-2] not in history_log:
                        history_log[metric.__str__()[:-2]] = m_out.cpu().detach().numpy()
                    else:
                        history_log[metric.__str__()[:-2]] += m_out.cpu().detach().numpy()
                total += inputs.size(0)
            for h in history_log:
                history_log[h] = history_log[h] / len(generator)
        return history_log

    class bce_accuracy:
        def __call__(self, inputs, targets):
            predict = torch.round(inputs)
            return torch.sum(predict == targets.float())

        def __str__(self):
            return 'bce_accuracy()'

    class categorical_accuracy:
        def __call__(self, inputs, targets):
            _, predicted = inputs.max(1)
            return predicted.eq(targets.long()).sum()

        def __str__(self):
            return 'categorical_accuracy()'

    def save_weights(self, path):
        state = {
            'optimizer': self.optimizer.state_dict(),
            'net': self.model.state_dict(),
        }
        torch.save(state, path)

    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
