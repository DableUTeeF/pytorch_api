import torch
from torch import nn
import time
from .utils import Progbar
from torch.optim.optimizer import Optimizer

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
            assert isinstance(optimizer, Optimizer), 'Optimizer should be an Optimizer object'
            self.optimizer = optimizer
        if loss is not None:
            if isinstance(loss, str):
                if loss == 'categorical_crossentropy' or loss == 'crossentropy':
                    self.loss = nn.CrossEntropyLoss()
                elif loss == 'binary_crossentropy' or loss == 'bce':
                    self.loss = nn.BCELoss()
                else:
                    raise ValueError('Invalid string loss')
            self.loss = loss
        else:
            self.loss = nn.BCELoss()
        self.metric = []
        if type(metric) != list:
            if metric == 'acc' or metric == 'accuracy':
                if isinstance(loss, nn.BCELoss):
                    self.metric = [self.bce_accuracy()]
                elif isinstance(loss, nn.CrossEntropyLoss):  # loss == nn.CrossEntropyLoss():
                    self.metric = [self.categorical_accuracy()]
                else:
                    raise ValueError('String metric should use with nn.BCELoss or nn.CrossEntropyLoss, found {}'.format(self.loss))
            else:
                self.metric = [metric]
        else:
            for m in metric:
                if metric == 'acc' or metric == 'accuracy':
                    if isinstance(loss, nn.BCELoss):
                        self.metric.append(self.bce_accuracy())
                    elif isinstance(loss, nn.CrossEntropyLoss):  # loss == nn.CrossEntropyLoss():
                        self.metric.append(self.categorical_accuracy())
                    else:
                        raise ValueError(
                            'String metric should use with nn.BCELoss or nn.CrossEntropyLoss, found {}'.format(
                                self.loss))
                else:
                    self.metric.append(m)

        self.metric.append(self.loss)
        self.to(device)

    def fit_generator(self, generator, epoch, validation_data=None, schedule=None):
        if len(generator) < 1:
            raise ValueError('generator length < 0')
        if self.loss is None:
            self.compile('sgd', None)
        if type(schedule) == list or type(schedule) == tuple:
            schedule = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            schedule)
        log = {}
        # todo: change enumerate(generator) to keras enqueuer
        # todo: test enqueuer multithread and DataLoader multiprocess speed
        # todo: add evaluate, predict from array
        # todo: use function.__name__ and object.__str__() separately
        # todo: use train_on_batch and add log from every batch option
        # todo: add auto permute option
        # todo: delay before second KeyboardInterupt
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
                        if hasattr(metric, '__name__'):
                            mname = metric.__name__
                        else:
                            mname = metric.__str__()[:-2]
                        m_out = metric(output, targets)
                        if mname not in history_log:
                            history_log[mname] = m_out.cpu().detach().numpy()
                            printlog.append([mname, m_out.cpu().detach().numpy()])
                        else:
                            history_log[mname] += m_out.cpu().detach().numpy()
                            printlog.append([mname, history_log[mname]/(idx+1)])

                    self.optimizer.zero_grad()
                    m_out.backward()
                    self.optimizer.step()
                    total += inputs.size(0)

                    progbar.update(idx, printlog)

                for h in history_log:
                    history_log[h] = history_log[h] / len(generator)
                metrics = []
                if validation_data:
                    val_metrics = self.evaluate_generator(validation_data)
                    for metric in val_metrics:
                        metrics.append(['val_'+metric, val_metrics[metric]])
                        if 'val_'+metric not in log:
                            log['val_'+metric] = []
                        log['val_'+metric].append(val_metrics[metric])
                progbar.update(len(generator), metrics, force=True)
                if schedule:
                    schedule.step()
                for key in history_log:
                    if key not in log:
                        log[key] = []
                    log[key].append(history_log[key])
        except KeyboardInterrupt:
            pass

        finally:
            return log

    def evaluate_generator(self, generator):
        if len(generator) < 1:
            raise ValueError('generator length < 0')
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
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                for metric in self.metric:
                    if hasattr(metric, '__name__'):
                        mname = metric.__name__
                    else:
                        mname = metric.__str__()[:-2]
                    m_out = metric(outputs, targets)
                    if mname not in history_log:
                        history_log[mname] = m_out.cpu().detach().numpy()
                    else:
                        history_log[mname] += m_out.cpu().detach().numpy()
                total += inputs.size(0)
            for h in history_log:
                history_log[h] = history_log[h] / len(generator)
        return history_log

    def predict_generator(self, generator):
        self.lastext = ''
        self.start_epoch_time = time.time()
        self.model.eval()
        prd = []
        with torch.no_grad():
            for idx, inputs in enumerate(generator):
                if type(inputs) == tuple:
                    inputs = inputs[0]
                inputs = inputs.to(self.device)
                prd.append(self.model(inputs))
        return prd

    def fit(self, x, y, batch_size, epoch, validation_data=None, lrstep=None):
        if len(x) != len(y):
            raise ValueError('x and y should have same length')
        if self.loss is None:
            self.compile('sgd', None)
        if lrstep:
            schedule = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            lrstep)
        step_per_epoch = round(len(x)/batch_size)
        log = {}
        try:
            for e in range(epoch):
                print('Epoch:', e+1)
                self.lastext = ''
                self.start_epoch_time = time.time()
                self.last_print_time = self.start_epoch_time
                total = 0
                self.model.train()
                progbar = Progbar(len(x))
                history_log = {}
                for idx in range(0, len(x), batch_size):
                    inputs, targets = x[idx:idx+batch_size], y[idx:idx+batch_size]
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
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
                    history_log[h] = history_log[h] / step_per_epoch
                metrics = []
                if validation_data:
                    val_metrics = self.evaluate_generator(validation_data)
                    for metric in val_metrics:
                        metrics.append(['val_'+metric, val_metrics[metric]])
                        if 'val_'+metric not in log:
                            log['val_'+metric] = []
                        log['val_'+metric].append(val_metrics[metric])
                progbar.update(step_per_epoch, metrics, force=True)
                if lrstep:
                    schedule.step()
                for key in history_log:
                    if key not in log:
                        log[key] = []
                    log[key].append(history_log[key])
        except KeyboardInterrupt:
            pass

        finally:
            return log

    class bce_accuracy:
        def __call__(self, inputs, targets):
            predict = torch.round(inputs)
            return torch.sum(predict == targets.float()).double()/targets.size(0)

        def __str__(self):
            return 'bce_accuracy()'

    class categorical_accuracy:
        def __call__(self, inputs, targets):
            _, predicted = inputs.max(1)
            return predicted.eq(targets.long()).double().sum()/targets.size(0)

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
