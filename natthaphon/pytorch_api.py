import torch
from torch import nn
import time
from .utils import Progbar, NumpyArrayGenerator, GeneratorEnqueuer
from torch.optim.optimizer import Optimizer
import os
import json
import numpy as np


class Model:
    r"""
    Pytorch high-level API similar to keras.
    Example usage
    >>> model = Model(nn.Module())
    >>> model.compile(torch.optim.SGD(model.model.parameters(), 0.1, 0.9), nn.MSELoss(), device='cuda')
    >>> log = model.fit(x, y, batch_size=32, epoch=5)
    >>> model.save_weights('weights.t7')
    >>> with open('log.json', 'r') as wr:
    >>>     json.dump(log, wr)
    """

    def __init__(self, model, optimizer=None, loss=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = 'cpu'
        self.metric = [loss]

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
            self.loss = nn.MSELoss()
        self.metric = []
        if type(metric) != list:
            if metric == 'acc' or metric == 'accuracy':
                if isinstance(loss, nn.BCELoss):
                    self.metric = [self.bce_accuracy()]
                elif isinstance(loss, nn.CrossEntropyLoss):  # loss == nn.CrossEntropyLoss():
                    self.metric = [self.categorical_accuracy()]
                else:
                    raise ValueError(
                        'String metric should use with nn.BCELoss or nn.CrossEntropyLoss, found {}'.format(self.loss))
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

    def fit_generator(self, generator,
                      epoch,
                      validation_data=None,
                      schedule=None,
                      data_format='channel_first'):
        r"""
        :param generator: Training Data Generator - should return x, y in __iter__.
                          x: A torch tensor or numpy ndarray with shape of (batch, channel, *shape) if
                             data_format is 'channel_first' or (batch, *shape, channel) if 'channel_last'
                          y: A torch tensor or numpy ndarray with same shape as what the loss function and
                             the calculation metrics desire
        :param epoch: Num epoch to train
        :param validation_data: A generator or a list of [x, y, batch_size] for validation
        :param schedule: On epoch end function, must contains step()
        :param data_format: Generator's x data format, if 'channel_last', permute input x to channel first before flow through the model
        :return: Training and validation log
        """
        assert len(generator) > 0, 'generator length must larger than 0'
        if self.loss is None:
            self.compile('sgd', None)
        if type(schedule) == list or type(schedule) == tuple:
            schedule = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            schedule)
        if not hasattr(validation_data, '__iter__') and validation_data is not None:
            assert len(validation_data) == 3, 'validation_data should be a list or a tuple of [x, y, batch_size]'
            validation_data = NumpyArrayGenerator(*validation_data)
        log = {}
        # todo: use train_on_batch and add log from every batch option
        try:
            for e in range(epoch):
                print('Epoch: {}/{}'.format(e + 1, epoch))
                self.lastext = ''
                self.start_epoch_time = time.time()
                self.last_print_time = self.start_epoch_time
                total = 0
                self.model.train()
                progbar = Progbar(len(generator))
                history_log = {}
                for idx, (inputs, targets) in enumerate(generator):
                    if isinstance(inputs, np.ndarray):
                        inputs = torch.from_numpy(inputs)
                    if isinstance(targets, np.ndarray):
                        targets = torch.from_numpy(targets)
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    if data_format == 'channel_last':
                        inputs_dims = range(len(inputs.size()))
                        inputs = inputs.permute(0, -1, *inputs_dims[1:-1]).float()
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
                        else:
                            history_log[mname] += m_out.cpu().detach().numpy()
                        printlog.append([mname, m_out.cpu().detach().numpy()])

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
                        metrics.append(['val_' + metric, val_metrics[metric]])
                        if 'val_' + metric not in log:
                            log['val_' + metric] = []
                        log['val_' + metric].append(val_metrics[metric])
                progbar.update(len(generator), metrics)
                if schedule:
                    schedule.step()
                for key in history_log:
                    if key not in log:
                        log[key] = []
                    log[key].append(history_log[key])
        except KeyboardInterrupt:
            print('\nCanceled, returning log in 5 second')
            print('Press cancel again to raise KeyboardInterrupt')
            time.sleep(5)
            return log
        except Exception as e:
            cwd = os.getcwd()
            with open(os.path.join(cwd, 'temp.json'), 'w') as wr:
                json.dump(log, wr)
            raise Exception(e)
        return log
        # finally:
        #     return log

    def evaluate_generator(self, generator, data_format='channel_first'):
        assert len(generator) > 0, 'generator length must larger than 0'
        if self.loss is None:
            self.compile('sgd', None)
        self.lastext = ''
        self.start_epoch_time = time.time()
        total = 0
        self.model.eval()
        history_log = {}
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(generator):
                if isinstance(inputs, np.ndarray):
                    inputs = torch.from_numpy(inputs)
                if isinstance(targets, np.ndarray):
                    targets = torch.from_numpy(targets)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if data_format == 'channel_last':
                    inputs_dims = range(len(inputs.size()))
                    inputs = inputs.permute(0, -1, *inputs_dims[1:-1]).float()
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

    def predict_generator(self, generator, data_format='channel_first'):
        self.model.eval()
        prd = None
        with torch.no_grad():
            for idx, inputs in enumerate(generator):
                if type(inputs) == tuple:
                    inputs = inputs[0]
                if isinstance(inputs, np.ndarray):
                    inputs = torch.from_numpy(inputs)
                inputs = inputs.to(self.device)
                if data_format == 'channel_last':
                    inputs_dims = range(len(inputs.size()))
                    inputs = inputs.permute(0, -1, *inputs_dims[1:-1]).float()
                inputs = inputs.to(self.device)
                if prd is None:
                    prd = self.model(inputs).cpu().detach().numpy()
                else:
                    prd = np.concatenate((prd, self.model(inputs).cpu().detach().numpy()),
                                         axis=0)
        return prd

    def fit(self, x, y,
            batch_size,
            epoch,
            validation_data=None,
            schedule=None,
            data_format='channel_first'):
        """
        :param x: Numpy array of inputs
        :param y: Numpy array of target
        :param batch_size: Train batch size
        :param epoch: Epoch to train
        :param validation_data: A generator or a list of [x, y, batch_size] for validation
        :param schedule: On epoch end function, must contains step()
        :param data_format: x's data format, if 'channel_last', permute input x to channel first before flow through the model
        :return: Training and validation log
        """
        assert len(x) == len(y), 'x and y should have same length'
        assert epoch > 0, 'Epoch should greater than 0'
        if not hasattr(validation_data, '__getitem__') and validation_data is not None:
            assert len(validation_data) == 3, 'validation_data should be a list or a tuple of [x, y, batch_size]'
            validation_data = NumpyArrayGenerator(*validation_data)
        return self.fit_generator(NumpyArrayGenerator(x, y, batch_size),
                                  epoch=epoch,
                                  validation_data=validation_data,
                                  schedule=schedule,
                                  data_format=data_format
                                  )

    def evaluate(self, x, y,
                 batch_size=1,
                 data_format='channel_first'):
        if len(x) == len(y):
            raise ValueError('x and y should have same length')
        return self.evaluate_generator(NumpyArrayGenerator(x, y, batch_size),
                                       data_format=data_format
                                       )

    def predict(self, x, batch_size=1, data_format='channel_first'):
        self.model.eval()
        prd = None
        lenx = int(np.ceil(len(x) / batch_size))
        for i in range(lenx):
            inputs = x[i: i + batch_size]
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs)
            inputs = inputs.to(self.device)
            if data_format == 'channel_last':
                inputs_dims = range(len(inputs.size()))
                inputs = inputs.permute(0, -1, *inputs_dims[1:-1]).float()
            if prd is None:
                prd = self.model(inputs).cpu().detach().numpy()
            else:
                prd = np.concatenate((prd, self.model(inputs).cpu().detach().numpy()),
                                     axis=0)
        return prd

    def fit_enqueuer(self, generator,
                     epoch,
                     validation_data=None,
                     schedule=None,
                     data_format='channel_first',
                     num_worker=1):
        assert epoch > 0, 'Epoch should greater than 0'
        if not hasattr(validation_data, '__getitem__') and validation_data is not None:
            assert len(validation_data) == 3, 'validation_data should be a list or a tuple of [x, y, batch_size]'
            validation_data = NumpyArrayGenerator(*validation_data)
        train_enqueuer = GeneratorEnqueuer(generator)
        train_enqueuer.start(workers=num_worker)
        train_generator = train_enqueuer.get()
        if validation_data is not None:
            val_enqueuer = GeneratorEnqueuer(validation_data)
            val_enqueuer.start(workers=num_worker)
            val_generator = val_enqueuer.get()
        else:
            val_generator = None
        return self.fit_generator(train_generator,
                                  epoch=epoch,
                                  validation_data=val_generator,
                                  schedule=schedule,
                                  data_format=data_format
                                  )

    def evaluate_enqueuer(self, generator,
                          data_format='channel_first',
                          num_worker=1):
        val_enqueuer = GeneratorEnqueuer(generator)
        val_enqueuer.start(workers=num_worker)
        val_generator = val_enqueuer.get()
        return self.evaluate_generator(val_generator,
                                       data_format=data_format
                                       )

    def predict_enqueuer(self, generator,
                         data_format='channel_first',
                         num_worker=1):
        enqueuer = GeneratorEnqueuer(generator)
        enqueuer.start(workers=num_worker)
        generator = enqueuer.get()
        return self.predict_generator(generator,
                                      data_format=data_format
                                      )

    @staticmethod
    def bce_accuracy():
        def __call__(inputs, targets):
            predict = torch.round(inputs)
            return torch.sum(predict == targets.float()).double() / targets.size(0)
        return __call__

    @staticmethod
    def categorical_accuracy():
        def __call__(inputs, targets):
            _, predicted = inputs.max(1)
            return predicted.eq(targets.long()).double().sum() / targets.size(0)
        return __call__

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
