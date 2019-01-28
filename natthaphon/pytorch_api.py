import torch
from torch import nn
import time
import sys
import numpy as np


class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)

    def update(self, current, values=None, force=False):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self.start)
        if self.verbose == 1:
            if (not force and (now - self.last_update) < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self.total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self.total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = np.mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += (' ' * (prev_total_width - self.total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self.last_update = now

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)


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

    def compile(self, optimizer, loss, metric=None):
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
        if metric == 'acc' or 'accuracy':
            if isinstance(loss, nn.BCELoss):
                self.metric = [self.bce_accuracy()]
            elif isinstance(loss, nn.CrossEntropyLoss):  # loss == nn.CrossEntropyLoss():
                self.metric = [self.categorical_accuracy()]
            else:
                raise ValueError('String metric should use with nn.BCELoss or nn.CrossEntropyLoss, found {}'.format(self.loss))
        else:
            self.metric = [metric]
        self.metric.append(self.loss)

    def fit_generator(self, generator, epoch, validation_data=None, lrstep=None):
        if self.loss is None:
            self.compile('sgd', None)
        if lrstep:
            schedule = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            lrstep)
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
