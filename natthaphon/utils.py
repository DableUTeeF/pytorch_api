import time
import sys
import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO
import os
import six
import traceback
import multiprocessing
import threading
import queue


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


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def text_summary(self, tag, value, step):
        """Log a scalar variable."""
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        text_tensor = tf.make_tensor_proto(value, dtype=tf.string)

        summary = tf.Summary()
        # value=[tf.Summary.Value(tag=tag, simple_value=meta, tensor=text_tensor)]
        summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
        self.writer.add_summary(summary, step)


class GeneratorEnqueuer:
    """Builds a queue out of a data generator.

    The provided generator can be finite in which case the class will throw
    a `StopIteration` exception.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each worker.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 seed=None):
        self.wait_time = wait_time
        self._generator = generator
        if os.name is 'nt' and use_multiprocessing is True:
            # On Windows, avoid **SYSTEMATIC** error in `multiprocessing`:
            # `TypeError: can't pickle generator objects`
            # => Suggest multithreading instead of multiprocessing on Windows
            raise ValueError('Using a generator with `use_multiprocessing=True`'
                             ' is not supported on Windows (no marshalling of'
                             ' generators across process boundaries). Instead,'
                             ' use single thread/process or multithreading.')
        else:
            self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self._manager = None
        self.queue = None
        self.seed = seed

    def _data_generator_task(self):
        if self._use_multiprocessing is False:
            while not self._stop_event.is_set():
                with self.genlock:
                    try:
                        if (self.queue is not None and
                                self.queue.qsize() < self.max_queue_size):
                            # On all OSes, avoid **SYSTEMATIC** error
                            # in multithreading mode:
                            # `ValueError: generator already executing`
                            # => Serialize calls to
                            # infinite iterator/generator's next() function
                            generator_output = next(self._generator)
                            self.queue.put((True, generator_output))
                        else:
                            time.sleep(self.wait_time)
                    except StopIteration:
                        break
                    except Exception as e:
                        # Can't pickle tracebacks.
                        # As a compromise, print the traceback and pickle None instead.
                        if not hasattr(e, '__traceback__'):
                            setattr(e, '__traceback__', sys.exc_info()[2])
                        self.queue.put((False, e))
                        self._stop_event.set()
                        break
        else:
            while not self._stop_event.is_set():
                try:
                    if (self.queue is not None and
                            self.queue.qsize() < self.max_queue_size):
                        generator_output = next(self._generator)
                        self.queue.put((True, generator_output))
                    else:
                        time.sleep(self.wait_time)
                except StopIteration:
                    break
                except Exception as e:
                    # Can't pickle tracebacks.
                    # As a compromise, print the traceback and pickle None instead.
                    traceback.print_exc()
                    setattr(e, '__traceback__', None)
                    self.queue.put((False, e))
                    self._stop_event.set()
                    break

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """
        try:
            self.max_queue_size = max_queue_size
            if self._use_multiprocessing:
                self._manager = multiprocessing.Manager()
                self.queue = self._manager.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                # On all OSes, avoid **SYSTEMATIC** error in multithreading mode:
                # `ValueError: generator already executing`
                # => Serialize calls to infinite iterator/generator's next() function
                self.genlock = threading.Lock()
                self.queue = queue.Queue(maxsize=max_queue_size)
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.seed)
                    thread = multiprocessing.Process(target=self._data_generator_task)
                    thread.daemon = True
                    if self.seed is not None:
                        self.seed += 1
                else:
                    thread = threading.Thread(target=self._data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if self._use_multiprocessing:
                if thread.is_alive():
                    thread.terminate()
            else:
                # The thread.is_alive() test is subject to a race condition:
                # the thread could terminate right after the test and before the
                # join, rendering this test meaningless -> Call thread.join()
                # always, which is ok no matter what the status of the thread.
                thread.join(timeout)

        if self._manager:
            self._manager.shutdown()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Yields
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        while self.is_running():
            if not self.queue.empty():
                success, value = self.queue.get()
                # Rethrow any exceptions found in the queue
                if not success:
                    six.reraise(value.__class__, value, value.__traceback__)
                # Yield regular values
                if value is not None:
                    yield value
            else:
                all_finished = all([not thread.is_alive() for thread in self._threads])
                if all_finished and self.queue.empty():
                    raise StopIteration()
                else:
                    time.sleep(self.wait_time)

        # Make sure to rethrow the first exception in the queue, if any
        while not self.queue.empty():
            success, value = self.queue.get()
            if not success:
                six.reraise(value.__class__, value, value.__traceback__)


class NumpyArrayGenerator:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.curidx = -1

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        i = idx * self.batch_size
        x = self.x[i:i + self.batch_size]
        y = self.y[i:i + self.batch_size]
        return x, y

    def __next__(self):
        self.curidx += 1
        return self[self.curidx]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
