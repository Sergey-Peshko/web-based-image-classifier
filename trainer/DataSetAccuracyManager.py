class DataSetAccuracyManager(object):
    def __init__(self, dataset, session, x, y_true, batch_size):
        self._dataset = dataset
        self._index_in_epoch = 0
        if batch_size > self._dataset.num_examples:
            self._batch_size = self._dataset.num_examples
        else:
            self._batch_size = batch_size
        self._session = session
        self._x = x
        self._y_true = y_true

    def next_batch(self):
        """
        Return the next `batch_size` examples from this data set.

        :return:
        """

        start = self._index_in_epoch
        self._index_in_epoch += self._batch_size

        if self._index_in_epoch >= (self._dataset.num_examples - 1):
            # set to last image in data set
            self._index_in_epoch = self._dataset.num_examples - 1
            assert self._batch_size <= self._dataset.num_examples

        end = self._index_in_epoch

        return self._dataset.images[start:end], self._dataset.labels[start:end]

    def calc_accuracy(self, accuracy_func):
        self._index_in_epoch = 0
        i = 0
        acc = 0
        while self._index_in_epoch != (self._dataset.num_examples - 1):
            i += 1

            # next epoch is come
            x_batch, y_true_batch = self.next_batch()

            feed_dict = {self._x: x_batch,
                         self._y_true: y_true_batch}

            acc = acc + ((self._session.run(accuracy_func, feed_dict=feed_dict) - acc)/float(i))

        return acc

    def calc_loss(self, cost_func):
        self._index_in_epoch = 0
        i = 0
        loss = 0
        while self._index_in_epoch != (self._dataset.num_examples - 1):
            i += 1

            # next epoch is come
            x_batch, y_true_batch = self.next_batch()

            feed_dict = {self._x: x_batch,
                         self._y_true: y_true_batch}

            loss = loss + ((self._session.run(cost_func, feed_dict=feed_dict) - loss) / float(i))

        return loss
