class DataSetEpochManager(object):
    def __init__(self, dataset):

        self._dataset = dataset
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """
        Return the next `batch_size` examples from this data set.

        :param batch_size:
        :return:
        """

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._dataset.num_examples:
            # set to last image in data set
            self._index_in_epoch = self._dataset.num_examples - 1
            assert batch_size <= self._dataset.num_examples

        if self._index_in_epoch == (self._dataset.num_examples - 1):
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        return self._dataset.images[start:end], self._dataset.labels[start:end]
