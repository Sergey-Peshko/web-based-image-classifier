class DataSet(object):
    def __init__(self, images, labels):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def get_all(self):
        return self._images, self._labels

    def get_part(self):
        return self._images[0: int(self._num_examples/3)], self._labels[0: int(self._num_examples/3)]
