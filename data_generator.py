import numpy as np
import tensorflow.keras
import tables  # load compressed data files
from tensorflow import constant  # used to convert array/list to a Keras Tensor

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras - the first batch is not properly read and used, so it reads it again (2 times)'
    def __init__(self, x_filename, y_filename, numb_batches_per_epoch, batch_size=32, shuffle=False):
        """
        Initialization
        :param x_filename: the name of the x file
        :param y_filename: the name of the y file
        :param numb_batches_per_epoch: number of batches (per epoch)
        :param batch_size: the default batch size is 1024
        :param shuffle: whether to shuffle the data
        """
        self.batch_size = batch_size
        self.numb_batches_per_epoch = int(numb_batches_per_epoch)
        self.x_filename = x_filename
        self.y_filename = y_filename
        # create array with all indices (e.g., for train: 0 through 530432 with step batch_size=1024) - to read data
        self.indexes = np.arange(int(numb_batches_per_epoch)*batch_size, step=batch_size)
        print('self.indexes', self.indexes)
        self.shuffle = shuffle
        self.batch_count_progress = 0  # count the number of batches that have read
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        print('__len__', self.numb_batches_per_epoch)
        return self.numb_batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        print('__getitem__    -----------    index: ', index)
        # Generate indexes of the batch
        current_batch_number = self.indexes[index]

        # Generate data
        '''
        if 'TRAIN' in self.x_filename:
            X, y, weights = self.__data_generation(current_batch_number)
            return X, y, weights
        el
        '''
        if not self.y_filename == '':
            X, y = self.__data_generation(current_batch_number)
            return X, y
        else:  # for TEST data read only the x values
            X = self.__data_generation(current_batch_number)
            return X


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print('on_epoch_end')
        self.batch_count_progress = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, current_batch_number):
        'Generates data containing batch_size samples'
        print('__data_generation    -----------    {} / {}    -----------    batch_number: {}'
              .format(self.batch_count_progress, self.numb_batches_per_epoch, current_batch_number))
        self.batch_count_progress += 1
        print(self.x_filename)

        # Read X batches for testing from file (pre-processed)
        with tables.File(self.x_filename, 'r') as h5f:
            x = h5f.get_node('/x_data' + str(current_batch_number)).read()  # get a specific chunk of data
            # print(x)
        #        print('X SHAPE AFTER', np.array(x, dtype=object).shape)

        if not self.y_filename == '':  # for TEST data read only the x values
            # Read y batches for testing from file (pre-processed)
            with tables.File(self.y_filename, 'r') as h5f:
                y = h5f.get_node('/y_data' + str(current_batch_number)).read()  # get a specific chunk of data
                # print(y)
        #            print('y SHAPE AFTER', np.array(y, dtype=object).shape)

        '''
        if 'TRAIN' in self.x_filename:  # for training return class weights as well
            """
                # FULL ABSTRACT
                KP count:  3882211 (78247442 / 3882211 = 20.15)
                KP WORDS count:  6584027 (78247442 / 6584027 = 11.88)
                NON-KP count:  78247442

                # SENTENCES
                KP count:  3863958  (77728129 / 3863958 = 20.11)
                KP WORDS count:  6534606  (77728129 / 6534606 = 11.89)
                NON-KP TEST count:  77728129
            """
            # The weight of label 0 (Non-KP) is 1 and the weight of class 1 (KP) is the number of occurences of class 0 (78247442 / 6584027 = 11.88)
            sample_weights = [[1 if label[0] else 11.88 for label in instance] for instance in y]
            print('class_weights', np.array(sample_weights, dtype=float).shape)
            return np.array(x), constant(y), np.array(sample_weights, dtype=float)  # (inputs, targets, sample_weights)
        '''




        if self.y_filename == '':  # for TEST data return only the x values
            return np.array(x)

        return np.array(x), constant(y)
