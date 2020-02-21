import tensorflow as tf
import keras
import numpy as np

from keras.utils import Sequence

class MyGenerator(Sequence):
    """Custom generator"""

    def __init__(self, train_paths, label_paths, 
                 batch_size=1, width=50, height=50, ch=3, timesteps=10,camera_num=5):
        """construction   

        :param train_paths: List of image file  
        :param label_paths: List of label file  
        :param batch_size: Batch size  
        :param width: Image width  
        :param height: Image height  
        :param ch: Num of image channels  
        :param camera_num: Num of camera 
        """

        self.train_paths = train_paths
        self.label_paths = label_paths
        self.length = len(train_paths)
        self.batch_size = batch_size
        self.sequence_len = (2*timesteps) + 1
        self.width = width
        self.height = height
        self.ch = ch
        self.camera_num = camera_num
        self.num_batches_per_epoch = int((self.length - 1) / batch_size) + 1


    def __getitem__(self, idx):
        """Get batch data   

        :param idx: Index of batch  

        :return imgs: numpy array of images 
        :return labels: numpy array of label  
        """

        start_pos = self.batch_size * idx
        end_pos = start_pos + self.batch_size
        if end_pos > self.length:
            end_pos = self.length
        item_trains = self.train_paths[start_pos : end_pos]
        item_labels = self.label_paths[start_pos : end_pos]
        trains = [np.empty((len(item_trains), self.sequence_len, self.height, self.width, self.ch), dtype=np.float32)]*self.camera_num
        labels = np.empty((len(item_trains), 5), dtype=np.float32)

        for i, (item_train, item_label) in enumerate(zip(item_trains, item_labels)):
            train = np.load(item_train)
            label = np.load(item_label)
            tmp_label=np.empty(5, dtype=np.float32)
            for j in range(self.camera_num):
                trains[j][i, :] = train[j]
                tmp_label[j] = label[j]
            labels[i] = tmp_label

        X_batch = [trains[0], trains[1], trains[2], trains[3], trains[4]]

        return X_batch, labels


    def __len__(self):
        """Batch length"""

        return self.num_batches_per_epoch


    def on_epoch_end(self):
        """Task when end of epoch"""
        pass