import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models import Conv1D, Conv2D, LSTM
from tqdm import tqdm
from glob import glob
import argparse
import warnings


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = label

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train(args):
    src_root = args.src_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params = {'N_CLASSES':5,
              'SR':sr,
              'DT':dt}
    models = {'conv1d':Conv1D(**params),
              'conv2d':Conv2D(**params),
              'lstm':  LSTM(**params)}
    assert model_type in models.keys(), '{} not an available model'.format(model_type)
    csv_path = os.path.join('logs', '{}_history.csv'.format(model_type))

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = ['Cymbal', 'Hi_Hat', 'Kick', 'Snare', 'Tom']
    x = wav_paths[0]
    labels = np.zeros((len(wav_paths), 5), dtype = float)
    for i in range(len(wav_paths)):
      title = os.path.split(wav_paths[i])[1]
      title = title.split("_")
      #normal case
      if len(title) <= 4:
        tmp = int(title[0])
        if tmp == 57 or tmp == 59 or tmp == 59 or tmp == 53 or tmp == 52 or tmp == 51 or tmp == 49 :
          labels[i][0] = 1
        elif tmp == 46 or tmp == 44 or tmp == 42 :
          labels[i][1] = 1
        elif tmp == 35 or tmp == 36  :
          labels[i][2] = 1
        elif tmp == 37 or tmp == 38 or tmp == 40 :
          labels[i][3] = 1
        elif tmp == 50 or tmp == 48 or tmp == 47 or tmp == 45 or tmp == 43 or tmp == 41 :
          labels[i][4] = 1  
      #mixup case
      else:
        tmpboth = [int(title[0]),int(title[3])]
        for tmp in tmpboth:
          if tmp == 57 or tmp == 59 or tmp == 59 or tmp == 53 or tmp == 52 or tmp == 51 or tmp == 49 :
            labels[i][0] = 1
          elif tmp == 46 or tmp == 44 or tmp == 42 :
            labels[i][1] = 1
          elif tmp == 35 or tmp == 36  :
            labels[i][2] = 1
          elif tmp == 37 or tmp == 38 or tmp == 40 :
            labels[i][3] = 1
          elif tmp == 50 or tmp == 48 or tmp == 47 or tmp == 45 or tmp == 43 or tmp == 41 :
            labels[i][4] = 1 
        

    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths,
                                                                  labels,
                                                                  test_size=0.1,
                                                                  random_state=0)

    assert len(label_train) >= args.batch_size, 'Number of train samples must be >= batch_size'

    tg = DataGenerator(wav_train, label_train, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    vg = DataGenerator(wav_val, label_val, sr, dt,
                       params['N_CLASSES'], batch_size=batch_size)
    model = models[model_type]
    cp = ModelCheckpoint('models/{}.multi2'.format(model_type), monitor='val_loss',
                         save_best_only=True, save_weights_only=False,
                         mode='auto', save_freq='epoch', verbose=1)
    csv_logger = CSVLogger(csv_path, append=False)
    model.fit(tg, validation_data=vg,
              epochs=30, verbose=1,
              callbacks=[csv_logger, cp])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_type', type=str, default='conv2d',
                        help='model to run. i.e. conv1d, conv2d, lstm')
    parser.add_argument('--src_root', type=str, default='clean_drum_norm1600',
                        help='directory of audio files in total duration')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sample_rate', '-sr', type=int, default=16000,
                        help='sample rate of clean audio')
    args, _ = parser.parse_known_args()

    train(args)

