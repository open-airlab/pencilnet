import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import numpy as np
import cv2
import datetime
import os
import random
import json

from utils_io import SyntheticDataset
from networks import Dronet, Dronet_half
from losses import mean_squarred_error, my_loss_2_2
from metrics import gate_center_mae_error, distance_mae_error, orientation_mae_error, gx_mae_error, gy_mae_error, gz_mae_error, o_mae_error
from logger import Logger

now = datetime.datetime.now()
name = now.strftime("%Y-%m-%d-%H-%M")
SAVE_DIR = './output-checkpoints'
VERBOSE = 1

config = {}
# Paths to annotation file and source data.
config['name'] = name
config['batch_size'] = 32
config['input_shape'] = (120, 160,3)
config['output_shape'] = (3,4,5)
config['epochs'] = 100
config['base_learning_rate'] = 0.0001  #0.001
config['lr_schedule'] = [(0.1, 5), (0.01, 8)]
config['l2_weight_decay'] = 2e-4
config['batch_norm_decay'] = 0.997
config['batch_norm_epsilon'] = 1e-5
config['optimizer'] = 'rmsprop'
config['loss'] = 'custom'

config["dataset_folder"] = "/home/huy/dataset_ws/Train_data/pencil_sim_training_images"

config['train_indices'] = os.path.join(config['dataset_folder'], 'train-indices.npy')
config['test_indices'] = os.path.join(config['dataset_folder'], 'test-indices.npy')
config['tensorboard_log_dir'] = "logs/fit/" + name
config['save_frequency'] = 10
config['info'] = 'Dronet -- au_dr and rmsprop and mae loss and bbox-output and lr 0.001'
config['metrics'] = [gate_center_mae_error, distance_mae_error, orientation_mae_error]



logger = Logger(name)
logger.set_save_dir("test_folder")
logger.set_config(config)

train_indices = np.load(config['train_indices'])
test_indices = np.load(config['test_indices'])


def train_generator():
    """ Generator for training samples."""
    s = SyntheticDataset(config['dataset_folder'], grid_shape=(config['output_shape'][1], config['output_shape'][0]))
    indices = train_indices
    random.shuffle(indices)
    print('Train samples: ', indices.shape)
    for i in indices:
        img, target = s.get_data_by_index(i)
        #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        # Normalize
        if img.shape != config['input_shape'] or target.shape!=config['output_shape']:
          #print("Will be converted: ", img.shape, target.shape)
          img = cv2.resize(img, (config['input_shape'][1], config['input_shape'][0]), interpolation=cv2.INTER_NEAREST)
          
        yield img/255., target
    

def test_generator():
    """ Generator for test samples."""
    s = SyntheticDataset(config['dataset_folder'], grid_shape=(config['output_shape'][1], config['output_shape'][0]))
    indices = test_indices
    #random.shuffle(indices)
    print('Test samples: ', indices.shape)
    for i in indices:
        img, target = s.get_data_by_index(i)
        #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        # Normalize
        if img.shape != config['input_shape'] or target.shape!=config['output_shape']:
          #print(img.shape, target.shape)
          img = cv2.resize(img, (config['input_shape'][1], config['input_shape'][0]), interpolation=cv2.INTER_NEAREST)

        yield img/255., target


def schedule(epoch):
  """ Schedule learning rate. """
  initial_learning_rate = config['base_learning_rate']
  learning_rate = initial_learning_rate
  for mult, start_epoch in config['lr_schedule']:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate


# Dataset preparation
train_dataset = tf.data.Dataset.from_generator(generator = train_generator,
                                            output_types = (tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape(config['input_shape']), tf.TensorShape(config['output_shape'])))
test_dataset = tf.data.Dataset.from_generator(generator = test_generator,
                                            output_types = (tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape(config['input_shape']), tf.TensorShape(config['output_shape'])))

# Preprocess data.
buffer_size = int(1024)
train_dataset = train_dataset.shuffle(buffer_size, reshuffle_each_iteration=True).batch(config['batch_size'], drop_remainder=True)
test_dataset = test_dataset.batch(config['batch_size'], drop_remainder=True)


# Choose either Dronet (DroNet - 1.0) or Dronet_half (DroNet - 0.5) for training

model = Dronet(config)
logger.set_network(Dronet)

# model = Dronet_half(config)
# logger.set_network(Dronet_half)


# Compile the model.
opt = config['optimizer']
model.compile(optimizer=opt, loss=[mean_squarred_error],
			  metrics=config['metrics'])
logger.set_loss(mean_squarred_error)
logger.save()

lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(schedule)


# Train the model.
history = model.fit(train_dataset,
          epochs=config['epochs'], 
          validation_data = test_dataset,
          validation_freq=1,
          callbacks=[lr_schedule_callback, 
                        logger.get_checkpoint_callback(), 
                        logger.get_tensorboard_callback(),
                        logger.get_csv_callback()], verbose=1)