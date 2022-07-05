import tensorflow as tf

import numpy as np
import cv2
import datetime
import os
import random

from utils_io import SyntheticDataset
from networks import PencilNet
from losses import multi_grid_loss
from metrics import gate_center_mae_error, distance_mae_error, orientation_mae_error
from logger import Logger

now = datetime.datetime.now()
name = now.strftime("%Y-%m-%d-%H-%M")

VERBOSE = 1

config = {}
# Paths to annotation file and source data.
config['name'] = name
config['batch_size'] = 32
config['input_shape'] = (120, 160,3)
config['output_shape'] = (3,4,5)
config['epochs'] = 500
config['base_learning_rate'] = 0.001
config['lr_schedule'] = [(0.1, 5), (0.01, 8)]
config['l2_weight_decay'] = 2e-4
config['batch_norm_decay'] = 0.997
config['batch_norm_epsilon'] = 1e-5
config['optimizer'] = 'adam'
config['loss'] = 'custom'

config["dataset_folder"] = "/home/huy/dataset_ws/Train_data/pencil_sim_training_images"


config['train_indices'] = os.path.join(config['dataset_folder'], 'train-indices.npy')
config['test_indices'] = os.path.join(config['dataset_folder'], 'test-indices.npy')
config['tensorboard_log_dir'] = "logs/fit/" + name
config['save_frequency'] = 10
config['info'] = 'au-dr, epoch 100  rmsprop and USE multi_grid_loss and lr 0.001'
config['metrics'] = [gate_center_mae_error, distance_mae_error, orientation_mae_error]



logger = Logger(name)
logger.set_save_dir("./trained_model")
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


# Load the model.
model = PencilNet(config)
logger.set_network(PencilNet)
print(model.summary())

# Compile the model.
opt = config['optimizer']
model.compile(optimizer=opt, loss=[multi_grid_loss],
			  metrics=config['metrics'])
logger.set_loss(multi_grid_loss)
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
