from __future__ import absolute_import, division, print_function
import base64
from cgi import test
# import imageio
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
# import PIL.Image
# import reverb
import zlib
import os
import cv2 as cv
import glob
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
import tensorboard
tfd = tfp.distributions
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
np.random.seed(2021)

directory_path = os.getcwd()
data_dir = directory_path+'/data/test_data/'
valid_dir = directory_path+'/data/validation_data/'


def parse_tfr_dynamics(element):
  data = {
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'prev_state_size':tf.io.FixedLenFeature([], tf.int64),
      'prev_state' : tf.io.FixedLenFeature([], tf.string),}
  content = tf.io.parse_single_example(element, data)
  state_size = content['state_size']
  prev_state_size = content['prev_state_size']
  raw_state = content['state']
  raw_prev_state = content['prev_state']
  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])
  prev_state = tf.io.parse_tensor(raw_prev_state, out_type=tf.float64)
  prev_state = tf.reshape(prev_state, shape=[prev_state_size])
  return (prev_state, state)


def get_dynamics_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_dynamics) #, num_parallel_calls=tf.data.AUTOTUNE)
    return pendulum_dataset


def parse_tfr_observation(element):
  data = {
      'img_height': tf.io.FixedLenFeature([], tf.int64),
      'img_width':tf.io.FixedLenFeature([], tf.int64),
      'img_depth':tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'prev_raw_image' : tf.io.FixedLenFeature([], tf.string),
      'state_size':tf.io.FixedLenFeature([], tf.int64),
      'state' : tf.io.FixedLenFeature([], tf.string),
      'prev_state' : tf.io.FixedLenFeature([], tf.string),}
  content = tf.io.parse_single_example(element, data)
  height = content['img_height']
  width = content['img_width']
  depth = content['img_depth']
  raw_image = content['raw_image']
  prev_raw_image = content['prev_raw_image']
  state_size = content['state_size']
  raw_state = content['state']
  image = tf.io.parse_tensor(raw_image, out_type=tf.float16)
  image = tf.reshape(image, shape=[height,width,depth])
  prev_image = tf.io.parse_tensor(prev_raw_image, out_type=tf.float16)
  prev_image = tf.reshape(prev_image, shape=[height,width,depth])

  observation = tf.concat((image, prev_image), axis=2)
  state = tf.io.parse_tensor(raw_state, out_type=tf.float64)
  state = tf.reshape(state, shape=[state_size])
  return (observation, state)


def get_observation_dataset(tfr_dir:str=data_dir, pattern:str="*pendulum.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    pendulum_dataset = tf.data.TFRecordDataset(files)
    pendulum_dataset = pendulum_dataset.map(parse_tfr_observation)
    return pendulum_dataset


def build_dynamics_model():
    NUM_TRAIN_EXAMPLES = 1311296
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    model = tf.keras.Sequential([
        tfp.layers.DenseFlipout(64, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu, input_shape=[5]),
        tfp.layers.DenseFlipout(32, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
        tfp.layers.DenseFlipout(16, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu),
        layers.Dense(5)
    ])
    model.compile(optimizer='adam', loss=[tf.keras.losses.MeanAbsoluteError()])
    model.summary()
    filepath = directory_path+'/models/dynamics_model'
    tf_callback = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch',options=None)
                , keras.callbacks.TensorBoard(log_dir='logs')]
    return model, tf_callback


def build_simple_observation_model():
    NUM_TRAIN_EXAMPLES = 2233056
    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
    input_layer = tf.keras.Input(shape=(2,500,500,3))
    encode_1 = layers.TimeDistributed(layers.Conv2D(16, kernel_size=5, strides=(3,3), padding='same', activation='relu', kernel_initializer='he_uniform'))(input_layer)
    encode_2 = layers.TimeDistributed(layers.Conv2D(16, kernel_size=4, strides=(2,1), padding='same', activation='relu', kernel_initializer='he_uniform'))(encode_1)
    encode_3 = layers.TimeDistributed(layers.Conv2D(16, kernel_size=3, strides=(2,1), padding='same', activation='relu', kernel_initializer='he_uniform'))(encode_2)
    flaten_4 = layers.TimeDistributed(layers.Flatten())(encode_3)
    layers_5 = layers.TimeDistributed(layers.Dense(2048, activation=tf.nn.relu, kernel_initializer='he_uniform'))(flaten_4)
    flaten_6 = layers.Flatten()(layers_5)
    layers_6 = tfp.layers.DenseFlipout(1024, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu)(flaten_6)
    layers_7 = tfp.layers.DenseFlipout(256, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu)(layers_6)
    layers_8 = tfp.layers.DenseFlipout(32, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.relu)(layers_7)
    output_layer = tfp.layers.DenseFlipout(5, kernel_divergence_fn=kl_divergence_function)(layers_8)
    model = Model(inputs=[input_layer], outputs=[output_layer])

    model.compile(optimizer='adam', loss=[tf.keras.losses.MeanSquaredError()])
    model.summary()
    filepath = directory_path+'/models/observation_model'
    tf_callback = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch',options=None)
                , keras.callbacks.TensorBoard(log_dir='logs')]
    return model, tf_callback






if __name__=='__main__':
    # Train dynamics model
    val_dataset = get_dynamics_dataset(tfr_dir=valid_dir)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors()).shuffle(buffer_size=100000).batch(10000)
    dyn_dataset = get_dynamics_dataset()
    dyn_dataset = dyn_dataset.apply(tf.data.experimental.ignore_errors())
    dyn_train = dyn_dataset.shuffle(buffer_size=100000).batch(32).cache().prefetch(tf.data.AUTOTUNE)
    dyn_model, tf_callback1 = build_dynamics_model()
    dyn_model.fit(dyn_train, validation_data=val_dataset, epochs=100, verbose=1, callbacks=tf_callback1)


    # Train observation model
    val_dataset = get_observation_dataset(tfr_dir=valid_dir)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors()).shuffle(buffer_size=100000).batch(10000)
    obs_dataset = get_observation_dataset()
    obs_dataset = obs_dataset.apply(tf.data.experimental.ignore_errors())
    obs_train = obs_dataset.shuffle(buffer_size=100000).batch(128).prefetch(tf.data.AUTOTUNE)
    obs_model, tf_callback2 = build_simple_observation_model()
    obs_model.fit(obs_train, validation_data=val_dataset, epochs=100, verbose=1, callbacks=tf_callback2)


