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
from timeit import default_timer as timer
from matplotlib import rcParams, rc
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

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
# from cartpole_noise import CartPoleEnvNoise
from Pendulum_example.custom_ip import InvertedPendulumEnv3DNoise
np.random.seed(2021)

directory_path = os.getcwd()

gym_env = InvertedPendulumEnv3DNoise()
env = suite_gym.wrap_env(gym_env)
eval_env = tf_py_environment.TFPyEnvironment(gym_env)



def run_DSF_filter(env, dyn_model, obs_model, saved_policy):
    num_ensembles = 100
    state_size = 5
    time_step = env.reset()
    obs=time_step.observation.numpy()
    state = np.concatenate((obs, np.array([0]).reshape(1,1)), axis=1)
    policy_state = saved_policy.get_initial_state(batch_size=1)
    prev_state = state

    dyn_pred_ensemble = np.asarray([]).reshape(0,num_ensembles,state_size)
    flt_pred_ensemble = np.asarray([]).reshape(0,num_ensembles,state_size)
    true_state = np.asarray([]).reshape(0,state_size)
    ensemble_filtered = np.asarray([]).reshape(0,state_size)

    for i in range(num_ensembles):
        ensemble_filtered = np.concatenate((ensemble_filtered, state.reshape(1,state_size)), axis=0)

    for j in range(113):
        if not time_step.is_last():
            policy_step = saved_policy.action(time_step, policy_state)
            policy_state = policy_step.state
            action = policy_step.action.numpy()

            time_step = env.step(action)
            obs=time_step.observation.numpy()
            state = np.concatenate((obs, np.array([(action-10)*.1]).reshape(1,1)), axis=1)

            env.render()

            ensemble_obs = obs_model(state, training=False).numpy().reshape(1,num_ensembles,state_size)
            cov_obs = np.cov(ensemble_obs.reshape(num_ensembles,state_size).T)
            
            ensemble_filtered[:,4]=prev_state[0,4]
            ensemble_dyn = dyn_model(ensemble_filtered, training=False).numpy().reshape(1,num_ensembles,state_size)
            cov_dyn = np.cov(ensemble_dyn.reshape(num_ensembles,state_size).T)

            S = np.add(cov_obs, cov_dyn)
            ensemble_filtered = (np.dot(ensemble_dyn.reshape(num_ensembles,state_size), np.dot(cov_obs,np.linalg.pinv(S))) 
                               + np.dot(ensemble_obs.reshape(num_ensembles,state_size), np.dot(cov_dyn,np.linalg.pinv(S))))
            flt_pred_ensemble = np.concatenate((flt_pred_ensemble, ensemble_filtered.reshape(1,num_ensembles,5)), axis=0)

            prev_state = state
            true_state = np.concatenate((true_state, state), axis=0)

        else:
            break






if __name__=='__main__':

    ctrl_model = tf.compat.v2.saved_model.load(directory_path+'/models/ctrl0')
    dyn_model = models.load_model(directory_path+'/models/dynamics_model')
    obs_model = models.load_model(directory_path+'/models/observation_model')

    run_DSF_filter(eval_env, dyn_model, obs_model, ctrl_model)
