from __future__ import absolute_import, division, print_function
import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import reverb
import zlib
import os
import cv2 as cv

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import PolicySaver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from Pendulum_example.custom_ip import InvertedPendulumEnv3DNoise

num_iterations = 200000 # @param {type:"integer"}
initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}
batch_size = 128  # @param {type:"integer"}
learning_rate = 1e-4  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}
num_eval_episodes = 3  # @param {type:"integer"}
num_data_collection_episodes = 20  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
eval_time_threshold = 1000
fc_layer_params = (512, 512, 256, 256) #NN layer sizes


directory_path = os.getcwd()
data_dir = directory_path+'/data/test_data/'
valid_dir = directory_path+'/data/validation_data/'

gym_env = InvertedPendulumEnv3DNoise()
env = suite_gym.wrap_env(gym_env)
train_py_env = suite_gym.wrap_env(gym_env)
eval_py_env = suite_gym.wrap_env(gym_env)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

data_size = env.time_step_spec().observation.shape[0] + 1

print('Observation Spec:')
print(env.time_step_spec().observation)

print('Reward Spec:')
print(env.time_step_spec().reward)

print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
action = env.action_space.sample()
print('Action:')
print(action)










def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))


def create_model(env):
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    return q_net


def create_agent(train_env, q_net):
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0))
    agent.initialize()
    return agent


def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0
  for i in range(num_episodes):
    time_step = environment.reset()
    episode_return = 0.0
    for _ in range(eval_time_threshold):
        if time_step.is_last():
            break
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
    total_return += episode_return
  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a floast_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_images(prev_state, state, prev_img, img):
    img_arr = np.asarray(img, dtype=np.float16)
    prev_img_arr = np.asarray(prev_img, dtype=np.float16)
    state_arr = np.asarray(state, dtype=np.float64).reshape(-1,data_size)
    prev_state_arr = np.asarray(prev_state, dtype=np.float64).reshape(-1,data_size)
    data = {
        "img_height" : _int64_feature(img_arr.shape[-3]),
        "img_width" : _int64_feature(img_arr.shape[-2]),
        "img_depth" : _int64_feature(img_arr.shape[-1]),
        "raw_image" : _bytes_feature(serialize_array(img_arr)),
        "prev_raw_image" : _bytes_feature(serialize_array(prev_img_arr)),
        "state_size" : _int64_feature(state_arr.shape[-1]),
        "state": _bytes_feature(serialize_array(state_arr)),
        "prev_state_size" : _int64_feature(prev_state_arr.shape[-1]),
        "prev_state": _bytes_feature(serialize_array(prev_state_arr)),
    }
    return data


def collect_data(environment, policy, num_episodes=10, starting_shard=1):
    for i in range(num_episodes):
        if i < num_episodes/10:
            current_shard_name = "{}{}_{}{}.tfrecords".format(valid_dir, i+starting_shard, num_episodes, 'pendulum')
        else:
            current_shard_name = "{}{}_{}{}.tfrecords".format(data_dir, i+starting_shard, num_episodes, 'pendulum')
        time_step = environment.reset()
        file_writer = tf.io.TFRecordWriter(current_shard_name)
        prev_obs = time_step.observation.numpy()
        prev_action_step = policy.action(time_step)
        episode_return = 0

        for j in range(500):
            if not time_step.is_last():
                prev_action=prev_action_step.action.numpy()
                action_noise = np.random.normal(0,5)
                new_action = np.clip(prev_action+action_noise, 0, 20).astype(int)
                prev_action = new_action if np.random.uniform(0,1) >= 0.8 else prev_action
                print(prev_action)

                time_step = environment.step(prev_action)
                obs=time_step.observation.numpy()
                action_step = policy.action(time_step)
                action=action_step.action.numpy()
                raw=environment.render(mode='rgb_array').numpy()
                img = raw.reshape(500,500,3)[:,:,:]

                # speckle noise
                gauss = np.random.normal(0,1,img.size)
                gauss = gauss.reshape(img.shape[0],img.shape[1], img.shape[2]).astype('uint8')
                noise1 = img + img * gauss
                # cv.imwrite('noise1.png', noise1)

                # speckle subtractive noise
                gauss = np.random.normal(0,1,img.size)
                gauss = gauss.reshape(img.shape[0],img.shape[1], img.shape[2]).astype('uint8')
                noise2 = cv.add(img,gauss).reshape(500,500,3)
                # cv.imwrite('noise2.png', noise2)

                # Gaussian Blur
                noise3 = cv.GaussianBlur(img.reshape(500,500,3), (7,7), 0).reshape(500,500,3)
                # cv.imwrite('noise3.png', noise3)

                # speckle black noise
                gauss = np.random.normal(0,1,img[:,:,1].size)
                gauss = gauss.reshape(img.shape[0],img.shape[1]).astype('uint8')
                gauss = np.concatenate((gauss.reshape(500,500,1), gauss.reshape(500,500,1), gauss.reshape(500,500,1)), axis=2)
                noise4 = img + img * gauss
                # cv.imwrite('noise4.png', noise4)

                # cv.imwrite('true.png', img)
                # cv.imshow("full_img", noise3)
                # cv.waitKey(1)

                if j == 0:
                    prev_img = img
                else:

                    prev_state = np.concatenate((prev_obs, prev_action.reshape(1,1)), axis=1)
                    state = np.concatenate((obs, action.reshape(1,1)), axis=1)

                    data = parse_images(prev_state, state, prev_img.reshape(1,230,500,3), img.reshape(1,230,500,3))

                    record_bytes = tf.train.Example(features=tf.train.Features(feature=data)).SerializeToString()
                    file_writer.write(record_bytes)

                    prev_obs = obs
                    prev_action_step = action_step
                    prev_img = img
                    reward = time_step.reward
                    episode_return += 1
            else:
                break
        file_writer.close()
        print("episode return = ", episode_return)



def create_replay(agent):
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=2)

    return replay_buffer, rb_observer

def plot_data(iterations, returns):
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.show()












if __name__=='__main__':
    q_net = create_model(env)
    agent = create_agent(train_env, q_net)
    replay_buffer, rb_observer = create_replay(agent)

    example_environment = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))
    time_step = example_environment.reset()
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())

    py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps).run(train_py_env.reset())

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    best_return = avg_return

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    eval_num=1

    collect_data(eval_env, agent.policy, num_data_collection_episodes, eval_num)

    for _ in range(num_iterations):

        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            collect_data(eval_env, agent.policy, num_data_collection_episodes, eval_num)
            eval_num +=1
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
            if avg_return >= best_return:
                best_return = avg_return
                PolicySaver(agent.policy).save(data_dir)

    iterations = range(0, num_iterations + 1, eval_interval)
    plot_data(iterations, returns)