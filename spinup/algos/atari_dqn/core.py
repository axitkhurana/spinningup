import numpy as np
from scipy import ndimage
import tensorflow as tf
from gym.spaces import Box, Discrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder_shape(shape=None):
    comb_shape = combined_shape(None, shape)
    return tf.placeholder(dtype=tf.float32, shape=comb_shape)


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def atari_nn(x, act_dim):
    x1 = tf.layers.conv2d(x, filters=16, kernel_size=(8, 8), strides=(4, 4), activation=tf.nn.relu)
    x2 = tf.layers.conv2d(x1, filters=32, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.relu)
    shape = int(np.prod(x2.get_shape()[1:]))
    x2_flat = tf.reshape(x2, [-1,  shape])
    x3 = tf.layers.dense(x2_flat, units=256, activation=tf.nn.relu)
    return tf.layers.dense(x3, units=act_dim)


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape[0])
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def _rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def process_input(obs):
    # convert to grayscale
    gray_obs = _rgb2gray(obs)
    # down sample to 110x84
    smaller = ndimage.zoom(gray_obs, 0.525)
    # crop to playing area 84x84
    crop = smaller[16:-10, :]
    return crop


def reset(env, skip_steps):
    obs = []
    o = env.reset()
    o = process_input(o)
    for _ in range(skip_steps):
        obs.append(o)
    return np.stack(obs, axis=-1)


def step(env, skip_steps, action):
    reward = 0
    done = False
    obs = []

    for i in range(skip_steps):
        o, r, d, _ = env.step(action)
        o = process_input(o)
        obs.append(o)
        if r > 0:
            reward += 1
        if r < 0:
            reward -= 1
        if d:
            for _ in range(skip_steps - i - 1):
                obs.append(o)
            done = True
            break

    observation = np.stack(obs, axis=-1)
    return observation, reward, done, None


"""
Actor-Critics
"""
def mlp_actor_critic(x, activation=tf.nn.relu, output_activation=None, action_space=None):
    act_dim = action_space.n
    with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
        # Q value for each action
        q = atari_nn(x, act_dim)
    return q
