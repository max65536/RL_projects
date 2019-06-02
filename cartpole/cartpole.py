#
# cartpole.py, exercise sheet 2, Advanced Machine Learning course, RWTH Aachen University, summer term 2019, Jonathon Luiten
#

import gym
# render a dummy environment before importing tensorflow to circumvent tensorflow/openai-gym integration bug
# g_env = gym.make('CartPole-v0')
# g_env.render()

import tensorflow as tf
import random
import numpy as np

num_training_episodes = 2000
episode_length = 200

env = gym.make('CartPole-v0')
# env.render()
monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

print(env.action_space,'    ',env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

Q=np.zeros([10**4,env.action_space.n])

alpha=0.4
gamma=0.99

def bins(left,right,num):
    return np.linspace(left,right,num+1)[1:-1]

def get_state_bins(observation):

    cart_pos, cart_v, pole_angle, pole_v = observation
    state_bins=0
    state_bins=np.digitize(cart_pos, bins=bins(-2.4, 2.4, 10))
    state_bins=state_bins*10+np.digitize(cart_v, bins=bins(-1.0, 1.0, 10))
    state_bins=state_bins*10+np.digitize(pole_angle, bins=bins(-0.42, 0.42, 10))
    state_bins=state_bins*10+np.digitize(pole_v, bins=bins(-3.5, 3.5, 10))

    # print(observation,'=',state_bins)
    return state_bins

def get_action(observation,Q,episode):
    state_next=get_state_bins(observation)
    # epsilon=1/np.sqrt(episode+1)
    epsilon=0.5 * (0.99 ** episode)
    if epsilon<=np.random.uniform(0,1):
        action_next=np.argmax(Q[state_next])
    else:
        action_next=np.random.choice([0,1])
    return action_next

def run_episode( env, sess ,episode):

    #Initialize S
    observation = env.reset()
    episode_return = 0
    state=get_state_bins(observation)

    for t in range( episode_length ):

        # random policy
        # action = 0 if random.uniform(0,1) < 0.5 else 1
        action = get_action(observation,Q,episode)

        observation_next, reward, done, info = env.step(action)
        state_next=get_state_bins(observation_next)

        if done and t<199:
            reward=-200

        episode_return += 1

        maxQ=max(Q[state_next,0],Q[state_next,1])

        # print('maxQ=',maxQ)

        Q[state,action]+=alpha*(reward+gamma*maxQ-Q[state,action])
        state=state_next
        observation=observation_next

        # disable rendering for faster training
#         env.render()

        if done:
            print("episode ended early")
            break


    print("episode return: %f"%(episode_return), end=' ')

    return episode_return


scores=np.zeros(num_training_episodes, dtype=int)

for i in range( num_training_episodes ):

    print('episode=',i,end=' ')
    episode_return = run_episode( env, sess ,i)
    scores[i]=episode_return
    if i>=100:
        average=sum(scores[i-100:i])/100
        print('average=',average)
        if average>=195:
            print('success!')

monitor.close()






