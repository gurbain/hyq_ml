import gym
import gym_hyq
import time
import math
import sys


def get_action_base(t):

    action = []
    for i in range(8):
        action.append(0)

    return action


def run(env_name, steps):

    # Init the environment
    print('Env \'{}\' with {} timesteps'.format(env_name,steps))
    env = gym.make(env_name)
    time.sleep(5)
    state = env.reset()

    # Loop on time steps
    for i in range(steps):

        action = get_action_base(float(i))
        state, reward, episode_over, _ = env.step(action)

        if i%100 == 0:
            print('\n\n')
            print("[HyQ Gym] {}/{} \tReward: ".format(i, steps, reward))
            print("[HyQ Gym] State:")
            print(state)
            print("[HyQ Gym] Action space:")
            print(env.action_space)

    # Close all
    env.close()


if __name__ == "__main__":

    run(str(sys.argv[1]), int(sys.argv[2]))