import gym
import gym_hyq
import math
import sys


def get_action_base():

    action = [-0.2, 0.6, -1.7, # LF HAA, HFE, KFE
              -0.2, 0.6, -1.7, # RF HAA, HFE, KFE
              -0.2, -0.6, 1.7, # LH HAA, HFE, KFE
              -0.2, -0.6, 1.7] # RH HAA, HFE, KFE
    return action

def get_action_knee_wave(i):

    f = 1
    r = 25
    action = get_action_base()
    action[2] += 0.2 * math.sin(2 * math.pi * f * i / r)
    action[5] += 0.2 * math.sin(2 * math.pi * f * i / r)
    action[8] += 0.2 * math.sin(2 * math.pi * f * i / r)
    action[11] += 0.2 * math.sin(2 * math.pi * f * i / r)
    return action

def run(env_name, steps):

    # Init the environment
    print('\n[HyQ Gym Test] Create env \'{}\' with {} timesteps\n'.format(env_name,steps))
    env = gym.make(env_name)
    state = env.reset()

    # Loop on time steps
    prev_i = 0
    prev_t = 0
    for i in range(steps):

        action = get_action_knee_wave(i)
        state, reward, episode_over, _ = env.step(action)

        if i%30 == 0 and i != 0:
            print("\n[HyQ Gym Test] {}/{} \tReward: {}".format(i, steps, reward))
            print("[HyQ Gym Test] Real Time: {0:.2f}".format(env._elapsed_seconds) + \
                  "\tSim Time: {0:.2f}".format(env.env.sim.sim_time) + \
                  "\tRate: {0:.2f}s".format((env.env.sim.sim_time - prev_t) / (i - prev_i)))
            print("[HyQ Gym Test] State: [" + \
                  ", ".join(["%.2f"%item for item in state.flatten().tolist()[0]]) + "]")
            print("[HyQ Gym Test] LF KFE Action: {0:.2f}".format(action[2]) + \
                  " and State: {0:.2f}".format(state[0, 7]))
            print("[HyQ Gym Test] Action space: {}".format(env.action_space))
            prev_t = env.env.sim.sim_time
            prev_i = i

    # Close all
    env.close()


if __name__ == "__main__":

    if len(sys.argv) == 2:
        run(str(sys.argv[1]), int(sys.argv[2]))
    else:
        run('hyq-v0', 500)