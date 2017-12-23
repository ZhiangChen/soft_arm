from DQN import DeepQNetwork
from env import Env
import numpy as np


EPS = 300000
STEP = 400
action_space = ['f', 'b', 'l', 'r', 'fl', 'fr', 'bl', 'br']
DIST = 0.01
R = 0.5
MEMORYCAPACITY = 400000

def compute_reward(state, state_):
    """
    if distance is decreasing, reward +1; if distance is increasing, reward -1;
    if reach the goal, reward +100; if fall into obstacle, reward -100;
    otherwise, reward 0
    """
    obs = state[0:2]
    goal = state[2:4]
    point = state[4:6]
    point_ = state_[4:6]
    dist = np.linalg.norm(point - goal)
    dist_ = np.linalg.norm(point_ - goal)
    r = np.linalg.norm(point_ - obs)
    if r < R:
        return -100.0, -1.0

    if dist_ < dist:
        if dist_ < DIST:
            return 100.0, dist_
        return 1.0, dist_

    if dist_ > dist:
        return -1.0, dist_

    return 0.0, dist_


if __name__ == "__main__":
    # maze game
    env = Env()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.0001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=1000,
                      memory_size=MEMORYCAPACITY,
                      # output_graph=True
                      )

    for episode in range(EPS):
        env.build_map()
        value = 0

        for step in range(STEP):
            state = env.state.copy()
            action = RL.choose_action(state)
            env.step(action_space[action])
            state_ = env.state.copy()
            reward, dist = compute_reward(state, state_)

            RL.store_transition(state, action, reward, state_)
            value += reward
            if dist < DIST:
                break

            if RL.memory_counter > MEMORYCAPACITY:
                RL.learn()

        if (episode+1)%100 == 0:
            print episode+1
            print value
            if dist < DIST & dist>0:
                print "Got Target"
            if dist < 0:
                print "Got Obstacle"
            if dist > DIST:
                print "Failed Target"
            print '*'*40

    RL.save_model()

    while True:
        env.reset_map()
        value = 0
        for step in range(STEP):
            state = env.state.copy()
            action = RL.choose_action(state)
            env.step(action_space[action])
            state_ = env.state.copy()
            reward, dist = compute_reward(state, state_)

            env.render()
            value += reward
            if dist < DIST:
                break

        print value
        print '\n'
