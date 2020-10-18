from Resource_Allocation.Parameter import Parameter
from Resource_Allocation.algorithms.PrioritizedDQN_tf1 import DQNPrioritizedReplay
from Resource_Allocation.algorithms.DoubleDQN_tf1 import  DoubleDQN
from Resource_Allocation.algorithms.DQN_tf1 import DeepQNetwork
from Resource_Allocation.algorithms.DuelingDQN_tf1 import DuelingDQN
import Resource_Allocation.ue_allocation as ue_allocation
import Resource_Allocation.power_allocation as power_allocation
import Resource_Allocation.Util as util
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import csv


# 用来运行
def run(parameter, episode, env, RL):
    step_list = []
    total_step = 0
    for episode in range(episode):
        step = 0
        while True:
            observation = env.reset()
            reward = -1
            for index in range(parameter.step_threshold):
                extracted_observation = env.extract_features(observation)
                action = -1
                if np.random.uniform() < RL.epsilon:
                    actions_value = RL.predict(extracted_observation)
                    while not env.is_valid_action(action):
                        action = np.argmax(actions_value)
                        actions_value[0][action] = -100
                else:
                    while not env.is_valid_action(action):
                        action = np.random.randint(0, RL.n_actions)

                # action = -1
                # while not env.is_valid_action(action):
                #     action = RL.choose_action(extracted_observation)
                # if not env.is_valid_action(action):
                #     wrong_observation = copy.copy(observation)
                #     wrong_observation[index] = action
                #     RL.store_transition(env.extract_features(observation),action,-10,env.extract_features(wrong_observation))
                # RL take action and get next observation and reward
                observation_, reward, reliability, done = env.step(action)
                extracted_observation_ = env.extract_features(observation_)
                RL.store_transition(extracted_observation, action, reward, extracted_observation_)
                if step > RL.memory_size and (step % 5 == 0):
                    RL.learn()
                observation = observation_
                step += 1
                total_step += 1
                if done:
                    RL.add_epsilon()
                    start = env.reset()
                    agent_action = []
                    for step_index in range(parameter.step_threshold):
                        action = -1
                        actions_value = RL.predict(env.extract_features(start))
                        while action in start.tolist():
                            action = np.argmax(actions_value)
                            actions_value[0][action] = -100
                        agent_action.append(action)
                        start[step_index] = action
                    agent_reliability = env.calculate_reliability(env.get_board(start))
                    agent_reward = env.calculate_reward(agent_reliability)
                    print('episode:', episode, 'steps:', step, 'epsilon:', round(RL.epsilon, 3), 'reliability:',
                          reliability, 'reward:', reward, 'agent_reliability', agent_reliability, 'agent_reward',
                          agent_reward, 'agent_action', agent_action)
                    step_list.append(step)
            if reward > 0:
                # 进入下一个episode
                break
    return step_list

def get_RL(name):
    if name== 'DQN':
        DQN = DeepQNetwork(n_actions=parameter.bs_num * parameter.subcarrier_num,
                           # n_features = 6, std min_max
                           # n_features=30, #one hot
                           n_features=16,
                           learning_rate=0.05,
                           reward_decay=0.9,
                           e_greedy=0.9,
                           replace_target_iter=128,
                           memory_size=512,
                           batch_size=32,
                           epochs=40,
                           e_greedy_increment=0.001)
        return DQN
    elif name =='PDQN':
        PDQN = DQNPrioritizedReplay(n_actions=parameter.bs_num * parameter.subcarrier_num,
                                    # n_features = 6, std min_max
                                    # n_features=30, #one hot
                                    n_features=16,
                                    learning_rate=0.05,
                                    reward_decay=0.9,
                                    e_greedy=0.9,
                                    replace_target_iter=128,
                                    memory_size=512,
                                    batch_size=32,
                                    epochs=40,
                                    e_greedy_increment=0.001)
        return PDQN
    elif name == 'DDQN':
        DDQN = DoubleDQN(n_actions=parameter.bs_num * parameter.subcarrier_num,
                         # n_features = 6, std min_max
                         # n_features=30, #one hot
                         n_features=16,
                         learning_rate=0.05,
                         reward_decay=0.9,
                         e_greedy=0.9,
                         replace_target_iter=128,
                         memory_size=512,
                         batch_size=32,
                         epochs=40,
                         e_greedy_increment=0.001)
        return DDQN
    elif name=='DulingDQN':
        Dueling = DuelingDQN(n_actions=parameter.bs_num * parameter.subcarrier_num,
                                # n_features = 6, std min_max
                                # n_features=30, #one hot
                                n_features=16,
                                learning_rate=0.05,
                                reward_decay=0.9,
                                e_greedy=0.9,
                                replace_target_iter=128,
                                memory_size=512,
                                batch_size=32,
                                epochs=40,
                                e_greedy_increment=0.001)
        return Dueling

if __name__ == '__main__':
    parameter = Parameter(
        bs_num=5,
        subcarrier_num=10,
        ue_num=4,
        step_threshold=6,
        power_level=8,  # 功率级别
    )
    ue_sequence = util.generate_ue_sequence(parameter.ue_num, parameter.step_threshold)
    power_sequence = util.generate_power_sequence(parameter.step_threshold, parameter.power_level)
    bs_locations = util.generate_bs_locations(parameter.bs_num)
    ue_locations = util.generate_ue_locations(parameter.ue_num)

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    sequence = np.array(
        np.linspace(-1, parameter.ue_num - 1, num=parameter.ue_num + 1).reshape(parameter.ue_num + 1, 1))
    one_hot_encoder.fit(sequence)

    scaler_X = np.zeros((parameter.ue_num+1,parameter.step_threshold))-1
    for i in range(parameter.ue_num+1):
        scaler_X[i] = scaler_X[i] + i
    scaler_encoder = preprocessing.StandardScaler().fit(scaler_X)

    min_max_encoder = preprocessing.MinMaxScaler().fit(scaler_X)

    total_list = []
    episode = 1000
    repeat = 1
    for i in range(repeat):
        algorithms = ['DQN']
        environment = ue_allocation.Environment(parameter, ue_sequence, power_sequence, bs_locations, ue_locations,
                                                encoder=None,
                                                target=10)
        RL = DeepQNetwork(n_actions=parameter.bs_num * parameter.subcarrier_num,
                           # n_features = 6, std min_max
                           # n_features=30, #one hot
                           n_features=16,
                           learning_rate=0.05,
                           reward_decay=0.9,
                           e_greedy=0.9,
                           replace_target_iter=128,
                           memory_size=512,
                           batch_size=32,
                           epochs=40,
                           e_greedy_increment=0.001)
        step_list = run(parameter, episode, environment, RL)
        #tf.reset_default_graph()
        total_list.append(step_list)
        with open('csvdir.csv', 'a', newline='')as f:
            ff = csv.writer(f)
            ff.writerow(step_list)
        # for algorithm_index in range(len(algorithms)):
        #     RL = get_RL(algorithms[algorithm_index])
        #     step_list = run(parameter, episode, environment, RL)
        #     #tf.reset_default_graph()
        #     total_list.append(step_list)
        #     with open('csvdir.csv', 'a', newline='')as f:
        #         ff = csv.writer(f)
        #         ff.writerow(step_list)
    for index in range(len(total_list)):
        step_list = np.array(total_list[index])
        smooth = 10
        points = int( step_list.shape[0]/ smooth)
        x = np.linspace(1, points, num=points)
        step_list = np.array(step_list).reshape(points, smooth)
        plt.plot(x, np.average(step_list, axis=1))
    plt.show()
