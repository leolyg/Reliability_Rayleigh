from Resource_Allocation.Parameter import Parameter
import Resource_Allocation.Util as util
from Resource_Allocation.Availability_Rayleigh import Availability_Rayleigh
from Resource_Allocation.algorithms.PrioritizedDQN_tf1 import DQNPrioritizedReplay
from Resource_Allocation.algorithms.DQN_tf1 import DeepQNetwork
import math
import numpy as np
import copy
import matplotlib.pyplot as plt


class Environment:
    def __init__(self,parameter,board=np.array([])):
        self.bs_num = parameter.bs_num
        self.subcarrier_num = parameter.subcarrier_num
        if(board.shape[0]==0):
            sequence = util.generate_a_sequence(parameter.ue_num, parameter.step_threshold)
            self.board = util.generate_subcarrier_allocation_scheme(sequence, parameter.bs_num, parameter.subcarrier_num,parameter.step_threshold)
        else:
            self.board = board
        self.locations = util.get_location(self.board)
        self.power_level = parameter.power_level
        self.step_threshold = parameter.step_threshold
        self.action_list = np.zeros(self.step_threshold)-1
        self.bs_locations = util.generate_bs_locations(parameter.bs_num)
        self.ue_locations = util.generate_ue_locations(parameter.ue_num)
        self.parameter = parameter
        self.reset()

    def reset(self):
        self.power_board = np.zeros((self.bs_num, self.subcarrier_num), dtype=np.int)
        self.current_step = 0
        self.action_list = np.zeros(self.step_threshold, dtype=np.int)-1
        observation = np.append(self.action_list, 0)
        return observation


    def step(self,action):
        row,column = self.locations[self.current_step%self.step_threshold]
        self.action_list[self.current_step % self.step_threshold] = action
        power_levels = range(1,self.power_level+1)
        self.power_board[row][column] = power_levels[action]
        self.current_step += 1
        observation = np.append(self.action_list,self.current_step % self.step_threshold)

        if self.current_step >= self.step_threshold:
            reward = self.get_certain_reward(self.power_board)
            reliability = 0
            #reliability = self.get_reliability()
            #reward = self.get_reward(reliability)
            done = False
            if reward > 0:
                done = True
                reward = float(reward/self.current_step)
            return observation, reward, reliability, done
        else:
            return observation, 0, 0, False

    def get_reliability(self):
        availability = Availability_Rayleigh(self.parameter, self.bs_locations, self.ue_locations, self.power_board)
        average = availability.calculate_average_availibility(self.board)
        return average

    def get_certain_reward(self,power_board):
        reward = 100
        goal = [1,2,3,4]
        power_list = []
        for i in range(power_board.shape[0]):
            for j in range(power_board.shape[1]):
                if(power_board[i][j]!=0):
                    power_list.append(power_board[i][j])
        for index in range(4):
            if(power_list[index]!=goal[index]):
                return 0
        return reward/self.current_step

    def get_reward(self,average):
        target_nine_number = 2
            ##计算当前是几个9
        count = 0
        temp = average
        while True:
            temp = temp * 10
            if (math.floor(temp) == 9):
                count = count + 1
            else:
                break
            temp = temp - 9
        if (count == 0):
            return 0
        elif count < target_nine_number:
            return 0
        else:
            return count

def extract_features(observation,step_threshold):
    features = np.zeros(step_threshold)
    count = 0
    for index in range(observation.shape[0]):
            if(observation[index]!=0):
                features[count] = observation[index]
                count = count + 1
    return features

def run(episode,env,RL,step_threshold = 10):
    step = 0
    final_reliability = 0
    step_list = []
    for episode in range(episode):
        # initial observation
        observation = env.reset()
        for i in range(step_threshold):
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, reliability, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if step > RL.memory_size:
                RL.learn()
            # swap observation
            observation = observation_
            step += 1
            # break while loop when end of this episode
            if done:
                print('episode:',episode,'steps:',env.current_step,'epsilon:',RL.epsilon,'reward:',reward)
                #找到了
                RL.add_epsilon()
                final_reliability = reliability
                step_list.append(env.current_step)
                break
            if i==step_threshold-1:
                print('episode:',episode,'steps:',env.current_step,'epsilon:',RL.epsilon,'reward:',reward,)

    return step_list
def evaluation():
    print("final evaluation")
    locations = util.get_location(board)
    power_board = np.zeros((parameter.bs_num, parameter.subcarrier_num), dtype=np.int)
    power_levels = range(1, parameter.power_level + 1)
    for i in range(parameter.step_threshold):
        action = RL.choose_action(extract_features(power_board.reshape(-1), RL.n_features))
        row, column = locations[i]
        power_board[row][column] = power_levels[action]
    print(power_board)
    # availability = Availability_Rayleigh(parameter, environment.bs_locations, environment.ue_locations, power_board)
    # average = availability.calculate_average_availibility(board)
    # print(average)

if __name__ == '__main__':
    parameter = Parameter(
            bs_num=5,
            subcarrier_num=5,
            ue_num=2,
            step_threshold=4,
            power_level=4,  # 功率级别
            theta=1  # power_budget radio
    )
    board = np.array([0,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,1,-1,-1,-1,-1,-1,-1]).reshape(parameter.bs_num,parameter.subcarrier_num)
    environment = Environment(parameter,board)
    RL = DQNPrioritizedReplay(parameter.power_level, parameter.step_threshold+1, #提取特征之后的
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=128,
                 epochs = 40,
                 e_greedy_increment=0.001,prioritized=False)
    for index in range(1):
        step_list = run(1000,environment,RL)
        #RL.plot_cost()
        # step_list = np.array(step_list).reshape(100,10)
        # print(step_list.shape)
        # x = np.linspace(0, 100,num = 100)
        # print(x)
        # plt.plot(x,np.average(step_list,axis=1))
        # plt.show()