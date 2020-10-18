from Resource_Allocation.Parameter import Parameter
import Resource_Allocation.Util as util
from Resource_Allocation.Availability_Rayleigh import Availability_Rayleigh
from Resource_Allocation.algorithms.PrioritizedDQN_tf1 import DQNPrioritizedReplay
import Resource_Allocation.AutoEncoder as AE
import math
from sklearn.preprocessing import OneHotEncoder
import copy
from keras.utils import to_categorical

# 1. 生成随机功率序列
# 2. 初始化队列
# 3 使用DQN不断生成位置的队列（action数组）
# 4 队列满了以后开始计算可靠性
# 5 如果没有找到则不断进队出队生成不同的分配方案
# 6 收敛到一定步骤之后结束


import numpy as np


class Environment:
    def __init__(self, parameter, ue_sequence, power_sequence, bs_locations, ue_locations, encoder, target):
        self.parameter = parameter
        self.ue_sequence = ue_sequence
        self.power_sequence = power_sequence
        self.bs_locations = bs_locations
        self.ue_locations = ue_locations
        self.current_step = 0
        self.action_list = np.zeros(self.parameter.step_threshold) - 1
        self.encoder = encoder #None表示autoencoder
        self.decode_model = AE.load_encode_model(parameter,16)
        self.target = target

    def reset(self):
        self.current_step = 0
        self.action_list = np.zeros(self.parameter.step_threshold) - 1
        return self.action_list

    # 如果出现不合格的action就让RL重新生成一个
    def is_valid_action(self, action):
        if action == -1:
            return False
        result = True
        for i in range(len(self.action_list)):
            if self.action_list[i] == action:
                return False
        return result

    def step(self, action):
        self.action_list[self.current_step] = action
        self.current_step = self.current_step + 1
        if self.current_step < self.parameter.step_threshold:
            observation = copy.copy(self.action_list)
            return observation, 0, 0, False
        else:
            observation = copy.copy(self.action_list)
            reliability = self.calculate_reliability(self.get_board())
            reward = self.calculate_reward(reliability)
            done = False
            if reward > 0:
                done = True
            return observation, reward, reliability, done

    def get_board(self, action_list = np.array([])):
        board = np.zeros((self.parameter.bs_num, self.parameter.subcarrier_num)).astype(int) - 1
        for index in range(len(self.action_list)):
            ue = self.ue_sequence[index]
            if action_list.shape[0] == 0:
                action = self.action_list[index]
            else:
                action = action_list[index]
            row = int(action / self.parameter.subcarrier_num)
            column = int(action - row * self.parameter.subcarrier_num)
            board[row][column] = ue
        return board

    def get_power_board(self):
        power_board = np.zeros((self.parameter.bs_num, self.parameter.subcarrier_num))
        for index in range(self.parameter.step_threshold):
            power = self.power_sequence[index]
            action = self.action_list[index]
            row = int(action / self.parameter.subcarrier_num)
            column = int(action - row * self.parameter.subcarrier_num)
            power_board[row][column] = power
        return power_board

    def calculate_reliability(self, board):
        power_board = self.get_power_board()
        availability = Availability_Rayleigh(self.parameter, self.bs_locations, self.ue_locations)
        reliability = availability.calculate_average_availibility(board, power_board)
        return reliability

    def calculate_reward(self, reliability):
        ##计算当前是几个9
        count = 0
        temp = reliability
        while True:
            temp = temp * 10
            if (math.floor(temp) == 9):
                count = count + 1
            else:
                break
            temp = temp - 9
        if (count < self.target):
            return -1
        else:
            return 10 ** (count + 1 - self.target)

    def extract_features(self, observation,):
        if self.encoder!=None:
            # one_hot编码或者standard scaler或者min_max
            X = observation.reshape(1,-1)
            extracted_observation = self.encoder.transform(X)
        else:
            # 自编码器
            decode_model = self.decode_model
            extracted_observation = decode_model.predict(observation.reshape(1,-1))
        return extracted_observation.reshape(-1)

    def one_hot(self, observation,categories):
        return to_categorical(observation,categories).reshape(-1)



