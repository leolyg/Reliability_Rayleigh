import copy
import math
import csv
import time
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Input, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop ,adam
from sklearn import preprocessing
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.utils import to_categorical
from availability.Availability_Rayleigh import Availability_Rayleigh
from availability.Parameter import Parameter
from DQN.AutoEncoder import AutoEncoder


#分配UE
class UE_DQN:
    def __init__(self,parameter):
        self.parameter = parameter
        self.ue_net = self.build_net()
        self.ue_target = self.build_net()
    def build_net(self):
        parameter = self.parameter
        expand_state =parameter.ue_num+1
        self.expand_state = expand_state
        state_num = parameter.step_threshold*expand_state
        output_num = parameter.ue_num
        self.learning_rate = 0.001
        ue_net = Sequential()
        ue_net.add(Dense(64,input_dim=state_num,kernel_initializer='random_uniform'))
        ue_net.add(LeakyReLU(alpha=0.05))
        ue_net.add(Dense(64,kernel_initializer='random_uniform'))
        ue_net.add(LeakyReLU(alpha=0.05))
        ue_net.add(Dense(64,kernel_initializer='random_uniform'))
        ue_net.add(LeakyReLU(alpha=0.05))
        ue_net.add(Dense(64,kernel_initializer='random_uniform'))
        ue_net.add(LeakyReLU(alpha=0.05))
        ue_net.add(Dense(output_num,activation='linear',kernel_initializer='random_uniform'))
        ue_net.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return ue_net
    def update_target_model(self):
        self.ue_target.set_weights(self.ue_net.get_weights())

    def data_preprocess(self,data,method):
        if(method=='one_hot'):
            return self.to_one_hot(data)
        elif(method == 'AE'):
            ##加入之前读取AE模型的代码
            AE = AutoEncoder()
            AE.load_model()
            return AE.encoding(data)

    def train(self,train_x,train_y):
        train_x = self.data_preprocess(train_x,'one_hot')
        return self.ue_net.fit(train_x,train_y,epochs=10, verbose=0)
    def predict(self,test_x):
        test_x = self.data_preprocess(test_x,'one_hot')
        return self.ue_net.predict(test_x, verbose=0)
    def target_predict(self,test_x,):
        test_x = self.data_preprocess(test_x,'one_hot')
        return self.ue_target.predict(test_x)

    def to_one_hot(self,state):      
        result = np.zeros((state.shape[0], state.shape[1]*self.expand_state))
        for i in range(state.shape[0]): 
            array = np.array([])
            for j in range(state.shape[1]):
                ue_index = state[i][j]
                array = np.concatenate((array,to_categorical(ue_index,num_classes=self.expand_state)))
            result[i] =array
        return result

    def save(self):
        self.ue_net.save("DQN/save/dqn_ue_allocation.h5")
    

class Brain:
    def __init__(self, parameter):
        ue_dqn = UE_DQN(parameter)
        self.parameter = parameter
        self.current_state = np.zeros(parameter.step_threshold,dtype=np.int)-1
        self.step_count = 0
        self.e_greedy = 0.1
        self.e_greedy_increasement = 0.001
        self.gamma = 0.9
        self.memory_size = 512 #经验回放的回放池大小
        self.memory = np.zeros((self.memory_size,parameter.step_threshold*2 +2))
        self.ue_dqn = ue_dqn

    #训练完成后预测
    def predict(self,state):
        values = self.ue_dqn.predict(state.reshape(1,-1))
        return np.argmax(values)

    def clear(self):
        self.current_state = np.zeros(self.parameter.step_threshold,dtype=np.int)-1
        self.step_count = 0
        
    def is_terminal(self):
        if(self.step_count == self.parameter.step_threshold):
            return True
        else:
            last_action = self.current_state[self.step_count-1]
            index = 0
            count = 0
            for index in range(self.step_count):
                if(self.current_state[index] == last_action):
                    count = count+1
            if(count>=3):
                return True
            else:
                return False

    def get_reward(self,done,state = None):
        if(state is None):
            state = self.current_state
        unique_num = len(np.unique(state))
        ue_num = self.parameter.ue_num
        step_threshold = self.parameter.step_threshold
        if(done):
            if(unique_num == ue_num ):
                if(state[step_threshold-1]!=-1):
                    return 10
                else:
                    return -1
            else:
                return -1
        else:
            return 0 

    def net_max_action(self):
        values = self.ue_dqn.target_predict(self.current_state.reshape(1,-1))[0]
        action = np.argmax(values)
        return action

    def random_action(self):
        action = np.random.randint(0,self.parameter.ue_num)
        return action
        
    def choose_action(self):
        if(np.random.uniform()<self.e_greedy):
            action = self.net_max_action()
        else:
            action  = self.random_action()
        return action
    
    def step(self,action):
        state = copy.deepcopy(self.current_state.reshape(-1))
        self.current_state[self.step_count] = action
        self.step_count = self.step_count+1
        next_state = copy.deepcopy(self.current_state.reshape(-1))
        if(self.is_terminal()):
            done = True
            if(self.e_greedy<1):
                self.e_greedy = self.e_greedy+self.e_greedy_increasement
            reward = self.get_reward(done)
        else:           
            done = False
            reward = self.get_reward(done)
        return state,action,reward,next_state,done
    def store(self,state,action,reward,next_state):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter % self.memory_size
        action_array = np.array([action,reward])
        memory_item = np.concatenate((state,action_array,next_state),axis=0)
        self.memory[index] = np.hstack(memory_item)
        self.memory_counter += 1

    def learn(self,batch_size=128):
        if self.memory_counter > batch_size:
            index_range = min(self.memory_size,self.memory_counter)
            sample_index = np.random.choice(index_range, size=batch_size)
        else:
            sample_index = np.arange(self.memory_counter)
        #mini_batch data
        mini_batch = self.memory[sample_index,:]
        state_size = self.parameter.step_threshold
        ue_index = mini_batch[:,state_size].astype(int)
        reward = mini_batch[:,state_size+1]
        ue_trainX = mini_batch[:,:state_size]
        ue_predict = self.ue_dqn.predict(ue_trainX)
        batch_index = np.arange(batch_size, dtype=np.int32)
        ue_predict[batch_index,ue_index] = reward + self.gamma* np.max(self.ue_dqn.target_predict(mini_batch[batch_index,-state_size:]),axis=1) 
        self.ue_dqn.train(ue_trainX,ue_predict)

def run(parameter,episode):
    since = time.time()
    brain = Brain(parameter)
    total_step = 0
    reward_count = 0
    reward_list = []
    for i in range(episode):
        for step_index in range(parameter.step_threshold):
            action  = brain.choose_action()
            current_state,action,reward,next_state,done = brain.step(action)
            brain.store(current_state,action,reward,next_state)
            total_step = total_step +1 
            if(total_step>200 and total_step%8 == 0):
                brain.learn()
            if(total_step%64 == 0):
                brain.ue_dqn.update_target_model()
            if done:
                reward = brain.get_reward(done,next_state)
                if(reward==10):
                    reward_count = reward_count+1
                    with open('candidate_sequence.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(next_state)
                if(i%10==0):
                    print("episode: {}/{}, reward: {} epsilon {}".format(i, episode, reward, round(brain.e_greedy,3)))
                # print(brain.current_state)
                reward_list.append(reward)
                brain.clear()
                break
    print('reward_count:',reward_count)

    #brain.ue_dqn.save()
    time_elapsed = time.time() - since
    print('The function run {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return reward_list

if __name__ == '__main__':
    begin = time.time()
    print("----------begin------------")
    repeat_time = 1
    #for episode in [1000,2000,3000]:
    scenarios = [ 'scenario2']
    for episode in [3000]:
        for t in scenarios:
            all_reward = []
            parameter = Parameter(scenario=t)
            for i in range(repeat_time):
                reward = run(parameter,episode = episode)
                all_reward.append(reward)
            with open('sequence_output.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                for i in all_reward:
                    writer.writerow(i)
    print("----------end------------")
    time_elapsed = time.time() - begin
    print('The function run {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
