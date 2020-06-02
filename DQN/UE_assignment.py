import copy
import csv
import time

import numpy as np
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

from availability.Availability_Rayleigh import Availability_Rayleigh
from availability.Parameter import Parameter


class Location_DQN:
    def __init__(self, *args, **kwargs):
        self.location_net = self.build_net()
        self.location_target = self.build_net()

    def build_net(self):
        parameter = Parameter()
        state_num = parameter.bs_num * parameter.subcarrier_num
        output_num = state_num
        self.parameter = parameter
        self.learning_rate = 0.001
        model = Sequential()
        model.add(Dense(64,input_dim=state_num*parameter.ue_num,kernel_initializer='random_uniform'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(64,kernel_initializer='random_uniform'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(64,kernel_initializer='random_uniform'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(64,kernel_initializer='random_uniform'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(output_num,activation='linear',kernel_initializer='random_uniform'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate)) 
        return model
    
    def update_target_model(self):
        self.location_target.set_weights(self.location_net.get_weights())
    
    def train(self,train_x,train_y):
        train_x = self.to_one_hot(train_x)
        return self.location_net.fit(train_x,train_y,epochs=10, verbose=0)
    def predict(self,test_x):
        test_x = self.to_one_hot(test_x)
        return self.location_net.predict(test_x, verbose=0)
    def target_predict(self,test_x):
        test_x = self.to_one_hot(test_x)
        return self.location_target.predict(test_x)

    def to_one_hot(self,state):
        result = np.zeros((state.shape[0], state.shape[1] * self.parameter.ue_num))
        for i in range(state.shape[0]): 
            array = np.array([])
            for j in range(state.shape[1]):
                ue_index = state[i][j]
                array = np.concatenate((array,to_categorical(ue_index,num_classes=self.parameter.ue_num)))
            result[i] =array
        return result
    def save(self):
        self.location_net.save("DQN/save/dqn_location_allocation.h5")

class Brain:
    def __init__(self, parameter):
        parameter = parameter
        state_num = parameter.bs_num * parameter.subcarrier_num
        self.threshold = parameter.step_threshold
        self.state_num = state_num
        self.dqn = Location_DQN()
        self.current_state = np.zeros(state_num,dtype=np.int)-1
        self.step_count = 0
        self.e_greedy = 0.9
        self.e_greedy_increasement = 0.000
        self.gamma = 0.9
        self.parameter = parameter
        self.memory_size = 512 #经验回放的回放池大小
        self.memory = np.zeros((self.memory_size,state_num*2 +2))
        self.ue_array = np.array([0,1,2,3,4,5,0,1])
    
    def predict(self,state):
        values = self.dqn.predict(state.reshape(1,-1))
        return np.argmax(values)

    def clear(self):
        self.current_state = np.zeros(self.state_num,dtype=np.int)-1
        self.step_count = 0

    #TODO:中途退出机制
    def is_terminal(self):
        if(self.step_count == self.threshold):
            return True
        else:
            return False

    def get_reward(self,done,state = None):
        if(state is None):
            state = self.current_state
        availability = Availability_Rayleigh(self.parameter)
        reward = availability.get_reward(state.reshape(self.parameter.bs_num,self.parameter.subcarrier_num))
        return reward

    def is_valid_action(self,action):
        if action is None:
            return False
        else:
            ue = self.ue_array[self.step_count]
            # 已经被占用了
            if (self.current_state[action] != -1):
                return False
            bs_num = self.parameter.bs_num
            subcarrier_num = self.parameter.subcarrier_num
            bs_index = int(action / subcarrier_num)
            subcarrier_index = action - bs_index * subcarrier_num
            # 同一个BS不能有重复
            same_bs = self.current_state[bs_index * subcarrier_num:(bs_index + 1) * subcarrier_num]
            for i in range(subcarrier_num):
                if (same_bs[i] == ue):
                    return False
            same_subcarrier_array = np.linspace(subcarrier_index, subcarrier_index+(bs_num-1)*subcarrier_num, num=bs_num,dtype=int)
            same_subcarrier = self.current_state[same_subcarrier_array]
            # 同一个Subcarrier不能有重复
            for i in range(bs_num):
                if (same_subcarrier[i] == ue):
                    return False
            return True
    def net_max_action(self):
        action = None
        values = self.dqn.predict(self.current_state.reshape(1, -1))[0]
        while(self.is_valid_action(action)==False):
            if(action is not None):
                values[action] -= 100
            action = np.argmax(values)
        return action
    def random_action(self):
        action = None
        while(self.is_valid_action(action)==False):
            bs_index = np.random.randint(0,self.parameter.bs_num)
            subcarrier_index  =np.random.randint(0,self.parameter.subcarrier_num)
            action =  bs_index*self.parameter.subcarrier_num+subcarrier_index
        return action
    def choose_action(self):
        if(np.random.uniform()<self.e_greedy):
            action = self.net_max_action()
        else:
            action  = self.random_action()
        return action
    def step(self,action):
        state = copy.deepcopy(self.current_state.reshape(-1))
        ue = self.ue_array[self.step_count]
        self.current_state[action] = ue
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
        state_size = self.state_num
        index = mini_batch[:,state_size].astype(int)
        reward = mini_batch[:,state_size+1]
        trainX = mini_batch[:,:state_size]
        batch_index = np.arange(batch_size, dtype=np.int32)
        q_eval = self.dqn.predict(trainX)
        q_next = self.dqn.target_predict(mini_batch[batch_index,-state_size:])
        
        algorithm = self.parameter.algorithm
        if(algorithm == 'DQN'):
            selected_q_next = np.max(q_next,axis=1) 
            q_eval[batch_index,index] = reward + self.gamma * selected_q_next
            self.dqn.train(trainX,q_eval)
        elif(algorithm == 'Double_DQN'):
            max_act4next = np.argmax(self.dqn.predict(mini_batch[batch_index,-state_size:]), axis=1) # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next
            q_eval[batch_index,index] = reward + self.gamma * selected_q_next
            self.dqn.train(trainX,q_eval)
        elif(algorithm == 'Dueling_DQN'):
            pass
        elif(algorithm =='Prioritized_DQN'):
            pass

        



def run(parameter,episode):
    brain = Brain(parameter)
    episode = episode
    total_step = 0
    reward_count = 0
    reward_list = []
    for i in range(episode):
        assignment_count=len(brain.ue_array)
        for step_index in range(assignment_count):
            action  = brain.choose_action()
            current_state,action,reward,next_state,done = brain.step(action)
            brain.store(current_state,action,reward,next_state)
            total_step = total_step +1 
            if(total_step>200 and total_step%8 == 0):
                brain.learn()
            if(total_step%64 == 0):
                brain.dqn.update_target_model()
            if done:
                reward = brain.get_reward(done,next_state)
                if(reward==10):
                    reward_count = reward_count+1
                print("episode: {}/{}, reward: {} epsilon {}".format(i, episode, reward, brain.e_greedy))
                print('reward_count:',reward_count)
                reward_list.append(reward)
                brain.clear()
                break
    with open('DQN/save/allocation_output.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(reward_list)
if __name__ == '__main__':
    begin = time.time()
    print("----------begin------------")
    repeat_time = 1
    parameter = Parameter(scenario='scenario0')
    for i in range(repeat_time):
        run(parameter,episode=1000)
    print("----------end------------")
    time_elapsed = time.time() - begin
    print('The function run {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
