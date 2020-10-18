import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from Resource_Allocation.algorithms.DQN_tf1 import DeepQNetwork
from Resource_Allocation.algorithms.PrioritizedDQN_tf1 import DQNPrioritizedReplay
from Resource_Allocation.algorithms.QLearning import QLearningTable
import copy
import matplotlib.pyplot as plt



def build_net(input_shape,output_shape):
    input_vec = Input(shape=(input_shape,))
    layer1 = Dense(32, kernel_initializer='random_uniform', bias_initializer='zeros',activation='relu')(input_vec)
    layer2 = Dense(32, kernel_initializer='random_uniform', bias_initializer='zeros',activation='relu')(layer1)
    layer3 = Dense(output_shape, kernel_initializer='random_uniform', bias_initializer='zeros',activation='softmax')(layer2)
    model = Model(inputs=input_vec, outputs=layer3)
    model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def run_iris_classification():
    iris = load_iris()
    X = iris_data = iris['data']
    iris_target = iris['target'].reshape(iris_data.shape[0],1)
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(iris_target).toarray()
    model = build_net(4,3)
    model.fit(iris_data,Y,batch_size=8,epochs=20)
    scores = model.evaluate(X, Y, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    predict_class = np.argmax(model.predict(X), axis=1)
    # print(predict_class)
    count = 0
    for i in range(X.shape[0]):
        if(iris_target[i][0] != predict_class[i]):
            count = count +1
    print("error count %s",count)
    K.clear_session()

def test():
    #只要epochs够，就不会出现变化太小的问题
    data_X = np.array([
        [-1,-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,0,-1,-1,-1],
        [-1, -1, -1, -1, 0, 1, -1, -1],
        [-1, -1, -1, -1, 0, 1, 2, -1]
    ])
    data_Y = np.array([[0], [1], [2], [3]])
    X = to_categorical(data_X.reshape(-1),4).reshape(4,32)
    print(X)
    Y = to_categorical(data_Y)
    model = build_net(32,4)
    model.fit(X,Y,epochs=8)
    scores = model.evaluate(X, Y, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    predict = np.argmax(model.predict(X), axis=1)
    print(predict)
    K.clear_session()

def get_reward(observation,goal):
    for index in range(goal.shape[0]):
        if observation[index] != goal[index]:
            return -1
    return 100

def preprocessing(observation):
    processed_observation = np.zeros(observation.shape[0])
    for i in range(observation.shape[0]):
        if observation[i]==-1:
            processed_observation[i] = 0.2
        else:
            processed_observation[i] = round((observation[i]+1)*0.2,1)
    return processed_observation.tolist()

def onehot(observation):
    vector = np.append(action_space,-1)
    to_categorical(vector,5)
    one_hot = to_categorical(observation,5)
    return one_hot.reshape(-1)

def run_Q_learning(episode,Q_table):
    total_step = 0
    step_list  =[]
    for e in range(episode):
        step = 0
        while True:
            reward = None
            observation = copy.copy(init_observation)
            for i in range(step_threshold):
                previous_observation = str(preprocessing(copy.copy(observation)))
                action = Q_table.choose_action(previous_observation)
                observation[i] = action
                next_observation = str(preprocessing(copy.copy(observation)))
                if i == (step_threshold-1):
                    reward = get_reward(observation,goal)
                    if(reward>0):
                        next_observation = 'terminal'
                else:
                    reward = 0
                Q_table.learn(previous_observation, action, reward, next_observation)
                step += 1
                total_step += 1
            if reward > 0:
                predicted_observation = copy.copy(init_observation)
                for i in range(step_threshold):
                    action = Q_table.choose_predict_action(str(preprocessing(predicted_observation)))
                    predicted_observation[i] = action
                step_list.append(step)
                print("episode:", e, 'steps:', step, 'reward:', reward, 'predicted_observation:', predicted_observation)
                break
    return step_list

def run(episode,RL):
    #神经元为 [16, 16, 16]时比较好
    total_step = 0
    step_list  =[]
    for e in range(episode):
        step = 0
        while True:
            reward = None
            observation = copy.copy(init_observation)
            for i in range(step_threshold):
                previous_observation = np.array(onehot(copy.copy(observation)))
                action_index = RL.choose_action(previous_observation)
                observation[step % action_space.shape[0]] = action_space[action_index]
                next_observation = np.array(onehot(copy.copy(observation)))
                if i == (step_threshold - 1):
                    reward = get_reward(observation, goal)
                else:
                    reward = 0
                RL.store_transition(previous_observation, action_index, reward, next_observation)
                if total_step > RL.memory_size and total_step % 5 == 0:
                    RL.learn()
                step += 1
                total_step += 1
            if reward > 0:
                for index in range(4):
                    RL.learn()
                memory = RL.memory
                predicted_observation = copy.copy(init_observation)
                for i in range(step_threshold):
                    action_index = RL.choose_predict_action(onehot(predicted_observation))
                    predicted_observation[i % action_space.shape[0]] = action_space[action_index]
                RL.add_epsilon()
                step_list.append(step)
                print("episode:", e, 'steps:', step, 'epsilon:', round(RL.epsilon,4), 'reward:', reward,'predicted_observation:', predicted_observation)
                break
    return step_list


if __name__ == '__main__':
    action_features = 4
    state_features = 4
    step_threshold = 4
    episode = 1000
    init_observation = np.zeros(state_features) - 1
    goal = np.array([1,2,3,4])
    action_space = np.array([1,2,3,4])

    # for i in range(32):
    #     RL.store_transition(np.array([-1, -1, -1, -1]), 0, 0, np.array([2, -1, -1, -1]))
    #     RL.store_transition(np.array([-1, -1, -1, -1]), 0, 0, np.array([3, -1, -1, -1]))
    #     RL.store_transition(np.array([-1, -1, -1, -1]), 0, 0, np.array([4, -1, -1, -1]))
    #     RL.store_transition(np.array([-1,-1,-1,-1]), 0, 0,np.array([1,-1,-1,-1]))
    #     RL.store_transition(np.array([1, -1, -1, -1]), 1, 0, np.array([1, 2, -1, -1]))
    #     RL.store_transition(np.array([1, 2, -1, -1]), 2, 0, np.array([1, 2, 3, -1]))
    #     RL.store_transition(np.array([1, 2, 3, -1]), 3, 1, np.array([1, 2, 3, 4]))
    # for i in range(4):
    #     RL.learn()
    # 使用Q-learning
    Q_table = QLearningTable(actions=list(action_space.tolist()))
    #step_list = run_Q_learning(episode, Q_table)

    DQN = DeepQNetwork(action_features, (action_features+1)*state_features,  # 提取特征之后的
                              learning_rate=0.01,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=128,
                              memory_size=256,
                              batch_size=32,
                              epochs=32,
                              e_greedy_increment=0.005,)
    ##DeepQNetwork
    PDQN = DQNPrioritizedReplay(action_features, (action_features+1)*state_features,  # 提取特征之后的
                              learning_rate=0.01,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=128,
                              memory_size=256,
                              batch_size=32,
                              epochs=32,
                              e_greedy_increment=0.005, prioritized=False)

    smooth = 10
    points = int(episode / smooth)
    x = np.linspace(1, points, num=points)
    total_step = []
    algorithms = [PDQN]
    for i in range(len(algorithms)):
        step_list = run(episode, algorithms[i])
        step_list = np.array(step_list)
        total_step.append(step_list)
        plt.plot(x, np.average(np.array(step_list).reshape(points, smooth), axis=1))
    plt.show()


