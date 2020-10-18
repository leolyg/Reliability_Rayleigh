
#/usr/bin/python3
import csv 
import numpy as np
from availability.Parameter import Parameter

np.random.seed(1)

#本文件用来生成数据和提供一些随机方法
def get_location(board):
    locations = []
    for row in range(board.shape[0]):
        for column in range(board.shape[1]):
            if(board[row][column]!=-1):
                locations.append((row,column))
    return locations

def generate_bs_locations(bs_num):
    bs_locations = np.random.rand(bs_num, 2) * 1000
    bs_locations[0] = np.array([500,500])
    return bs_locations

def generate_ue_locations(ue_num):
    ue_locations = np.random.rand(ue_num, 2) * 1000
    return ue_locations

##随机生成用户分配的序列
def generate_ue_sequence(ue_num, step_threshold):
    sequence = []
    for i in range(step_threshold):
        if (i < ue_num):
            sequence.append(i)
        else:
            ue_index = np.random.randint(0, ue_num)
            sequence.append(ue_index)
    return sequence

def generate_power_sequence(step_threshold,power_level):
    return np.random.randint(1, power_level,size=step_threshold)


def is_valid_action(board, random_bs, random_subcarrier_num, random_ue):
    if random_bs == -1:
        return False
    if board[random_bs][random_subcarrier_num] != -1:
        return False
    for subcarrier_index in range(board.shape[1]):
        if board[random_bs][subcarrier_index] == random_ue:
            return False
    for bs_index in range(board.shape[0]):
        if board[bs_index][random_subcarrier_num] == random_ue:
            return False
    return True

def generate_subcarrier_allocation_scheme(sequence, bs_num, subcarrier_num, step_threshold):
    board = np.zeros((bs_num, subcarrier_num), dtype=np.int) - 1
    step = 0
    while step < step_threshold:
        random_bs = -1
        random_subcarrier_num = -1
        ue_index = -1
        while is_valid_action(board, random_bs, random_subcarrier_num, ue_index) == False:
            random_bs = np.random.randint(bs_num)
            random_subcarrier_num = np.random.randint(subcarrier_num)
            ue_index = sequence[step]
        board[random_bs][random_subcarrier_num] = ue_index
        step = step + 1
    return board


def generate_power_allocation_scheme(board, power_level):
    bs_num = board.shape[0]
    subcarrier_num = board.shape[1]
    power_board = np.zeros((bs_num, subcarrier_num), dtype=np.int)
    for row in range(bs_num):
        for column in range(subcarrier_num):
            if (board[row][column] != -1):
                power_board[row][column] = np.random.randint(1, power_level + 1)
    return power_board

def random_power_allocation(parameter):
    sequence = generate_ue_sequence(parameter.ue_num, parameter.step_threshold)
    board = generate_subcarrier_allocation_scheme(sequence, parameter.bs_num, parameter.subcarrier_num,parameter.step_threshold)
    power_board = generate_power_allocation_scheme(board, parameter.power_level)
    return power_board

## 生成随机的位置信息
def generate_enviroment_data(parameter):
    bs_num = parameter.bs_num
    ue_num = parameter.ue_num
    bs_locations = generate_ue_locations(bs_num)
    bs_file_name = 'data/bs_location_'+parameter.ue_num+'_'+parameter.subcarrier_num+'_'+parameter.ue_num+'_'+parameter.step_threshold +'.csv'
    ue_file_name = 'data/ue_location_'+parameter.ue_num+'_'+parameter.subcarrier_num+'_'+parameter.ue_num+'_'+parameter.step_threshold + '.csv'
    power_file_name = 'data/power_sequence_'+parameter.ue_num+'_'+parameter.subcarrier_num+'_'+parameter.ue_num+'_'+parameter.step_threshold +'.csv'
    #生成BS位置信息
    with open(bs_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in bs_locations:
            writer.writerow(row)
    #生成UE位置信息
    us_locations = generate_ue_locations(ue_num)
    with open(ue_file_name,'w',newline='')as f:
        writer = csv.writer(f)
        for row in us_locations:
            writer.writerow(row)
    #生成Subcarrier的功率级别序列
    subcarrier_power =np.random.randint(low = 1,high=parameter.power_level,size=parameter.step_threshold)
    with open(power_file_name,'w',newline='')as f:
        writer = csv.writer(f)
        writer.writerow(subcarrier_power)


## 生成功率等级
def generate_power_level(parameter):
    n = int(parameter.power_level*parameter.theta)
    total_power_array = []
    while len(total_power_array) < parameter.bs_num:
        sample = parameter.subcarrier_num
        slice_array = []
        power_array = []
        while len(slice_array) <sample-1:
            random_int = np.random.randint(n)
            if random_int not in slice_array and random_int !=0:
                slice_array.append(random_int)
        slice_array.sort()
        for index in range(len(slice_array)):
            if index ==0:
                power_array.append(slice_array[index])
            else:
                power_array.append(slice_array[index] - slice_array[index-1])
        power_array.append(n - slice_array[sample-2])
        #重新取一次排列
        np.random.shuffle(power_array)
        total_power_array.append(power_array)
    return total_power_array

## 读取环境信息
def read_environment_data(parameter):
    bs_locations = []
    ue_loactions = []
    subcarrier_powers = []
    bs_file_name = 'data/bs_location_' + parameter.ue_num + '_' + parameter.subcarrier_num + '_' + parameter.ue_num + '_' + parameter.step_threshold + '.csv'
    ue_file_name = 'data/ue_location_' + parameter.ue_num + '_' + parameter.subcarrier_num + '_' + parameter.ue_num + '_' + parameter.step_threshold + '.csv'
    power_file_name = 'data/power_sequence_' + parameter.ue_num + '_' + parameter.subcarrier_num + '_' + parameter.ue_num + '_' + parameter.step_threshold + '.csv'

    with open(bs_file_name) as bs_file:
        csv_reader = csv.reader(bs_file)
        for row in csv_reader:
            bs_locations.append(row)
    with open(ue_file_name)as ue_file:
        csv_reader = csv.reader(ue_file)
        for row in csv_reader:
            ue_loactions.append(row)
    with open(power_file_name)as power_file_name:
        csv_reader = csv.reader(power_file_name)
        for row in csv_reader:
            subcarrier_powers.append(row)
    return bs_locations,ue_loactions,subcarrier_powers



