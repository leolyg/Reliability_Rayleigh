
import numpy as np 
import random

def is_valid_action(board,random_bs,random_subcarrier_num,random_ue):
    if random_bs == -1:
        return False
    
    if board[random_bs][random_subcarrier_num] !=-1:
        return False
    for subcarrier_index in range(board.shape[1]):
        if board[random_bs][subcarrier_index] == random_ue:
            return False

    for bs_index in range(board.shape[0]):
        if board[bs_index][random_subcarrier_num] == random_ue:
            return False
    return True

def generate_a_plan(sequence,bs_num,subcarrier_num,step_threshold):
    board =np.zeros((bs_num,subcarrier_num),dtype=np.int)-1
    step = 0 
    while step < step_threshold:
        random_bs = -1
        random_subcarrier_num = -1 
        ue_index = -1
        while is_valid_action(board,random_bs,random_subcarrier_num,ue_index) == False:
            random_bs = np.random.randint(bs_num)
            random_subcarrier_num = np.random.randint(subcarrier_num)
            ue_index = sequence[step]
        board[random_bs][random_subcarrier_num] = ue_index
        step = step+1
    return board
    
