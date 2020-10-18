# 用于计算网络可靠性
# 使用瑞利分布的实现方式
import math

class Availability_Rayleigh:

    def __init__(self, parameter,bs_location,ue_location):
        self.parameter = parameter
        self.bs_locations = bs_location
        self.ue_locations = ue_location

    # 传入修改后的功率方案
    def change_subcarrier_powers(self,subcarrier_powers):
        self.subcarrier_powers = subcarrier_powers

    # 判断是第几种类型的available
    # type = 0 表示没有占用载波
    # type = 1 表示所有基站中只有该载波被占用
    # type = 2 表示所有基站中有多个载波被占用
    def get_intererence_type(self, board, row, column):
        if board[row][column] == -1:
            return 0
        for index in range(len(board)):
            if (index == row):
                continue
            else:
                if board[index][column] != -1:
                    return 2
        return 1

    # 计算用户设备和基站之间的距离
    def calculate_distance(self, basestation_index, ue_index):
        if len(self.bs_locations) == 0 or len(self.ue_locations) == 0:
            self.util.read_environment_data()
        basestation_location = self.bs_locations[basestation_index]
        ue_location = self.ue_locations[ue_index]
        distance = ((float(basestation_location[0]) - float(ue_location[0])) ** 2 + (
                    float(basestation_location[1]) - float(ue_location[1])) ** 2) ** 0.5
        return distance

    def calculate_lambda(self, basestation_index, subcarrier_index, ue_index):
        L = self.parameter.power_level
        band_num = self.parameter.band_num
        q = math.floor(subcarrier_index / (self.parameter.subcarrier_num / band_num))
        P_max = 0
        l_s_m = int(self.subcarrier_powers[basestation_index][subcarrier_index])
        alpha_q = self.parameter.BS_alpha[q]
        if basestation_index == 0:
            P_max = self.parameter.BS_power[0]
        else:
            P_max = self.parameter.BS_power[1]
        C_q = (self.parameter.BS_wavelengh[q] / (4 * math.pi)) ** 2
        distance = self.calculate_distance(basestation_index, ue_index)
        d_alpha_q = distance ** (-alpha_q)
        if (l_s_m == 0):
            print("got it")
        return L / (l_s_m * P_max * C_q * d_alpha_q)

    #计算单个连接的可靠性
    def calculate_availability(self,board,basestation_index,subcarrier_index):
        availability_type = self.get_intererence_type(board,basestation_index,subcarrier_index)
        tau = self.parameter.tau
        N_0 = self.parameter.N_0
        lambda_s = self.calculate_lambda(basestation_index,subcarrier_index,board[basestation_index][subcarrier_index])
        if availability_type == 0:
            return 0
        elif availability_type == 1:
            return math.exp(-lambda_s*tau*N_0)
        elif availability_type ==2:
            Pi = 1
            Sigma= 0 
            #计算公式中连乘的lambda
            for i in range(len(board)):
                if board[i][subcarrier_index] != -1:
                    Pi = Pi * self.calculate_lambda(i,subcarrier_index,board[i][subcarrier_index])
            for j in range(len(board)):
                if board[j][subcarrier_index] !=-1 and j != basestation_index:
                    lambda_j = self.calculate_lambda(j,subcarrier_index,board[j][subcarrier_index])
                    numerator = math.exp(-N_0*lambda_s*tau)
                    pi_difference = 1 
                    for  k in range(len(board)) :
                        if k != j and k != basestation_index and board[k][subcarrier_index]!=-1:
                            pi_difference = pi_difference * (self.calculate_lambda(k,subcarrier_index,board[k][subcarrier_index]) - lambda_j )
                    denominator = lambda_s*(lambda_j + lambda_s * tau) * pi_difference
                    Sigma = Sigma + (numerator/denominator) 
            return Pi * Sigma

    def calculate_average_availibility(self, board, detail=False):
        return self.calculate_average_availibility(board,self.subcarrier_powers)

    #用来计算整体的平均可靠性
    def calculate_average_availibility(self, board, sub_carrier_powers, detail=False):
        self.change_subcarrier_powers(sub_carrier_powers)
        n = self.parameter.ue_num
        availibility_lists = []
        #是否存在未分配的
        exist_unallocated_ue =False
        ##初始化
        for i in range(n):
            availibility_lists.append([])
        ##计算每个节点的可靠性并存储
        for row in range(len(board)):
            for column in range(len(board[row])):
                ue_index = board[row][column]
                if ue_index !=-1:
                    availability = self.calculate_availability(board,row,column)
                    availibility_lists[ue_index].append(availability)

        ##计算平均可靠性
        sum = 0
        for row in range(len(availibility_lists)):
            ##如果该UE没被分配
            if(len(availibility_lists[row])==0):
                exist_unallocated_ue = True
            pro = 1
            for column in range(len(availibility_lists[row])):
                pro = pro * (1-availibility_lists[row][column])
            sum =sum +(1-pro)
        average = sum/n 
        if(detail):
            return average,exist_unallocated_ue,availibility_lists
        else:
            return average

    def calculate_average_availibility_with_power_plan(self, allocation_plan,power_plan):
        self.change_subcarrier_powers(power_plan)
        return self.calculate_average_availibility(allocation_plan)[0]


    def get_number_of_nines(self, board):
        average, exist_unallocated_ue = self.calculate_average_availibility(board)
        if (exist_unallocated_ue):
            return 0
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
        return count


    #计算当前局势的reward（平均可靠性的函数）
    def get_reward(self,board):
        average,exist_unallocated_ue = self.calculate_average_availibility(board)
        target_nine_number = self.parameter.target_nine_number
        ##如果存在未分配的UE，直接返回一个负的reward
        if(exist_unallocated_ue):
            return -1
        ##计算当前是几个9
        count = 0 
        temp = average
        while True:
            temp = temp * 10
            if(math.floor(temp) == 9):
                count  = count + 1
            else:
                break
            temp = temp -9
        if(count == 0 ):
            return average
        elif count < target_nine_number:
            return count
        else:
            return count
