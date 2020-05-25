# 用于计算网络可靠性
# 父类的实现方式
# 所有的衰减模型都要继承该方法
from availability.Parameter import Parameter
from availability.Util import Util
import math

class Availability_Base():
    def __init__(self, parameter):
        self.parameter = parameter
        self.util = Util()
        self.bs_locations,self.ue_locations,self.subcarrier_powers = self.util.read_environment_data()
    
    #判断是第几种类型的available
    # type = 0 表示没有占用载波
    # type = 1 表示所有基站中只有该载波被占用 
    # type = 2 表示所有基站中有多个载波被占用
    def get_intererence_type(self,board,row,column):
        if board[row][column] == -1:
            return 0
        for index in range(len(board)):
            if(index == row):
                continue
            else:
                if board[index][column] !=-1:
                    return 2
        return 1

    #计算用户设备和基站之间的距离    
    def calculate_distance(self,basestation_index,ue_index):
        if len(self.bs_locations) ==0 or len(self.ue_locations) ==0:
            self.util.read_environment_data()
        basestation_location = self.bs_locations[basestation_index]
        ue_location = self.ue_locations[ue_index]
        distance = ((float(basestation_location[0]) - float(ue_location[0]))**2+(float(basestation_location[1]) - float(ue_location[1]))**2)**0.5
        return distance

    def calculate_lambda(self,basestation_index,subcarrier_index,ue_index):
        L = self.parameter.power_level
        band_num = self.parameter.band_num
        q = math.floor(subcarrier_index/(self.parameter.subcarrier_num/band_num))
        P_max = 0
        l_s_m = int(self.subcarrier_powers[basestation_index][subcarrier_index])
        alpha_q = self.parameter.BS_alpha[q]
        if basestation_index == 0 :
            P_max = self.parameter.BS_power[0]
        else:
            P_max = self.parameter.BS_power[1]
        C_q = (self.parameter.BS_wavelengh[q]/(4*math.pi))**2
        distance = self.calculate_distance(basestation_index,ue_index)
        d_alpha_q=distance**(-alpha_q)
        if(l_s_m == 0):
            print("got it")
        return L/(l_s_m*P_max*C_q*d_alpha_q)
