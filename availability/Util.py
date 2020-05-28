
#/usr/bin/python3
import csv 
import math
import numpy as np
from availability.Parameter import Parameter

#用来生成数据
class Util:
    def __init__(self, bs_file_name='availability/data/bs_location.csv', ue_file_name='availability/data/ue_location.csv', subcarrier_file_name='availability/data/subcarrier_power.csv'):
        self.bs_file_name = bs_file_name
        self.ue_file_name = ue_file_name
        self.subcarrier_file_name = subcarrier_file_name
        self.parameter = Parameter()
        
    ## 生成随机的位置信息
    def generate_enviroment_data(self):
        bs_num = self.parameter.bs_num
        ue_num = self.parameter.ue_num
        bs_locations = np.random.rand(bs_num-1,2)*1000
        first_row = ['500','500']
        #生成BS位置信息
        with open(self.bs_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(first_row)
            for row in bs_locations:
                writer.writerow(row)
        #生成UE位置信息
        us_loactions = np.random.rand(ue_num,2)*1000
        with open(self.ue_file_name,'w',newline='')as f:
            writer = csv.writer(f)
            for row in us_loactions:
                writer.writerow(row)
        #生成Subcarrier的功率级别
        subcarrier_power =np.random.randint(low = 1,high=self.parameter.power_level,size=(self.parameter.bs_num, self.parameter.subcarrier_num))
        #subcarrier_power = self.generate_power_level()
        with open(self.subcarrier_file_name,'w',newline='')as f:
            writer = csv.writer(f)
            for row in subcarrier_power:
                writer.writerow(row)
    ## 生成功率等级
    def generate_power_level(self):
        n = int(self.parameter.power_level*self.parameter.theta)
        total_power_array = []
        while len(total_power_array) < self.parameter.bs_num:
            sample = self.parameter.subcarrier_num
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
    def read_environment_data(self):
        bs_locations = []
        ue_loactions = []
        subcarrier_powers = []
        with open(self.bs_file_name) as bs_file:
            csv_reader = csv.reader(bs_file)
            for row in csv_reader:
                bs_locations.append(row)
        with open(self.ue_file_name)as ue_file:
            csv_reader = csv.reader(ue_file)
            for row in csv_reader:
                ue_loactions.append(row)
        with open(self.subcarrier_file_name)as subcarrier_file:
            csv_reader = csv.reader(subcarrier_file)
            for row in csv_reader:
                subcarrier_powers.append(row)

        return bs_locations,ue_loactions,subcarrier_powers


    ## 读取功率级别
    def read_power_data(self):
        subcarrier_powers = []
        with open(self.subcarrier_file_name)as subcarrier_file:
            csv_reader = csv.reader(subcarrier_file)
            for row in csv_reader:
                subcarrier_powers.append(row)
        return subcarrier_powers

if __name__ == "__main__":
    util = Util()
    #util.generate_enviroment_data()
    print(util.read_environment_data())
    #print(util.read_power_data())
