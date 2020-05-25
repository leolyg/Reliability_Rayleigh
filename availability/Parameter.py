import math

#用于储存实验中的参数(主要是环境设定方面的)

class Parameter:
    def __init__(self,
                    scenario =None,
                    episode = 10000,#用户分配的episode
                    power_optimization_episode = 10000,#功率优化的episode
                    episode_log_step=10,#每隔多少步记录一次执行的结果
                    bs_num = 5,
                    subcarrier_num = 12,
                    ue_num = 6,
                    step_threshold = 8,#分配的次数
                    fixed_num = 4,
                    band_num=2,
                    power_level = 16,#功率级别
                    theta = 1, #power_budget radio
                    target_nine_number= 4,
                    algorithm_index = 0
                    ):
        self.episode = episode
        self.power_optimization_episode = power_optimization_episode
        self.episode_log_step =episode_log_step 
        self.bs_num = bs_num
        self.subcarrier_num = subcarrier_num
        self.ue_num = ue_num
        self.band_num = band_num
        self.step_threshold =step_threshold
        self.fixed_num = fixed_num
        self.tau = (2**(1/180000))#或者取值1
        self.power_level = power_level
        self.theta = theta
        self.P_BS_macro_dbm=46;#宏基站功率,相当于40W
        self.P_BS_pico_dbm=30;#微基站功率相当于1W
        self.BS_wavelengh_1 = 0.375#带宽1
        self.BS_wavelengh_2 = 0.125#带宽2
        self.BS_alpha_1 = 3 #alpha1
        self.BS_alpha_2 = 4 #alpha2
        self.BS_power = [self.transfer_dbm_to_power(self.P_BS_macro_dbm), self.transfer_dbm_to_power(self.P_BS_pico_dbm)]#功率数组
        #self.BS_power = [self.P_BS_macro_dbm, self.P_BS_pico_dbm]#功率数组
        self.BS_wavelengh = [self.BS_wavelengh_1,self.BS_wavelengh_2]#波长数组
        self.BS_alpha = [self.BS_alpha_1,self.BS_alpha_2] #alpha数组
        BW = 1e7;#带宽
        N_0_dbm = -174 + 10*math.log(10,BW)
        self.N_0 = 10**((N_0_dbm-30)/10)
        self.target_nine_number = target_nine_number
        self.encodeFlag=True
        self.variant = ['DQN','Double_DQN','Dueling_DQN','Prioritized_DQN']
        self.algorithm = self.variant[algorithm_index]

        if scenario == 'scenario0':
            self.bs_num = 4
            self.subcarrier_num = 6
            self.ue_num = 3
            self.step_threshold = 5
        if scenario == 'scenario1':
            self.bs_num = 5
            self.subcarrier_num = 10
            self.ue_num = 4
            self.step_threshold = 6
        elif scenario == 'scenario2':
            self.bs_num = 5
            self.subcarrier_num = 10
            self.ue_num = 6
            self.step_threshold = 8
        elif scenario == 'scenario3':
            self.bs_num = 5
            self.subcarrier_num = 10
            self.ue_num = 8
            self.step_threshold = 8
        elif scenario == 'scenario4':
            self.bs_num = 5
            self.subcarrier_num = 12
            self.ue_num = 10
            self.step_threshold = 12
        elif scenario == 'scenario5':
            self.bs_num = 5
            self.subcarrier_num = 10
            self.ue_num = 6
            self.step_threshold = 10
        elif scenario == 'scenario6':
            self.bs_num = 5
            self.subcarrier_num = 12
            self.ue_num = 6
            self.step_threshold = 8

        elif scenario == 'scenario7':
            self.bs_num = 5
            self.subcarrier_num = 10
            self.ue_num = 8
            self.step_threshold = 12


    ## dbm转化为瓦特  
    def transfer_dbm_to_power(self,dbm_value):
        p = 10**((dbm_value-30)/10)
        return p


    