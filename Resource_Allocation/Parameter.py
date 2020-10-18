import math

# 用于储存实验中的参数(主要是环境设定方面的)
scenarios = [
    {
        'bs_num': 10,
        'subcarrier_num': 20,
        'ue_num': 4,
        'rho': 6,
    }, {
        'bs_num': 10,
        'subcarrier_num': 20,
        'ue_num': 6,
        'rho': 9,
    }, {
        'bs_num': 10,
        'subcarrier_num': 20,
        'ue_num': 8,
        'rho': 12,
    }, {
        'bs_num': 10,
        'subcarrier_num': 20,
        'ue_num': 10,
        'rho': 15,
    }, {
        'bs_num': 10,
        'subcarrier_num': 20,
        'ue_num': 12,
        'rho': 18,
    }, {
        'bs_num': 10,
        'subcarrier_num': 20,
        'ue_num': 14,
        'rho': 21
    }, {
        'bs_num': 10,
        'subcarrier_num': 20,
        'ue_num': 16,
        'rho': 24,
    }, {
        'bs_num': 10,
        'subcarrier_num': 20,
        'ue_num': 18,
        'rho': 27
    }, {
        'bs_num': 10,
        'subcarrier_num': 20,
        'ue_num': 20,
        'rho': 30
    },{
        'bs_num':5,
        'subcarrier_num':5,
        'ue_num':4,
        'rho':6

    },{
      'bs_num':5,
      'subcarrier_num':5,
      'ue_num':2,
      'rho':4
    }
]


class Parameter:
    def __init__(self,
                 bs_num,
                 subcarrier_num,
                 ue_num,
                 step_threshold,
                 scenario_index =None,
                 power_level=16,  # 功率级别
                 theta=1,  # power_budget radio
                 ):
        if(scenario_index!=None):
            self.bs_num = scenarios[scenario_index]['bs_num']
            self.subcarrier_num = scenarios[scenario_index]['subcarrier_num']
            self.ue_num = scenarios[scenario_index]['ue_num']
            self.step_threshold = scenarios[scenario_index]['rho']
        else:
            self.bs_num = bs_num
            self.subcarrier_num = subcarrier_num
            self.ue_num = ue_num
            self.step_threshold = step_threshold
        self.band_num = 2
        self.tau = 1  # 或者取值1
        self.power_level = power_level
        self.theta = theta
        self.P_BS_macro_dbm = 46;  # 宏基站功率,相当于40W
        self.P_BS_pico_dbm = 30;  # 微基站功率相当于1W
        self.BS_wavelengh_1 = 0.375  # 带宽1
        self.BS_wavelengh_2 = 0.125  # 带宽2
        self.BS_alpha_1 = 3  # alpha1
        self.BS_alpha_2 = 4  # alpha2
        self.BS_power = [self.transfer_dbm_to_power(self.P_BS_macro_dbm),
                         self.transfer_dbm_to_power(self.P_BS_pico_dbm)]  # 功率数组
        self.BS_wavelengh = [self.BS_wavelengh_1, self.BS_wavelengh_2]  # 波长数组
        self.BS_alpha = [self.BS_alpha_1, self.BS_alpha_2]  # alpha数组
        BW = 1e7;  # 带宽
        N_0_dbm = -174 + 10 * math.log(10, BW)
        self.N_0 = 10 ** ((N_0_dbm - 30) / 10)

    ## dbm转化为瓦特
    def transfer_dbm_to_power(self, dbm_value):
        p = 10 ** ((dbm_value - 30) / 10)
        return p
