import csv
import numpy as np
from availability.Availability_Rayleigh import Availability_Rayleigh
from availability.Parameter import Parameter
import Random_assignment as rand_assignment
import time

# 参数设置
ue_num = 6
#需要同时进化sequence
#等同于scenario2

UE_sequence = [0, 1, 2, 3, 4, 5, 0, 1]
scenario = 'scenario2'
parameter = Parameter(scenario)
sequence_lengh = len(UE_sequence)

bs_num = 5
subcarrrier_num = 10
POP_SIZE = 1000
N_GENERATIONS = 100
MUTATION_RATE = 0.003

# 随机生成数量为num的分配方案
def init_population(num):
    result = np.zeros([num, bs_num * subcarrrier_num])
    for i in range(num):
        plan = rand_assignment.generate_a_plan(UE_sequence, bs_num, subcarrrier_num, step_threshold=sequence_lengh)
        plan = plan.reshape(-1)
        result[i] = plan
    return result

def get_population_fitness(population):
    num = len(population)
    fitnesses = np.zeros(num)
    for i in range(num):
        fitnesses[i] = get_fitness(population[i])
    return fitnesses

def get_fitness(individual):
    individual = individual.reshape(bs_num, subcarrrier_num).astype(int)
    avilability = Availability_Rayleigh(parameter)
    reward = avilability.get_reward(individual)
    return reward

def select(population, fitnesses):
    idx = np.random.choice(np.arange(POP_SIZE), size=int(POP_SIZE / 2), replace=True,
                           p=fitnesses / fitnesses.sum())
    return population[idx]

def calculate_repermutation(father_difference, mother_difference):
    length = len(father_difference)
    father_first = father_difference[0:int(length / 2)]
    father_second = father_difference[int(length / 2):length]
    permutation = np.append(np.array([]), father_first)
    for i in range(len(mother_difference)):
        if (np.in1d(mother_difference[i], father_second)):
            indexes = np.argwhere(father_second == mother_difference[i])
            father_second = np.delete(father_second, indexes[0])
            permutation = np.append(permutation, mother_difference[i])
    return permutation

def crossover(population):
    count = len(population)
    children = np.zeros([count, bs_num * subcarrrier_num])
    for i in range(count):
        father = population[i]
        mother = population[count - 1 - i]
        indexes = np.nonzero(father - mother)
        if len(indexes) == 0:
            children[i] = father
        else:
            father_difference = father[indexes]
            mother_difference = mother[indexes]
            permutation = calculate_repermutation(father_difference, mother_difference)
            child = father
            child[indexes] = permutation
            children[i] = child
    return children

def mutate(population):
    for i in range(len(population)):
        # 对数据进行交换
        if np.random.rand() < MUTATION_RATE:
            x = np.random.randint(bs_num * subcarrrier_num)
            y = np.random.randint(bs_num * subcarrrier_num)
            x_value = population[i][x]
            y_value = population[i][y]
            population[i][x] = y_value
            population[i][y] = x_value
    return population

def evaluate_population(index, population):
    fitnesses = get_population_fitness(population)
    max_index = np.argmax(fitnesses)
    print("best individual ", population[max_index])
    print("iteration count:", index)
    print("best fitness:", fitnesses[max_index])
    print("average fitness",np.average(fitnesses))
    return fitnesses[max_index],np.average(fitnesses)

def run():
    # init population
    population = init_population(POP_SIZE)
    result_list = []
    for index in range(N_GENERATIONS):
        # # get fitness
        fitnesses = get_population_fitness(population)
        # # select good individuals
        population = select(population, fitnesses)
        # # crossover
        children = crossover(population)
        # #补充保持种群不变
        population = np.vstack([population, children])
        # # mutate
        population = mutate(population)
        
        best_fitness,average_fitness = evaluate_population(index, population)
        
       # result_list.append(best_fitness)
        result_list.append(average_fitness)
    with open('ga_result.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_list)



if __name__ == '__main__':
    repeat_count = 100
    start = time.time()
    for i in range(repeat_count):
        print("-----start run -----")
        run()
        print("------end run ------")
    end = time.time()
    print("循环运行时间:%.2f秒" % (end - start))
