import numpy as np
import csv

class UE_selection:

    def __init__(self):
        self.ue_num = 6
        self.step_threshold = 8
        self.POP_SIZE = 100
        self.N_GENERATIONS = 100
        self.MUTATION_RATE = 0.003

    def init_population(self, num):
        population = np.random.randint(low=0, high=self.ue_num, size=(num, self.step_threshold))
        return population

    def get_population_fitness(self, population):
        num = len(population)
        fitness = np.zeros(num)
        for i in range(num):
            fitness[i] = self.get_fitness(population[i])
        return fitness

    def get_fitness(self, individual):
        unique = np.unique(individual)
        if len(unique) < self.ue_num/2:
            return len(unique)*0.5
        else:
            return len(unique)

    def select(self, population, fitness):
        idx = np.random.choice(np.arange(self.POP_SIZE), size=int(self.POP_SIZE / 2), replace=True,
                               p=fitness / fitness.sum())
        return population[idx]

    def crossover(self, population):
        count = len(population)
        children = np.zeros([count, self.step_threshold])
        for i in range(count):
            father = population[i]
            mother = population[count-1-i]
            first_part = father[0:int(self.step_threshold/2)]
            second_part = mother[int(self.step_threshold/2):self.step_threshold]
            child = np.append(first_part, second_part)
            children[i] = child
        return children

    def mutate(self, population):
        for i in range(len(population)):
            # 对数据进行交换
            if np.random.rand() < self.MUTATION_RATE:
                index = np.random.randint(self.step_threshold)
                ue_index = np.random.randint(self.ue_num)
                population[i][index] = ue_index
        return population

    def evaluate_population(self, index, population):
        fitness = self.get_population_fitness(population)
        max_index = np.argmax(fitness)
        # print("best individual ", population[max_index])
        print("iteration count:", index)
        print("best fitness:", fitness[max_index])
        print("best individual", population[max_index])
        print("average fitness",np.average(fitness))
        return fitness[max_index]

    def run(self):
        # init population
        population = self.init_population(self.POP_SIZE)
        self.evaluate_population(0, population)
        result_list = []
        for index in range(self.N_GENERATIONS):
            # # get fitness
            fitness = self.get_population_fitness(population)
            # # select good individuals
            population = self.select(population, fitness)
            # # crossover
            children = self.crossover(population)
            # #补充保持种群不变
            population = np.vstack([population, children])
            # # mutate
            population = self.mutate(population)
            best_fitness = self.evaluate_population(index+1, population)
            result_list.append(best_fitness)
        with open('ga_result.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)

if __name__ == '__main__':
    selection = UE_selection()
    #population = selection.init_population(10)
    #print(selection.get_population_fitness(population))
    selection.run()



