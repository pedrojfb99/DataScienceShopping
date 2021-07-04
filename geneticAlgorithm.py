import collections
import json
import pickle

import numpy as np
import os
import random
from itertools import combinations
from mlxtend.frequent_patterns import fpgrowth as fpnorm
from fpgrowth_py import fpgrowth as fptwo
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

import main
from simulatorOfTheSimulator import SoS


class CDGenetic:

    population = []
    popSize = 0
    numGenes = 0
    aptidoes = []

    g = 1
    pm = 0.2
    pc = 0.3

    def __init__(self, size, notToMove, config, sim, shelves, toStay, epochs):

        self.epochs = epochs
        self.popSize = size
        self.stay = notToMove
        self.sim = sim
        self.shelves = list(shelves)
        self.initializePopulation(config)
        self.toStay = toStay
        self.aptidoes = {}


        self.all_config = []

        for s in self.shelves:

            if s in self.sim.shoppingClass.config:
                self.all_config.append(s)

        self.evolution()

    def initializePopulation(self,config):


        for indice in range(self.popSize):

            to_switch = []

            # Obtenho a prateleiras que vão sofrer alterações
            for i in config:
                if i in self.sim.shoppingClass.config:

                    if not self.stay[i]:
                        to_switch.append(config[i])


            random.shuffle(to_switch)


            # Insere-se os valores randomizados
            aux = []
            counter = 0
            for i in config:
                if i in self.sim.shoppingClass.config:
                    if not self.stay[i]:
                        aux.append(to_switch[counter])
                        counter +=1
                    else:
                        aux.append(config[i])


            self.population.append((indice,aux))


    def calculatProfitConfig(self, config):

        config = {self.all_config[i]: config[i] for i in range(len(self.all_config))}
        self.sim.shoppingClass.changeShoppingConfigCustom(config)

        #self.sim.shoppingClass.plotImportance()

        customers = self.sim.generateCustomers(1000)

        sales = self.sim.simulateCustomers(customers)

        profit = self.sim.evaluateShoppingCost(sales)

        return profit


    def mutation(self, config):


        config = {self.all_config[i]: config[i] for i in range(len(self.all_config))}
        result = self.sim.shoppingClass.changeShoppingConfigCustom(config)


        halfResult = []
        if len(result) > 2:

            if len(result) % 2 == 0:
                halfResult = random.sample(result, int(len(result)/2))
            else:
                halfResult = random.sample(result, int(len(result)/2) - 1)


        saving = []
        for i in self.shelves:

            if i in halfResult and i not in self.toStay:
                saving.append(config[i])

        random.shuffle(saving)

        counter = 0

        for i in self.shelves:

            if i in halfResult and i not in self.toStay:
                config[i] = saving[counter]
                counter+=1


        return list(config.values())



    def fillAptidao(self):

        for i in self.population:
            self.aptidoes[i[0]] = self.calculatProfitConfig(i[1])




    def getMax(self):
        maximum = -1

        for i in self.aptidoes:

            if self.aptidoes[i] > maximum:

                maximum = self.aptidoes[i]

        return maximum


    def getProfitChrom(self, id):

        return self.aptidoes[id]

    def pickBetter(self):

        self.aptidoes = dict(sorted(self.aptidoes.items(), key=lambda item: item[1]))

        auxiliar = list(self.aptidoes.keys())
        auxiliar.reverse()

        best = auxiliar[0]

        for i in self.population:
            if i[0] == best:

                return [self.aptidoes[best],i]


    def pickNext(self):

        self.aptidoes = dict(sorted(self.aptidoes.items(), key=lambda item: item[1]))

        auxiliar = list(self.aptidoes.keys())
        auxiliar.reverse()

        best = auxiliar[:int(len(self.population)/2)]

        goNext = []
        for i in self.population:
            if i[0] in best:
                goNext.append(i)

        return goNext



    def evolution(self):

        for epoch in range(self.epochs):

            #Selecionar os individuos para mutação
            aux = self.population.copy()

            for i in aux:
                print(i[1])
                exit()
                result = self.mutation(i[1])

                self.population.append((len(self.population), result))

            self.fillAptidao()
            self.pickNext()
            self.population = self.pickNext()
            pickle.dump(self.pickBetter(),open(f"results/{epoch}.p","wb"))
            print(f"Epoch : {epoch}\n"
                  f"Best : {self.getMax()}")
            print(self.population)





def encodeData():

    df = pd.read_csv('products.txt', delimiter="\t")
    dataHere = df['Nome'].str.strip()

    indexes = [x for x in range(0,len(dataHere))]

    df['ID'] = indexes
    #print(data.to_numpy())


    return df


products = encodeData()



def loadMain():
    if os.path.exists("productPair.csv"):
        dfAux = pd.read_csv("productPair.csv")

        shelves = dfAux['shelve'].to_numpy()
        products_aux = dfAux['product'].to_numpy()


        dictionary = {}
        for i, j in zip(shelves,products_aux):
            dictionary[i] = j

        return dict(collections.OrderedDict(sorted(dictionary.items())))


def generateSimulator(config):

    #QoL for display
    pd.set_option('display.max_columns', 30)

    data, explanations = main.cvtCsvDataframe(pd.read_csv("data.csv"), pd.read_csv("explanations.csv"))

    mergedReceiptExplanations = pd.merge(data, explanations, on='receiptID', how='outer')

    simulator = SoS(config,main.obtainStaminaDistribution(mergedReceiptExplanations['DISTANCE'].to_numpy()), explanations)

    return simulator

def aux():


    data = loadMain()

    notToMove = {}

    to_stay = [6,7,8,9,185,208,231,254,215,216,217,218,470,471,472,473,250,273,296,251,412,435,389,437,391,366,368,460,247,297,365,296,181,434]


    for i in data:

        if int(i) in to_stay:

            notToMove[i] = True

        else:
            notToMove[i] = False

    sim = generateSimulator([])
    CDGenetic(2, notToMove, data, sim, data.keys(), to_stay,100)

    #fp = FPGrowth(data2,0.35,0.4)




if __name__ == '__main__':



    pd.set_option('display.max_columns', 30)

    aux()

    exit()



