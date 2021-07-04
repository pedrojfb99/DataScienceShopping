import collections

import pandas as pd
import numpy as np
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import pprint
import csv


def encodeData():

    df = pd.read_csv('products.txt', delimiter="\t")
    dataHere = df['Nome'].str.strip()

    indexes = [x for x in range(0,len(dataHere))]

    df['ID'] = indexes
    #print(data.to_numpy())


    return df


products = encodeData()

class Cell:

    def __init__(self, id, product, rank):
        self.id = id
        self.product = product
        self.rank = rank



class Shopping:

    def __init__(self,size,config):

        self.entrance = 414
        self.exit = 115

        self.productsAux = {}


        #Load shopping's configuration
        self.config = self.loadConfig(size)

        #Initialize shopping with custom class
        self.shopping = np.empty(shape=(size[0],size[1]), dtype=object)

        #Initialize the shopping with the default shelves' layout
        if len(config) ==  0:
            self.initializeShopping(size)
        else:
            self.changeShoppingConfig(config)

        #Get a graph representation of the shopping hallways
        self.graphShopping = nx.DiGraph(self.createGraph())

        #Calculate each shelve importance
        self.calculateImportance()

        #self.plotImportance()

        print("Shopping Initialized !!!\n")



    def plotImportance(self):

        '''
        Plots the importance of every shelve in the supermarket

        :return: None
        '''

        size = self.shopping.shape

        aux = np.zeros((size[0],size[1]))

        for j in range(size[1]):

            for i in range(size[0]):

                if self.shopping[i][j].id in self.config:
                    current = int(self.shopping[i][j].rank)
                else:
                    current = int(-40)

                aux[i][j] = current

        sns.heatmap(aux, linewidths=0.5)
        plt.show()




    def distanceToEntrance(self, currentCell):
        '''
        Calculates the distance of each shelf from the entrance
        :param currentCell: Receives the current shopping cell
        :return: Return the distance to the closest neighbor
        '''

        neighbors = []
        notToAdd = [1,461,483,23]

        try:
            #If there are neighbors in the top
            if currentCell[0] > 0:

                #Get the top neighbor
                neighbors.append(self.shopping[currentCell[0] - 1][currentCell[1]].id)

            #If there are neighbors on the left
            if currentCell[1] > 0:
                neighbors.append(self.shopping[currentCell[0]][currentCell[1] - 1].id)

            #If there are neighbors on the right
            if currentCell[1] < self.shopping.shape[1]:
                neighbors.append(self.shopping[currentCell[0]][currentCell[1] + 1].id)

            #If there are neighbors on the bottom
            if currentCell[0] < self.shopping.shape[0]:
                neighbors.append(self.shopping[currentCell[0] + 1][currentCell[1]].id)
        except:
            pass


        pathsLenght = []

        for possiblePath in neighbors:
            if possiblePath not in notToAdd and possiblePath not in self.config:
                pathsLenght.append(len(nx.dijkstra_path(self.graphShopping, self.entrance, possiblePath)))

        return -min(pathsLenght), len(pathsLenght)


    def findNeighbors(self, currentCell, typeSearch, n):
        '''

        :param currentCell: Current cell to search
        :param typeSearch: Type of search 1 - Halls 2- Shelves
        :return: Return the neighbors
        '''

        neighbors = []

        for i in range(n):

            try:
                #If there are neighbors in the top
                if currentCell[0] > 0:

                    #Get the top neighbor
                    neighbors.append(self.shopping[currentCell[0] - i][currentCell[1]].id)

                #If there are neighbors on the left
                if currentCell[1] > 0:
                    neighbors.append(self.shopping[currentCell[0]][currentCell[1] - i].id)

                #If there are neighbors on the right
                if currentCell[1] < self.shopping.shape[1]:
                    neighbors.append(self.shopping[currentCell[0]][currentCell[1] + i].id)

                #If there are neighbors on the bottom
                if currentCell[0] < self.shopping.shape[0]:
                    neighbors.append(self.shopping[currentCell[0] + i][currentCell[1]].id)
            except:
                pass

        aux = []

        if typeSearch == 1:

            notToAdd = [1,461,483,23]

            for i in neighbors:
                if i not in self.shoppingClass.config and i not in notToAdd:
                    aux.append(i)

        else:
            notToAdd = [1, 461, 483, 23]

            for i in neighbors:
                if i in self.shoppingClass.config and i not in notToAdd:
                    aux.append(i)

        return aux


    def getCellRank(self,cell):
        '''
        Returns the rank of each cell
        :param cell:
        :return:
        '''

        size = self.shopping.shape

        for j in range(size[1]):

            for i in range(size[0]):

                if self.shopping[i][j].id == cell:
                    return products.loc[products['ID'] == self.shopping[i][j].product, 'Margem Lucro'].iloc[0]

        return 0

    def findNeighbors(self, currentCell, typeSearch, n):
        '''

        :param currentCell: Current cell to search
        :param typeSearch: Type of search 1 - Halls 2- Shelves
        :return: Return the neighbors
        '''

        neighbors = []

        for i in range(1,n+1):

            try:
                # If there are neighbors in the top
                if currentCell[0] > n+1:
                    # Get the top neighbor
                    neighbors.append(self.shopping[currentCell[0] - i][currentCell[1]].id)

                # If there are neighbors on the left
                if currentCell[1] > n+1:
                    neighbors.append(self.shopping[currentCell[0]][currentCell[1] - i].id)

                # If there are neighbors on the right
                if currentCell[1]+n < self.shopping.shape[1]:

                    neighbors.append(self.shopping[currentCell[0]][currentCell[1] + i].id)

                # If there are neighbors on the bottom
                if currentCell[0]+n < self.shopping.shape[0]:
                    neighbors.append(self.shopping[currentCell[0] + i][currentCell[1]].id)


            except:
                pass

        aux = []
        if typeSearch == 1:

            notToAdd = [1, 461, 483, 23]

            for i in neighbors:
                if i not in self.config and i not in notToAdd:
                    aux.append(i)

        else:
            notToAdd = [1, 461, 483, 23]
            for i in neighbors:
                if i in self.config and i not in notToAdd:
                    aux.append(i)

        return aux


    def getNeighborsQuality(self, cell, typeSearch):
        '''
        Returns the average rank of the n neighbors
        :param cell:
        :return:
        '''
        neighs = self.findNeighbors(cell, typeSearch,2)
        sum = 0


        #Calculate sum of each rank's neighbor
        for i in neighs: sum += self.getCellRank(i)

        return sum/len(neighs)

    def calculateImportance(self):
        '''
        Calculates the importance of each shelve

        :return: None
        '''

        size = self.shopping.shape
        controller = 1

        geneticRank = []

        for j in range(size[1]):

            for i in range(size[0]):

                if controller in self.config:
                        importance = self.distanceToEntrance([i,j])

                        productProfit = products.loc[products['ID'] == self.shopping[i][j].product, 'Margem Lucro'].iloc[0]

                        neighborsAverageRank = self.getNeighborsQuality([i,j],2)


                        if importance[0] + productProfit + neighborsAverageRank < 10:
                            geneticRank.append(self.shopping[i][j].id)

                        self.shopping[i][j].rank = importance[0] + productProfit + neighborsAverageRank

                controller += 1

        return geneticRank


    def add(self,adj_list, a, b):
        '''
        :param adj_list: the adjancency matrix
        :param a: first cell to compare
        :param b: second cell to comapre
        :return: None
        '''

        notToAdd = [1,461,483,23]

        if a not in notToAdd and b not in notToAdd:
            adj_list.setdefault(a, []).append(b)
            adj_list.setdefault(b, []).append(a)


    def createGraph(self):
        '''
        Creates a graph based on the hallways of the supermarket
        :return: returns the edge connections
        '''

        size = self.shopping.shape

        graphMatrix = np.zeros(size,dtype=int)

        controller = 1

        # Iterate through columns
        for j in range(size[1]):

            for i in range(size[0]):

                graphMatrix[i][j] = controller

                controller += 1



        adj_list = {}
        for i in range(len(graphMatrix)):
            for j in range(len(graphMatrix[i])):

                current = graphMatrix[i][j]

                if current not in self.config:

                    if j < len(graphMatrix[i]) - 1:
                        if graphMatrix[i][j + 1] not in self.config:
                            self.add(adj_list, current, graphMatrix[i][j + 1])

                    if i > 0:
                        if graphMatrix[i - 1][j] not in self.config:
                            self.add(adj_list, current, graphMatrix[i - 1][j])



        return adj_list




    def changeShoppingConfig(self, newConfig):

        size = self.shopping.shape

        controller = 1

        productsPer = 0

        #Iterate through columns
        for j in range(size[1]):

            for i in range(size[0]):

                #If it a shelve then add the product
                if controller in self.config:
                    try:
                        self.shopping[i][j] = Cell(controller,newConfig[productsPer],0)
                        self.productsAux[controller] = newConfig[productsPer]
                    except:
                        self.shopping[i][j] = Cell(controller,-100,0)

                    productsPer += 1

                #Corridors
                else:
                    self.shopping[i][j] = Cell(controller, -100, 0)

                controller += 1


        self.calculateImportance()


    def changeShoppingConfigCustom(self, newConfig):
        size = self.shopping.shape

        auxiliar = []

        controller = 1

        productsPer = 0

        #Iterate through columns
        for j in range(size[1]):

            for i in range(size[0]):

                #If it a shelve then add the product
                if controller in self.config:
                    try:
                        self.shopping[i][j] = Cell(controller,newConfig[controller],0)
                        self.productsAux[controller] = newConfig[controller]
                        auxiliar.append(newConfig[controller])
                    except:
                        self.shopping[i][j] = Cell(controller,-100,0)

                    productsPer += 1

                #Corridors
                else:
                    self.shopping[i][j] = Cell(controller, -100, 0)

                controller += 1

        return self.calculateImportance()

    def displayShopping(self):
        '''
        Displays the matrix representation of the shopping

        :return: None
        '''


        size = self.shopping.shape

        aux = np.zeros((size[0],size[1]))

        for j in range(size[1]):

            for i in range(size[0]):

                current = int(self.shopping[i][j].product)

                aux[i][j] = current

        sns.heatmap(aux, linewidths=0.5)
        plt.show()







    def initializeShopping(self,size):
        '''
        Initializes the matrix with the default products' layout

        :param size: Shopping's size
        :return: None
        '''

        productPerShelve = np.asarray(self.loadProducts())

        controller = 1

        productsPer = 0

        for j in range(size[1]):

            for i in range(size[0]):

                if controller in self.config:
                    try:
                        self.shopping[i][j] = Cell(controller,productPerShelve[productsPer],0)
                        self.productsAux[controller] = productPerShelve[productsPer]
                    except:
                        self.shopping[i][j] = Cell(controller,-100,0)

                    productsPer += 1


                else:
                    self.shopping[i][j] = Cell(controller, -100, 0)

                controller += 1







    #Load config
    def loadConfig(self,size):
        '''
        Gets all the possible products' positions
        :param size:
        :return:Array with the possible shelves
        '''

        aux = np.zeros(size)

        if os.path.exists("config.p"): return pickle.load(open("config.p"))
        else:

            counter = 1
            for j in range(size[1]):

                for i in range(size[0]):
                    aux[i][j] = counter

                    counter += 1

            # Get all the shelves in the supermarket
            generalConfig = np.asarray(list(aux[0, 1:-1]) + list(aux[1:-1, 0]) + list(aux[1:-1, -1]) + list(aux[-1, 1:-1]) +
                    list(aux[2:-6,2:4].flatten()) + list(aux[2:-6,6:8].flatten()) + list(aux[2:-6,9:11].flatten()) +
                    list(aux[2:-6,13:15].flatten()) + list(aux[2:-6,17:19].flatten()) +
                    list(aux[-3,2:4].flatten()) + list(aux[-4:-2,6:8].flatten()) + list(aux[-4:-2,10:13].flatten()) + list(aux[-4:-2,15:19].flatten()))



            #Remove entrance and exit
            generalConfig = generalConfig[generalConfig != 115]
            generalConfig = generalConfig[generalConfig != 414]


            #Remove duplicates
            config = np.unique(generalConfig)

            return config


    #Load products
    def loadProducts(self):
        '''
        Loads the products
        :return: List of products
        '''

        df = pd.read_csv('products.txt', delimiter="\t")

        #Get number of shelves for each product
        lst = df.to_numpy()[:,-2]

        aux = []

        for i in range(len(lst)):
            for k in range(lst[i]):
                aux.append(i)

        return aux


def loadConfig():

    with open('products.json') as f:
        data = json.load(f)

    return list(data.values())








if __name__ == '__main__':

    loadConfig()

    #Create a shopping
    shop = Shopping([23,21], [])

    defaultConfig = shop.loadProducts()

    with open("profitImportance.json", "r") as f:
        config= json.load(f)

    od = collections.OrderedDict(sorted(config.items()))
    ax = []

    for i in sorted(od):
        print(i)

    print(list(ax))
    exit()

    #random.shuffle(defaultConfig)

    shop.changeShoppingConfig(loadConfig())

    #shop.plotImportance()



    shop.plotImportance()




