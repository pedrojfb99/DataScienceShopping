import collections
import json
import os
from datetime import time
import random
from tqdm import tqdm

from main import cvtCsvDataframe
import pickle
import numpy as np
import pandas as pd
import random
import networkx as nx
import time
from main import FPGrowth
from shopping import Shopping, Cell
import main

# QoL for display
pd.set_option('display.max_columns', 30)

def encodeData():

    df = pd.read_csv('products.txt', delimiter="\t")
    dataHere = df['Nome'].str.strip()

    indexes = [x for x in range(0,len(dataHere))]

    df['ID'] = indexes
    #print(data.to_numpy())


    return df


products = encodeData()

'''
    It is suppose to simulate N amount of shopping trips given test wishlists and staminas.
    
    1 - Create a shopping with the given configuration
    
    2 - Generate N random wishlists and their stamina
    
    3 - Simulate each one and save the results 
    
    4 - Analyse the supermarket profit
'''

class SoS:

    def __init__(self, configuration, staminaDistr,explanations):

        self.shoppingClass = Shopping([23,21],configuration)

        #self.shoppingClass.changeShoppingConfig(configuration)

        self.shopping = self.shoppingClass.shopping

        self.staminaDistr = staminaDistr

        self.explanations = explanations

        self.auxNeighbors = self.getAuxNeighbors()

        self.auxNeighborsPrimary = self.getAuxNeighborsPrimary()


        data, explanations = cvtCsvDataframe(pd.read_csv("data.csv"), pd.read_csv("explanations.csv"))

        mergedReceiptExplanations = pd.merge(data, explanations, on='receiptID', how='outer')

        self.boughtAndWishlist = mergedReceiptExplanations[['PRODUCTS', 'WISHLIST']].to_numpy()


    def generateCustomers(self, samples):
        '''

        :return: Returns a sample of random customers with stamina and wishlist
        '''

        customers = []

        wishlists = list(self.explanations['WISHLIST'].to_numpy())
        randomWishlists = random.sample(wishlists,samples)


        staminas = self.staminaDistr.sample(samples)

        for i, j in zip(randomWishlists,staminas):
            customers.append((i,int(j)))

        return customers


    def findNeighbors(self, currentCell, typeSearch):
        '''

        :param currentCell: Current cell to search
        :param typeSearch: Type of search 1 - Halls 2- Shelves
        :return: Return the neighbors
        '''

        neighbors = []

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


    def findClosestProduct(self, item):
        '''

        :param item: Receives an item to search for
        :return: Returns the closest product path there is
        '''

        size = self.shopping.shape

        allPathsToItem = []

        for j in range(size[1]):

            for i in range(size[0]):

                if self.shopping[i][j].product == item:

                    pathsToThisCell = self.auxNeighborsPrimary[f"[{i},{j}]"]

                    for s in pathsToThisCell: allPathsToItem.append(s)

        pathsLenght = []
        paths = []


        for possiblePath in allPathsToItem:
                paths.append(nx.dijkstra_path(self.shoppingClass.graphShopping, self.shoppingClass.entrance, possiblePath))
                pathsLenght.append(len(nx.dijkstra_path(self.shoppingClass.graphShopping, self.shoppingClass.entrance, possiblePath)))



        #Return the minimium path
        return paths[np.argmin(pathsLenght)]


    def getAuxNeighborsPrimary(self):

        aux = {}

        size = self.shopping.shape
        for j in range(size[1]):

            for i in range(size[0]):

                    aux[f"[{i},{j}]"] = self.findNeighbors([i, j], 1)

        return aux

    def getAuxNeighbors(self):

        aux = {}

        size = self.shopping.shape
        for j in range(size[1]):

            for i in range(size[0]):

                    aux[f"[{i},{j}]"] = self.findNeighbors([i, j], 2)

        return aux




    def getCellProducts(self, cell):


        size = self.shopping.shape

        for j in range(size[1]):

            for i in range(size[0]):

                if self.shopping[i][j].id == cell:

                    cells = self.auxNeighbors[f"[{i},{j}]"]

                    products = []
                    for c in cells:
                        products.append(self.shoppingClass.productsAux[c])

                    return products



    def getProbabilityOfPicking(self, product):

        #Check if the file already exists
        if os.path.exists("probabilityBuy.p"): probToBuy = pickle.load(open("probabilityBuy.p","rb"))

        #Otherwise write it
        else:

            # organize_data()
            # Read the csv file and convert it to a well formatted dataframe


            aux = {}

            #For each receipt
            for p in tqdm(self.boughtAndWishlist):

                #go through the products bought
                for i in p[0]:

                    if i not in list(aux.keys()):
                        aux[i] = {'NotIn': 0, 'Counter':0}

                    #Increase counter
                    aux[i]['Counter'] += 1

                    #If the product bought is not in the wishlist
                    if i not in p[1]:

                        #Increase counter of times that the product was bought and was not in the wishlist
                        aux[i]['NotIn'] += 1


            probToBuy = {}

            for k in aux:
                probToBuy[k] = aux[k]['NotIn'] / aux[k]['Counter']



            pickle.dump(probToBuy,open("probabilityBuy.p","wb"))


        #Reutrn the respective probability
        return probToBuy[product]


    def simulateCustomers(self,customers):

        '''
        :param customers: Receives a list of customers
        :return: Returns the simulation results
        '''


        sales = []

        #For each customer
        for customer in tqdm(customers):

            currentWishlist = customer[0]
            currentWishlist.reverse()

            currentStamina = customer[1]

            productsBought = []
            #print(f"Customer wishlist: {currentWishlist}")

            #While the customer still has products the wants and still has stamina keep the simulation
            while len(currentWishlist) > 0 and currentStamina > 0:

                item = currentWishlist[0]
                #print(f"Looking for {products.loc[products['ID'] == item, 'Nome'].iloc[0]}")


                closest = self.findClosestProduct(item)
                #print(f"Found {products.loc[products['ID'] == item, 'Nome'].iloc[0]} on cell {closest[-1]}")


                for cell in range(len(closest)):
                    #print(f"I am on cell  {closest[cell]}")

                    prodcutsCloseToCell = self.getCellProducts(closest[cell])


                    for prod in prodcutsCloseToCell:

                        #If the product is in the wishlist then buy it
                        if prod in currentWishlist:
                            #print(f"Found {products.loc[products['ID'] == prod, 'Nome'].iloc[0]} which was in my wishlist, so I bought it.")

                            #Remove it from the wishlist
                            currentWishlist.remove(prod)
                            productsBought.append(prod)

                        #Otherwise calculate the probability of buying it
                        else:


                            #Probability of this product being picked without being in the wishlist
                            prob = self.getProbabilityOfPicking(prod)


                            #Random probability
                            randomProb = random.uniform(0,1)

                            #If it is bought
                            if randomProb <= prob:
                                productsBought.append(prod)
                                #print(f"Felt like buying {products.loc[products['ID'] == prod, 'Nome'].iloc[0]}, so I bought it.")


                    currentStamina -= 1
                    #print(f"Current stamina : {currentStamina}")

                    #Scenarios that the person leaves the shopping
                    if currentStamina <= 0:
                        #print("I got tired!")
                        break
                    elif len(currentWishlist) <= 0:
                        #print("Bought everything!")
                        break

            sales.append(productsBought)


        return sales



    def evaluateShoppingCost(self, sales):
        '''
        :param sales: Receives a list of sales from customers
        :return: Return the calcualte profit for those sales
        '''

        totalProfit = 0

        for sale in tqdm(sales):

            for product in sale:
                totalProfit += (products.loc[products['ID'] == product, 'Preço'].iloc[0] / products.loc[products['ID'] == product, 'Margem Lucro'].iloc[0])


        return totalProfit



def generateSimulator(config):

    #QoL for display
    pd.set_option('display.max_columns', 30)

    data, explanations = main.cvtCsvDataframe(pd.read_csv("data.csv"), pd.read_csv("explanations.csv"))

    mergedReceiptExplanations = pd.merge(data, explanations, on='receiptID', how='outer')

    simulator = SoS(config,main.obtainStaminaDistribution(mergedReceiptExplanations['DISTANCE'].to_numpy()), explanations)

    return simulator



def orderProductsPerImportanceAndProfit(shop):

    #Order products
    ordered = products.sort_values(by=['Margem Lucro'], ascending=True)
    ordered = ordered['ID'].to_numpy()

    aux = []
    for p in ordered:
        for _ in range(products.loc[products['ID'] == p,'Total Prateleiras'].iloc[0]):
            aux.append(p)

    size = [23,21]

    ranksShelves = {}

    #Order importance cells
    for j in range(size[1]):

        for i in range(size[0]):

            ranksShelves[shop.shopping[i][j].id] = shop.shopping[i][j].rank

    ranksShelves = dict(sorted(ranksShelves.items(), key=lambda item: item[1]))

    indice = 0

    for i in ranksShelves.keys():
        if i in shop.shoppingClass.config:
            ranksShelves[i] = int(aux[indice])
            indice += 1


    with open("profitImportance.json","w") as f:
        json.dump(ranksShelves,f)

    return ranksShelves


def orderProductsPerPair(shop):


    data, explanations = cvtCsvDataframe(pd.read_csv("data.csv"), pd.read_csv("explanations.csv"))

    if os.path.exists("productPair.csv"):
        dfAux = pd.read_csv("productPair.csv")

        shelves = dfAux['shelve'].to_numpy()
        products_aux = dfAux['product'].to_numpy()


        dictionary = {}
        for i, j in zip(shelves,products_aux):
            dictionary[i] = j


        return dict(collections.OrderedDict(sorted(dictionary.items())))


    else:
        #Order products
        ordered = products.sort_values(by=['Margem Lucro'], ascending=True)
        ordered = ordered['ID'].to_numpy()

        aux = []
        for p in ordered:
            for _ in range(products.loc[products['ID'] == p,'Total Prateleiras'].iloc[0]):
                aux.append(p)

        size = [23,21]

        ranksShelves = {}
        auxRankShelves = {}

        #Order importance cells
        for j in range(size[1]):

            for i in range(size[0]):

                ranksShelves[shop.shopping[i][j].id] = shop.shopping[i][j].rank
                auxRankShelves[shop.shopping[i][j].id] = False
        ranksShelves = dict(sorted(ranksShelves.items(), key=lambda item: item[1]))


        indice = 0


        pairShelves1 =  [[6,7,8,9],
                         [185,208,231,254],
                         [215,216,217,218],
                         [470,471,472,473],
                         [250,273,296,251]]

        resultsFP = FPGrowth(data,0.6,0.5)

        resultsFP = [list(s) for s in resultsFP]


        for shelves in pairShelves1:

            counter = 0

            for shelf in shelves:

                try:
                    sample = random.sample(resultsFP[counter],1)[0]
                    ranksShelves[shelf] = sample
                    aux.remove(sample)
                    auxRankShelves[shelf] = True
                except:
                    pass

                counter+=1



        #First place the products pairs
        for i in ranksShelves.keys():
            if i in shop.shoppingClass.config:
                if not auxRankShelves[i]:
                    ranksShelves[i] = int(aux[indice])
                    indice += 1


        dfAux = pd.Series(ranksShelves).to_frame()
        dfAux.to_csv("productPair.csv")


        return ranksShelves

def loadMain():
    if os.path.exists("productPair.csv"):
        dfAux = pd.read_csv("productPair.csv")

        shelves = dfAux['shelve'].to_numpy()
        products_aux = dfAux['product'].to_numpy()


        dictionary = {}
        for i, j in zip(shelves,products_aux):
            dictionary[i] = j

        return dict(collections.OrderedDict(sorted(dictionary.items())))

if __name__ == '__main__':

    all_config = []
    shelves = loadMain()

    sim = generateSimulator([])

    for s in shelves:

        if s in sim.shoppingClass.config:
            all_config.append(s)

    config = pd.read_csv("config.csv").to_numpy()[0][1:]


    #config = orderProductsPerImportanceAndProfit(sim)
    data = pickle.load(open("results/3.p", "rb"))[1][1]
    print(len(data))
    config = {all_config[i]: data[i] for i in range(len(all_config))}


    sim.shoppingClass.changeShoppingConfigCustom(config)

    sim.shoppingClass.plotImportance()

    customers = sim.generateCustomers(10000)

    sales = sim.simulateCustomers(customers)

    profit = sim.evaluateShoppingCost(sales)

    print(profit)


