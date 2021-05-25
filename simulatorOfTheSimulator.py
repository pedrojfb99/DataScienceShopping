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

from shopping import Shopping, Cell
import main

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

        self.shoppingClass = Shopping([23,21])

        print(len(configuration))

        #self.shoppingClass.changeShoppingConfig(configuration)

        self.shopping = self.shoppingClass.shopping

        self.staminaDistr = staminaDistr

        self.explanations = explanations


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

        :param currentCell: Receives the current shopping cell
        :return: Return the distance to the closest neighbor
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

                    pathsToThisCell = self.findNeighbors([i,j],1)

                    for s in pathsToThisCell: allPathsToItem.append(s)

        pathsLenght = []
        paths = []


        for possiblePath in allPathsToItem:
                paths.append(nx.dijkstra_path(self.shoppingClass.graphShopping, self.shoppingClass.entrance, possiblePath))
                pathsLenght.append(len(nx.dijkstra_path(self.shoppingClass.graphShopping, self.shoppingClass.entrance, possiblePath)))



        #Return the minimium path
        return paths[np.argmin(pathsLenght)]



    def getCellProducts(self, cell):

        size = self.shopping.shape

        for j in range(size[1]):

            for i in range(size[0]):

                if self.shopping[i][j].id == cell:

                    cells = self.findNeighbors([i,j],2)

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
            data, explanations = cvtCsvDataframe(pd.read_csv("data.csv"), pd.read_csv("explanations.csv"))

            mergedReceiptExplanations = pd.merge(data, explanations, on='receiptID', how='outer')

            boughtAndWishlist = mergedReceiptExplanations[['PRODUCTS', 'WISHLIST']].to_numpy()

            aux = {}

            #For each receipt
            for p in tqdm(boughtAndWishlist):

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
            print(currentWishlist)
            currentWishlist.reverse()
            print(currentWishlist)
            exit()
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


                    try:
                        time.sleep(1)
                        #print(f"Walking to cell {closest[cell + 1]}")
                    except:
                        pass

                    currentStamina -= 1
                    #print(f"Current stamina : {currentStamina}")

                    #Scenarios that the person leaves the shopping
                    if currentStamina <= 0:
                        #print("I got tired!")
                        break
                    elif len(currentWishlist) == 0:
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

        print("Calculating profit ...")
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


def testConfigs():


    config = pd.read_csv("config.csv").to_numpy()[0][1:]

    sim = generateSimulator(config)

    customers = sim.generateCustomers(5000)

    sales = sim.simulateCustomers(customers)

    profit = sim.evaluateShoppingCost(sales)



    print(profit)
    print(products)




if __name__ == '__main__':


    testConfigs()

