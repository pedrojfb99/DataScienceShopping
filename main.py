

import pandas as pd
import json
import numpy as np
from sklearn.neighbors import KernelDensity

import os
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


PATH_TO_RECEIPTS = "E:/receipts/"

folders = os.listdir(PATH_TO_RECEIPTS)
aux_products = []


def encodeData():

    df = pd.read_csv('products.txt', delimiter="\t")
    dataHere = df['Nome'].str.strip()

    indexes = [x for x in range(0,len(dataHere))]

    df['ID'] = indexes
    #print(data.to_numpy())


    return df


products = encodeData()

def organize_data():


    all_receipts = []


    #For each folder#
    for i in tqdm(folders):

        #Path to this folder#
        path_to_this = PATH_TO_RECEIPTS + i + "/"

        try:
            #Every receipt in this folder#
            receipts = os.listdir(PATH_TO_RECEIPTS + i)
        except:
            continue
        #For each receipt#
        for receipt in receipts:
            if "._" in receipt:
                continue

            actual = []

            #Read the file#
            with open(path_to_this + receipt, "r",encoding="utf8") as f:
                #Receipt data#

                id = receipt.split("_")[1].split(".")[0]
                actual.append(id)
                rec = []

                lines = f.readlines()

                #Get the customer NIF#
                cNIF = lines[1].split("NIF")[1].strip()
                actual.append(cNIF)
                #print(cNIF)


                #Iterate through each product in the receipt#
                for line in lines[5:-2]:
                    aux = line.split(":")

                    name = aux[0].split(">")[1].strip()
                    #print(name)

                    pCode = products.loc[products['Nome'] == name, 'ID'].iloc[0]


                    rec.append(int(pCode))

                #Products bought
                actual.append(rec)

                #Total
                actual.append(lines[-1].split(":")[1].split("euro")[0].strip())

                all_receipts.append(actual)


    print(len(all_receipts))
    df = pd.DataFrame.from_records(all_receipts,columns=['receiptID','cNIF','PRODUCTS','TOTAL'])
    df.to_csv("data.csv")


def cvtCsvDataframe(input1, input2):

    dat = input1.to_numpy()

    #clean the list format
    for i in range(len(dat)):

        #convert the string list format to array
        dat[i][3] = json.loads(dat[i][3])

    dat2 = input2.to_numpy()
    #clean the list format
    for i in range(len(dat2)):

        #convert the string list format to array
        dat2[i][2] = json.loads(dat2[i][2].strip())

    #Return a dataframe with the organized data
    return pd.DataFrame.from_records(dat,columns=["ID",'receiptID','cNIF', 'PRODUCTS','TOTAL']), pd.DataFrame.from_records(dat2,columns=["ID",'receiptID', 'WISHLIST','DISTANCE'])



#Returns the product's name respective id
def fromIdtoName(id):

    return products.loc[products['ID'] == id, 'Nome'].iloc[0]

def Extract():
    encodeData()
    organize_data()

def FPGrowth(data, support):

    #Get all transactions
    receipts = data['PRODUCTS'].to_numpy()

    #One hot encoded the products' list
    te = TransactionEncoder()
    te_ary = te.fit(receipts).transform(receipts)
    oneHotProducts = pd.DataFrame(te_ary, columns=te.columns_)

    #Calculate the FPGrowth«
    print("FPGrowth results: \n")
    fp = fpgrowth(oneHotProducts, min_support=support)

    #Sort the values by support
    fp = fp.sort_values(by=['support'], ascending=False)

    print(fp)

    print("\n\n FPGrowth Interpretable Results")
    ids = fp['itemsets'].to_numpy()
    names = []

    #Convert each id to the respective name for interpretation
    for i in ids:
        itemset = list(i)
        new_itemset = []
        for item in itemset:
            new_itemset.append(fromIdtoName(item))
        names.append(new_itemset)

    fp.drop('itemsets',axis = 1,inplace=True)
    fp['itemsets'] = names

    print(fp)


def dataAnalytics(data):

    print("Data Describe")
    print("--------------------------")
    print(data.describe())
    print("--------------------------\n\n")

    print("Transactions sorted by total")
    print("--------------------------")
    ordered_total = data.sort_values(by=['TOTAL'], ascending=False)
    print(ordered_total.head())
    print("--------------------------\n\n")

    print("Transactions Grouped by Client Total Value to the supermarket")
    print("--------------------------")
    groupedClients = data.groupby(['cNIF']).sum().sort_values(by=['TOTAL'], ascending=False)
    print(groupedClients)
    print("--------------------------\n\n")









def obtainStaminaDistribution(staminaData):
    '''

    :param staminaData: Receives the stamina data from all the wishlists
    :return: Returns the stamina distribution of the shopping
    '''

    #Stamina histogram
    sns.distplot(staminaData, hist=True, kde=False, color='blue', hist_kws={'edgecolor': 'black'})
    plt.savefig("staminaHistogram.jpg")
    plt.clf()

    # seaborn histogram
    sns.distplot(staminaData, hist=True, kde=True, color='blue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 2})
    plt.savefig("staminaHistogramDensity.jpg")
    plt.clf()

    #Isolate the density
    sns.kdeplot(data=staminaData)
    plt.savefig("staminaDensity.jpg")

    staminaData = staminaData.reshape(-1, 1)
    densityKDE = KernelDensity(bandwidth=1.0, kernel="gaussian")

    densityKDE.fit(staminaData)

    return densityKDE


if __name__ == '__main__':

    #QoL for display
    pd.set_option('display.max_columns', 30)


    #organize_data()
    #Read the csv file and convert it to a well formatted dataframe
    data, explanations = cvtCsvDataframe(pd.read_csv("data.csv"), pd.read_csv("explanations.csv"))

    dataAnalytics(data)


    print(data.sort_values(by=['receiptID'], ascending=False))

    print(explanations.sort_values(by=['receiptID'], ascending=False))


    mergedReceiptExplanations= pd.merge(data,explanations,on='receiptID',how='outer')


    stamina = obtainStaminaDistribution(mergedReceiptExplanations['DISTANCE'].to_numpy())


    #Calculate FP growth algorithm for the receipts
    #FPGrowth(data, 0.5)



