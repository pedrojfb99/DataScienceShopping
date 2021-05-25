import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import glob
import pandas as pd


PATH_TO_RECEIPTS = "E:/explanations/"

folders = os.listdir(PATH_TO_RECEIPTS)


def encodeData():

    df = pd.read_csv('products.txt', delimiter="\t")
    data = df['Nome'].str.strip()

    indexes = [x for x in range(0,len(data))]

    df['ID'] = indexes
    #print(data.to_numpy())


    return df

productsLoaded = encodeData()


def findLastProduct(data):

    i = 0

    for k in data:
        if "-" in k and "Looking" not in k and "I bought" not in k and "Walked" not in k:
            i +=1

    return i + 1

def organizeData():
    counter = 0
    all_receipts = []

    # For each folder#
    for i in tqdm(folders):

        # Path to this folder#
        path_to_this = PATH_TO_RECEIPTS + i + "/"

        try:
            # Every receipt in this folder#
            receipts = os.listdir(PATH_TO_RECEIPTS + i)
        except:
            continue
        # For each receipt#
        for receipt in receipts:
            if "._" in receipt:
                continue

            actual = []

            # Read the file#
            with open(path_to_this + receipt, "r", encoding="utf8") as f:

                # Receipt data#
                id = receipt.split("_")[1].split(".")[0]
                actual.append(id)


                lines = np.asarray(f.readlines())

                prods = findLastProduct(lines[1:])
                products = lines[1:prods]
                aux = []
                for p in range(len(products)):

                    name = products[p].split("- ")[1].strip().strip("\n")

                    pCode = productsLoaded.loc[productsLoaded['Nome'] == name, 'ID'].iloc[0]

                    aux.append(int(pCode))


                #Append wishlist
                actual.append(aux)

                #Append the total distance traveled
                distanceTraveled = ' '.join(map(str, lines[prods:])).split("Walked ")[-1].split(".")[0]

                actual.append(distanceTraveled)

                all_receipts.append(actual)


    df = pd.DataFrame.from_records(all_receipts, columns=['receiptID', 'WISHLIST', 'DISTANCE'])
    df.to_csv("explanations.csv")

if __name__ == '__main__':

    organizeData()

    os.chdir("./")
