import os
import pickle
import csv

import numpy as np

all = []
for i in os.listdir("results"):
    if ".py" not in i:
        print(i)
        all.append(pickle.load(open(f"results/{i}","rb")))

for el in all:
    print(el)

with open("profitConfig.csv", 'w', newline='') as myfile:

    lst = all[1][1][1]

    aux = []
    for i in lst:
        aux.append(i+1)

    wr = csv.writer(myfile)
    wr.writerow(aux)

with open("salesConfig.csv", 'w', newline='') as myfile:
    lst = all[3][1][1]

    aux = []
    for i in lst:
        aux.append(i + 1)

    wr = csv.writer(myfile)
    wr.writerow(aux)