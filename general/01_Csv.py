import numpy as np
import csv

with open("../data/randomfile.csv") as file_name:
    array = np.loadtxt(file_name, delimiter=",")

print(f'Como array {array}')

with open("../data/randomfile.csv") as file_name:
    file_read = csv.reader(file_name)
    array = list(file_read)

print(f'Como lista {array}')
