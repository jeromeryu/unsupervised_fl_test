from sys import argv
from matplotlib import pyplot as plt
import os
import csv
from datetime import date, datetime

file_list = []
for i in range(1, len(argv)):
    file_list.append(argv[i])

color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

x = []
for idx, path in enumerate(file_list):
    filename = os.path.basename(path)
    acc1 = []
    with open(path) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i==0:
                continue
            acc1.append(float(row[5]))
    if idx==0:
        x = range(1, len(acc1)+1) 
    print()
    plt.plot(x, acc1, color = color_list[idx], label = filename.split('_')[0]) 

time = datetime.now()
res_path = datetime.now().strftime('%Y%m%d%H%M%S_result.png')

plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

print(res_path)
plt.savefig(res_path)
