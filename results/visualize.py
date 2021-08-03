from sys import argv
from matplotlib import pyplot as plt
import os
import csv

res_path = argv[1]
filename = os.path.basename(res_path)
acc1 = []
acc5 = []
# with open(res_path) as f:
#     lines = f.readlines()
#     for line in lines:
#         l = line.split(" ")
#         acc1.append(float(l[6][:-1]))
#         acc5.append(float(l[9]))
with open(res_path) as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i==0:
            continue
        acc1.append(float(row[5]))
        acc5.append(float(row[6]))
        

x = []
for i, a in enumerate(acc1):
    x.append(i+1)
plt.plot(x, acc1, color='g', label='acc1')
plt.plot(x, acc5, color='r', label='acc5')
plt.xlabel('round')
plt.ylabel('accuracy')
plt.legend()

plt.title(filename.split('_')[0])
# plt.savefig('result.png')
plt.savefig(res_path.replace('.csv', '.png'))
