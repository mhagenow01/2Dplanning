import matplotlib.pyplot as plt
import numpy as np

with open('output.csv') as fin:
    data = list(map(float, fin.readline().split(',')[:-1]))

with open('trajectory.csv') as fin:
    traj = list(map(lambda r: list(map(float, r.split(','))), list(fin)))

data = np.array(data)
s = int(np.sqrt(len(data)))
data = data.reshape((s,s))

for x in traj:
    _x, _y = x
    _x = int(_x)
    _y = int(_y)
    data[_x-5:_x+5,_y-5:_y+5] = -100


plt.imshow(data[5:-5,5:-5])
plt.savefig("show.png")