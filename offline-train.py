import numpy as np

data = []
for i in range(1, 674):
    filename = 'state-'+str(i)+".npy"
    data.append(np.load(filename))

mx, my, mz = 0, 0, 0
for d in data:
    x, y, z = d.shape
    if x > mx:
        mx = x
    if y > my:
        my = y
    if z > mz:
        mz = z

