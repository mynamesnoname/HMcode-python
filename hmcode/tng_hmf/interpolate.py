import numpy as np
from scipy import interpolate

data = np.loadtxt('/home/wbc/code/HMcode-python/hmcode/tng_hmf/BoverD.txt', dtype=str, skiprows=1)
arrays = list(zip(*data))
y = np.array(arrays[1], dtype=float)

nudata = np.loadtxt('/home/wbc/code/HMcode-python/hmcode/tng_hmf/nu.txt', skiprows=1)
nue = np.array(nudata, dtype=float)


def dep(nu):
    intp = interpolate.interp1d(nue, y, kind='linear')
    depress = []
    if isinstance(nu, np.ndarray) or isinstance(nu, list):
        for i in range(len(nu)):
            if nu[i] < nue.min() or nu[i] > nue.max():
                depress.append(1)
            else:
                depress.append(intp(nu[i]))
        depress = np.array(depress)
    elif nu < nue.min() or nu > nue.max():
        depress = 1
    else:
        depress = intp(nu)
        depress = float(depress)

    return depress