
import sys

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set()

def smooth(x,window_len=100,window='hanning'):

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

# xlimits=1e7
# ylimits=2000

res1 = np.loadtxt(sys.argv[1])

plt.figure()
x = np.arange(res1.size)
y = np.cumsum(res1)
plt.plot(x, y)
plt.xlabel("Number of Episodes")
plt.ylabel("Cumulative Reward")

axes = plt.gca()
# axes.set_xlim([0,xlimits])
axes.set_ylim([-100,2400])
plt.show()
