
# from utils import generate_input_spikes
from brian2 import *
# import numpy as np

# ranges = [
# 	[-2.4, 2.4],
# 	[-inf, inf],
# 	[-41.8 rad, 41.8 rad],
# 	[-inf, inf],
# ]

ranges = [
	[-1.5, 1.5],
	[-1.5, 1.5],
	[-0.2, 0.2],
	[-2, 2],
]
# ress = [
# 	20,
# 	20,
# 	20,
# 	20,
# ]
# res19 = np.arange(19)
ress = [
	5,
	5,
	5,
	5,
]
res4 = np.arange(4)
done = False

def generate_input_spikes_from_encoding(observation, size, freq, time):

	# global done
	# if not done:
	# import pdb; pdb.set_trace()
	# 	done = True

	indices, times = generate_input_spikes(4, freq, time)
	indices += sum(ress)

	offset = 0
	for i, obs in enumerate(observation):
		ind = get_index(obs, ranges[i], ress[i])
		indices[indices == sum(ress) + i] = ind + offset
		offset += ress[i]

	return indices, times

def get_index(obs, rang, res):
	# return np.searchsorted(res19*(rang[1]-rang[0])/(res-2) + rang[0], obs)
	return np.searchsorted(res4*(rang[1]-rang[0])/(res-2) + rang[0], obs)

def generate_input_spikes(size, freq, time, dt=1*ms):
	dat = np.array(np.where(np.random.uniform(size=[size, int(time / dt) ]) < freq*dt))
	inds = np.argsort(dat[1, :])
	return dat[0][inds], dat[1][inds]*ms
