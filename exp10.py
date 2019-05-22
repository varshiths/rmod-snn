
from brian2 import *

def generate_input_spikes(neurons, frequency, period):
	_gen = PoissonGroup(neurons, frequency)
	_mon = SpikeMonitor(_gen)
	_net = Network(_gen, _mon)
	_net.run(period)

	indices = _mon.i.get_item(slice(None, None, None))
	times = _mon.t.get_item(slice(None, None, None))
	return indices, times

N = 2
ilayer = SpikeGeneratorGroup(N, [], []*ms)
network = Network(
		ilayer,
	)

for j in range(2):
	
	print(j)

	indices, times = generate_input_spikes(N, 40*Hz, 500*ms)
	# indices = np.arange(N)
	# times = indices * 500*ms / N

	# times += network.t

	ilayer.set_spikes(indices, times)
	network.run(500*ms)
