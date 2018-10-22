
from brian2 import PoissonGroup, SpikeMonitor, Network
from brian2 import sqrt, sum, square, Hz

def generate_input_spikes(neurons, frequency, period):
	# period is in milliseconds

	_gen = PoissonGroup(neurons, frequency)
	_mon = SpikeMonitor(_gen)
	_net = Network(_gen, _mon)
	_net.run(period)

	indices = _mon.i.get_item(slice(None, None, None))
	times = _mon.t.get_item(slice(None, None, None))
	return indices, times

def rms(a, b):
	return sqrt(sum(square(a-b)))
