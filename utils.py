
from brian2 import PoissonGroup, SpikeMonitor, Network

def generate_input_spikes(neurons, frequency, period):
	# period is in milliseconds

	_gen = PoissonGroup(neurons, frequency)
	_mon = SpikeMonitor(_gen)
	_net = Network(_gen, _mon)
	_net.run(period)

	indices = _mon.i.get_item(slice(None, None, None))
	times = _mon.t.get_item(slice(None, None, None))
	return indices, times
