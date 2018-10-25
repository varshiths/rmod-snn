
# from brian2 import PoissonGroup, SpikeMonitor, Network
# from brian2 import sqrt, sum, square, Hz

from brian2 import *
import matplotlib.pyplot as plt

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

def plot_rates(rate, filename):

	rate = rate / Hz

	x_pos = np.arange(rate.shape[1])
	means = np.mean(rate, axis=0)
	error = np.std(rate, axis=0)

	# Build the plot
	fig, ax = plt.subplots()
	ax.bar(x_pos, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
	ax.set_ylabel('Frequency of Output Spikes (Hz)')
	ax.set_xticks(x_pos)
	ax.set_xticklabels(["[0,0]", "[0,1]", "[1,0]", "[1,1]"])
	ax.set_title('Spiking Frequency of Output Neuron for various inputs')
	ax.yaxis.grid(True)

	# Save the figure and show
	plt.tight_layout()
	plt.savefig(filename)
	print("Output:", filename)

def plot_cum_reward(rewards, dists, f1, f2):

	plt.figure()

	tis = np.arange(rewards.size)/1000.0
	rc = np.cumsum(rewards)

	plt.plot(tis, rc)
	plt.title("Cumulative Reward obtained")
	plt.xlabel("time (s)")
	plt.ylabel("Total Reward ()");

	plt.tight_layout()
	plt.savefig(f1)
	print("Output:", f1)

	plt.figure()

	rc = dists

	plt.plot(tis, dists)
	plt.title("Distance between current and target output rate pattern")
	plt.xlabel("time (s)")
	plt.ylabel("Distance (Hz)");
	
	plt.tight_layout()
	plt.savefig(f2)
	print("Output:", f2)

