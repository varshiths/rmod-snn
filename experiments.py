
from brian2 import *

from neurons import IF, IF_out
from neurons import Vr, Vt, tau

from synapses import model_stdp, action_prespike_stdp, action_postspike_stdp
from synapses import model_mstdp, action_prespike_mstdp, action_postspike_mstdp
from synapses import Apre, Apost, taupre, taupost, gamma, tauz

from utils import generate_input_spikes

import matplotlib.pyplot as plt
import os

np.random.seed(0)
# np.set_printoptions(threshold=np.nan)

class Experiment0:

	def __init__(self, args):

		print("Experiment 0: Learing XOR with rate coded input")

		self.args = args
		
		self.define_network()
		self.restore_model()

		if args.test == 0:
			self.train()
		else:
			self.test()

	def define_network(self):

		self.ilayer = ilayer = SpikeGeneratorGroup(60, [], []*ms)
		self.hlayer = NeuronGroup(60, IF, threshold='v>Vt', reset='v=Vr', method='exact')
		self.olayer = NeuronGroup(1, IF_out, threshold='v>Vt', reset='v=Vr', method='exact')

		self.sih = Synapses(ilayer, self.hlayer, model=model_stdp, on_pre=action_prespike_stdp)
		self.sho = Synapses(self.hlayer, self.olayer, model=model_mstdp, on_pre=action_prespike_mstdp)

		self.sih.connect()
		self.sho.connect()

		self.smon_hlayer = StateMonitor(self.hlayer, 'v', record=True)
		self.smon_olayer = StateMonitor(self.olayer, 'v', record=True)

		self.kmon_hlayer = SpikeMonitor(self.hlayer)
		self.kmon_olayer = SpikeMonitor(self.olayer)

		self.network = Network(
				self.ilayer,
				self.hlayer,
				self.olayer,
				self.sih,
				self.sho,
				self.smon_hlayer,
				self.smon_olayer,
				self.kmon_hlayer,
				self.kmon_olayer,
			)

		self.hlayer.v = Vr
		self.olayer.v = Vr
		self.set_synapse_bounds()

	def set_synapse_bounds(self):

		self.sih.wmin = -5*mV
		self.sih.wmax = 5*mV

		self.sho.wmin = -5*mV
		self.sho.wmax = 5*mV

		# set half inhibitory and half excitatory for first layer
		inhib = np.random.choice(30, 15, replace=False)
		inhib = np.concatenate([inhib, 30 + np.random.choice(30, 15, replace=False)], axis=0)
		extit = np.array( list(set(np.arange(60)).difference(set(inhib))) )

		self.sih.wmin[extit, :] = 0*mV
		self.sih.wmax[inhib, :] = 0*mV

	def initialize_weights(self):

		print("Initializing weights at random")
		w1 = np.random.uniform(size=60*60)
		w2 = np.random.uniform(size=60*1)

		self.sih.w = (self.sih.wmax-self.sih.wmin) * w1 + self.sih.wmin
		self.sho.w = (self.sho.wmax-self.sho.wmin) * w2 + self.sho.wmin

	def set_plasticity(self, plastic=True):

		self.sih.plastic = plastic
		self.sho.plastic = plastic

	def restore_model(self):

		if os.path.isfile(self.args.model):
			self.network.restore(filename=self.args.model)
		else:
			self.initialize_weights()
			self.save_model()

	def save_model(self):

		self.network.store(filename=self.args.model)
		print("Saved Model in {}".format(self.args.model))

	def train(self):

		# import pdb; pdb.set_trace()

		print("Training")
		self.set_plasticity(True)

		x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
		y = np.array([0, 1, 1, 0])

		order = np.arange(x.shape[0])

		try:

			for i in range(1, self.args.nepochs+1):

				print("Epoch: {eno:d}".format(eno=i))

				np.random.shuffle(order)
				for j, index in enumerate(order):

					print("\tBatch: {bno:d}".format(bno=j))

					indices, times = generate_input_spikes(30*np.sum(x[index]), 40*Hz, 500*ms)
					times += i*2000*ms + j*500*ms
					if (x[index] == [0, 1]).all():
						indices += 30

					self.ilayer.set_spikes(indices, times)
					self.olayer.r = y[index]*2-1

					self.network.run(500*ms)

				if i % self.args.nepochs_per_save == 0:
					self.save_model()
			
		except KeyboardInterrupt as e:
			print("Training Interrupted. Refer to model saved in {}".format(self.args.model))

	def test(self):

		P = PoissonGroup(100, np.arange(100)*Hz + 10*Hz)
		spike_mon = SpikeMonitor(P)

		run(100*ms)

		import pdb; pdb.set_trace()

		print(spike_mon.i, spike_mon.t)
