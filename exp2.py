
from brian2 import *

from neurons import IF, IF_r
from neurons import Vr, Vt, tau

from synapses import model_mstdp, action_prespike_mstdp, action_postspike_mstdp
from synapses import model_mstdpet, action_prespike_mstdpet, action_postspike_mstdpet

from synapses import action_postspike_mstdp_ri, action_postspike_mstdpet_ri
# from synapses import Apre, Apost, taupre, taupost, tauz
# from synapses import gamma0, gamma1, gamma2

from utils import generate_input_spikes, rms

import matplotlib.pyplot as plt
import os

# np.set_printoptions(threshold=np.nan)
# set_device('cpp_standalone', build_on_run=False)

EPSILON = 1

Apre = 1
Apost = -1
taupre = 20*ms
taupost = 20*ms

tauz = 25*ms

gamma0 = 0.2 * mV
gamma1 = 1.0 * mV
gamma2 = 0.05 * mV

tauv = 2*second

class Experiment2:

	def __init__(self, args):


		if args.seed is not None:
			np.random.seed(args.seed)
			seed(args.seed)
			print(" Info: Seed set to {}".format(args.seed))
		else:
			print(" Warning: Consider setting seed to ensure reproducibility")
		defaultclock.dt = 1*ms
		
		print("Experiment 2: Learing target rate coding")

		self.args = args

		self.train()

		# device.build(directory='bin', compile=True, run=True, debug=False)

	def define_network(self, irates):

		self.ilayer = PoissonGroup(100, irates)
		self.olayer = NeuronGroup(100, IF_r, threshold='v>Vt', reset='v=Vr', method='linear')

		if self.args.rule == "mstdp":
			self.sio = Synapses(self.ilayer, self.olayer, model=model_mstdp, on_pre=action_prespike_mstdp, on_post=action_postspike_mstdp_ri)
		else:
			self.sio = Synapses(self.ilayer, self.olayer, model=model_mstdpet, on_pre=action_prespike_mstdpet, on_post=action_postspike_mstdpet_ri)

		self.sio.connect()

		self.smon_sio = StateMonitor(self.sio, 'w', record=True)

		self.smon_olayer = StateMonitor(self.olayer, 'v', record=True)

		self.kmon_ilayer = SpikeMonitor(self.ilayer)
		self.kmon_olayer = SpikeMonitor(self.olayer)

		self.network = Network(
				self.ilayer,
				self.olayer,
				self.sio,
				self.kmon_olayer,
			)

		if self.args.verbose:
			self.network.add(
					self.smon_sio,
					self.smon_olayer,

					self.kmon_ilayer,
					self.kmon_olayer,
					
					# self.smon_olayer_r,
				)

		self.olayer.v = Vr
		self.olayer.r = 1
		self.olayer.k = 0*Hz
		self.set_synapse_bounds()

	def set_synapse_bounds(self):

		self.sio.wmin = 0*mV
		self.sio.wmax = 1.25*mV
		
	def initialize_weights(self):

		print("Initializing weights at random")

		w1 = np.random.uniform(size=100*100)

		# w1 = (np.clip(np.random.randn(100*100), -1, 1)+1)/2

		self.sio.w = (self.sio.wmax-self.sio.wmin) * w1 + self.sio.wmin

	def set_plasticity(self, plasticity=True):

		self.sio.plastic = plasticity

		# self.sih.plastic = False
		# self.sho.plastic = False

	def restore_model(self):

		if os.path.isfile(self.args.model):
			self.network.restore(filename=self.args.model)
		else:
			self.initialize_weights()
			self.save_model()

	def save_model(self):

		self.network.store(filename=self.args.model)
		print("Saved Model in {}".format(self.args.model))

	def sample_rate(self, N, llim, ulim):

		rates = np.random.uniform(size=N)*(ulim-llim) + llim
		return rates

	def train(self):

		# import pdb; pdb.set_trace()

		irates = self.sample_rate(100, 0*Hz, 50*Hz)
		orates = self.sample_rate(100, 20*Hz, 100*Hz)

		self.define_network(irates)
		self.restore_model()

		self.set_plasticity(False)
		# commenced = False
		doter = np.ones(self.args.nepochs)*-1*Hz
		self.network.run(defaultclock.dt*50)
		self.set_plasticity(True)

		try:

			for i in range(1, self.args.nepochs+1):

				print("Epoch: {eno:d}".format(eno=i), self.network.t)
				# import pdb; pdb.set_trace()

				kt_1 = self.olayer.k
				dot_1 = rms(self.olayer.k, orates)

				self.network.run(defaultclock.dt)
				# self.network.run(defaultclock.dt*100)
				
				kt = self.olayer.k
				dot = rms(self.olayer.k, orates)

				self.olayer.r = np.sign(dot_1-dot)

				# if not commenced and rms(kt, kt_1) < EPSILON*Hz:
				# 	commenced = False
				# 	self.set_plasticity(True)

				doter[i-1] = dot

				if self.args.verbose:
					print("Spikes: {} {}".format(self.kmon_ilayer.num_spikes, self.kmon_olayer.num_spikes))
					print("Distance:", dot)

					# import pdb; pdb.set_trace()
					print("Synapses:", np.mean(self.smon_sio.w[:, -1]))

				if i % (self.args.nepochs_per_save*100) == 0:
					self.save_model()
			
		except KeyboardInterrupt as e:
			print("Training Interrupted. Refer to model saved in {}".format(self.args.model))
