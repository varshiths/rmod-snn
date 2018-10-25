
from brian2 import *

from neurons import IF, IF_m
from neurons import Vr, Vt, tau

from synapses import model_stdp, action_prespike_stdp, action_postspike_stdp
from synapses import model_mstdp, action_prespike_mstdp, action_postspike_mstdp
from synapses import model_mstdpet, action_prespike_mstdpet, action_postspike_mstdpet
# from synapses import Apre, Apost, taupre, taupost, tauz
# from synapses import gamma0, gamma1, gamma2

from utils import generate_input_spikes, plot_rates

import matplotlib.pyplot as plt
import os

# np.set_printoptions(threshold=np.nan)
# set_device('cpp_standalone', build_on_run=False)

Apre = 1
Apost = -1
taupre = 20*ms
taupost = 20*ms

tauz = 25*ms

gamma0 = 0.2 * mV
gamma1 = 0.01 * mV
gamma2 = 0.25 * mV


class Experiment1:

	def __init__(self, args):


		if args.seed is not None:
			np.random.seed(args.seed)
			seed(args.seed)
			print(" Info: Seed set to {}".format(args.seed))
		else:
			print(" Warning: Consider setting seed to ensure reproducibility")
		defaultclock.dt = 1*ms
		
		print("Experiment 1: Learing XOR with temporally coded input")

		self.args = args

		self.define_network()
		self.restore_model()

		if args.test == 0:
			self.train()
		else:
			self.test()

		# device.build(directory='bin', compile=True, run=True, debug=False)

	def define_network(self):

		self.ilayer = SpikeGeneratorGroup(2, [], []*ms)
		self.hlayer = NeuronGroup(20, IF_m, threshold='v>Vt', reset='v=Vr', method='linear')
		self.olayer = NeuronGroup(1, IF_m, threshold='v>Vt', reset='v=Vr', method='linear')

		if self.args.rule == "mstdp":
			self.sih = Synapses(self.ilayer, self.hlayer, model=model_mstdp, on_pre=action_prespike_mstdp, on_post=action_postspike_mstdp)
			self.sho = Synapses(self.hlayer, self.olayer, model=model_mstdp, on_pre=action_prespike_mstdp, on_post=action_postspike_mstdp)
		else:
			self.sih = Synapses(self.ilayer, self.hlayer, model=model_mstdpet, on_pre=action_prespike_mstdpet, on_post=action_postspike_mstdpet)
			self.sho = Synapses(self.hlayer, self.olayer, model=model_mstdpet, on_pre=action_prespike_mstdpet, on_post=action_postspike_mstdpet)

		self.sih.connect()
		self.sho.connect()

		self.smon_sih = StateMonitor(self.sih, 'w', record=True)
		self.smon_sho = StateMonitor(self.sho, 'w', record=True)

		self.smon_hlayer = StateMonitor(self.hlayer, 'v', record=True)
		self.smon_olayer = StateMonitor(self.olayer, 'v', record=True)

		self.kmon_ilayer = SpikeMonitor(self.ilayer)
		self.kmon_hlayer = SpikeMonitor(self.hlayer)
		self.kmon_olayer = SpikeMonitor(self.olayer)

		self.network = Network(
				self.ilayer,
				self.hlayer,
				self.olayer,
				self.sih,
				self.sho,
				self.kmon_olayer,
			)

		if self.args.verbose:
			self.network.add(
					self.smon_sih,
					self.smon_sho,
					self.smon_hlayer,
					self.smon_olayer,
					self.kmon_hlayer,

					self.kmon_ilayer,
					
					# self.smon_olayer_r,
				)

		self.hlayer.v = Vr
		self.olayer.v = Vr
		self.hlayer.r = 1
		self.olayer.r = 1
		self.set_synapse_bounds()

	def set_synapse_bounds(self):

		self.sih.wmin = -10*mV
		self.sih.wmax = 10*mV

		self.sho.wmin = 0*mV
		self.sho.wmax = 10*mV
		
		# self.sih.wmin = 0*mV
		# self.sho.wmin = 0*mV

		# set half inhibitory and half excitatory for first layer
		# inhib = np.random.choice(30, 15, replace=False)
		# inhib = np.concatenate([inhib, 30 + np.random.choice(30, 15, replace=False)], axis=0)
		# extit = np.array( list(set(np.arange(60)).difference(set(inhib))) )

		# self.sih.wmin[extit, :] = 0*mV
		# self.sih.wmax[inhib, :] = 0*mV

	def initialize_weights(self):

		print("Initializing weights at random")

		# w1 = np.random.uniform(size=60*60)
		# w2 = np.random.uniform(size=60*1)

		w1 = (np.clip(np.random.randn(2*20), -1, 1)+1)/2
		w2 = (np.clip(np.random.randn(20*1), -1, 1)+1)/2

		self.sih.w = (self.sih.wmax-self.sih.wmin) * w1 + self.sih.wmin
		self.sho.w = (self.sho.wmax-self.sho.wmin) * w2 + self.sho.wmin

	def set_plasticity(self, plasticity=True):

		self.sih.plastic = plasticity
		self.sho.plastic = plasticity

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

	def gen_zero_one_rep(self, period):

		res = defaultclock.dt
		npoints = int( period / res )

		zero_rep = np.random.choice(npoints, 50, replace=False)
		one_rep = np.random.choice(npoints, 50, replace=False)

		return np.stack([zero_rep, one_rep], axis=0) * res

	def train(self):

		# import pdb; pdb.set_trace()

		print("Training")
		self.set_plasticity(True)
		# self.set_plasticity(False)

		rep01 = self.gen_zero_one_rep(500*ms)

		x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
		y = np.array([0, 1, 1, 0])

		order = np.arange(x.shape[0])

		try:

			indices = np.concatenate( [np.zeros(rep01[0].size), np.ones(rep01[1].size)], axis=0 )
			for i in range(1, self.args.nepochs+1):

				print("Epoch: {eno:d}".format(eno=i))

				np.random.shuffle(order)
				for j, index in enumerate(order):

					times = np.reshape( rep01[x[index]], (-1))		
					times += self.network.t

					# if self.args.verbose:
					# 	print(" .{} Spike Range: {}/{}".format(j, np.min(times) if times.size != 0 else None, np.max(times) if times.size != 0 else None))
					# 	print(" .{} Indices Range: {}/{}".format(j, np.min(indices) if indices.size != 0 else None, np.max(indices) if indices.size != 0 else None))

					# import pdb; pdb.set_trace()

					self.ilayer.set_spikes(indices, times, sorted=False)
					self.hlayer.r = y[index]*2-1
					self.olayer.r = y[index]*2-1

					self.network.run(500*ms)

				if self.args.verbose:
					print("Total Spikes: {} {} {}".format(self.kmon_ilayer.num_spikes, self.kmon_hlayer.num_spikes, self.kmon_olayer.num_spikes))
					print("Max/Min Weights: {}/{} {}/{}".format(np.min(self.smon_sih.w[:, -1]), np.max(self.smon_sih.w[:, -1]), np.min(self.smon_sho.w[:, -1]), np.max(self.smon_sho.w[:, -1])))
					print("H Voltage:", np.mean(self.smon_hlayer.v[:, -20000:]))
					print("Output Voltage:", np.mean(self.smon_olayer.v[0, -20000:]))

				# import pdb; pdb.set_trace()

				if i % self.args.nepochs_per_save == 0:
					self.save_model()
			
		except KeyboardInterrupt as e:
			print("Training Interrupted. Refer to model saved in {}".format(self.args.model))

	def test(self):

		# import pdb; pdb.set_trace()

		print("Testing")
		self.set_plasticity(False)
		rate = np.zeros((self.args.nepochs, 4))*Hz

		for q in range(self.args.nepochs):
			
			rep01 = self.gen_zero_one_rep(500*ms)

			# if self.args.verbose:
			# 	print(rep01)

			x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
			y = np.array([0, 1, 1, 0])

			indices = np.concatenate( [np.zeros(rep01[0].size), np.ones(rep01[1].size)], axis=0 )
			for j, _input in enumerate(x):
				start = self.network.t

				times = np.reshape( rep01[_input], (-1))		
				times += start

				self.ilayer.set_spikes(indices, times)
				self.network.run(500*ms)

				end = self.network.t

				# import pdb; pdb.set_trace()

				spikes = self.kmon_olayer.t
				spikes = spikes[spikes <= end]
				spikes = spikes[start <= spikes]

				rate[q, j] =  float(spikes.size) / (500.0*ms)

			print("Rates:", rate[q])

		plot_rates(rate, "outputs/exp{}_{}_rates.png".format(1, self.args.rule))
