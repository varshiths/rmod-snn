
from brian2 import *

from neurons import IF, IF_m
from neurons import Vr, Vt, tau

from synapses import model_stdp, action_prespike_stdp, action_postspike_stdp
from synapses import model_mstdp, action_prespike_mstdp, action_postspike_mstdp
from synapses import model_mstdpet, action_prespike_mstdpet, action_postspike_mstdpet
# from synapses import Apre, Apost, taupre, taupost, tauz
# from synapses import gamma0, gamma1, gamma2

from cartpole import generate_input_spikes_from_encoding

import matplotlib.pyplot as plt
import os

import gym

# np.set_printoptions(threshold=np.nan)
# set_device('cpp_standalone', build_on_run=False)

Apre = 1
Apost = -1
taupre = 20*ms
taupost = 20*ms

tauz = 25*ms

gamma0 = 0.2 * mV
gamma1 = 0.1 * mV
gamma2 = 0.8 * mV

_LEARNING_STARTS = 15

class Experiment3:

	def __init__(self, args):


		if args.seed is not None:
			np.random.seed(args.seed)
			seed(args.seed)
			print(" Info: Seed set to {}".format(args.seed))
		else:
			print(" Warning: Consider setting seed to ensure reproducibility")
		defaultclock.dt = 1*ms
		
		print("Experiment 3: Learning CartPole-v0")

		self.args = args

		self.define_network()
		self.restore_model()

		if args.test == 0:
			self.train()
		else:
			self.test()

		# device.build(directory='bin', compile=True, run=True, debug=False)

	def define_network(self):

		self._input_size = 80
		self.ilayer = SpikeGeneratorGroup(self._input_size, [], []*ms)
		self.hlayer = NeuronGroup(80, IF_m, threshold='v>Vt', reset='v=Vr', method='linear')
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

		# self.smon_hlayer = StateMonitor(self.hlayer, 'v', record=True)
		self.smon_olayer = StateMonitor(self.olayer, 'v', record=True)

		# self.kmon_ilayer = SpikeMonitor(self.ilayer)
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
					# self.smon_sih,
					# self.smon_sho,

					# self.smon_hlayer,

					# self.smon_olayer,
					self.kmon_hlayer,

					# self.kmon_ilayer,
					
					# self.smon_olayer_r,
				)

		self.hlayer.v = Vr
		self.olayer.v = Vr
		self.hlayer.r = 1
		self.olayer.r = 1
		self.set_synapse_bounds()

	def set_synapse_bounds(self):

		param = 50

		self.sih.wmin = -param*mV
		self.sih.wmax = param*mV

		self.sho.wmin = 0*mV
		# self.sho.wmin = -param*mV
		self.sho.wmax = param*mV
		
		# self.sih.wmin = 0*mV
		# self.sho.wmin = 0*mV

		# set half inhibitory and half excitatory for first layer
		inhib = np.random.choice(self._input_size // 2, self._input_size // 4, replace=False)
		inhib = np.concatenate([inhib, self._input_size // 2 + np.random.choice(self._input_size // 2, self._input_size // 4, replace=False)], axis=0)
		extit = np.array( list(set(np.arange(self._input_size)).difference(set(inhib))) )

		self.sih.wmin[extit, :] = 0*mV
		self.sih.wmax[inhib, :] = 0*mV

	def initialize_weights(self):

		print("Initializing weights at random")

		# w1 = np.random.uniform(size=60*60)
		# w2 = np.random.uniform(size=60*1)

		w1 = (np.clip(np.random.randn(self._input_size*self._input_size), -1, 1)+1)/2
		w2 = (np.clip(np.random.randn(self._input_size*1), -1, 1)+1)/2

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

	def train(self):

		# import pdb; pdb.set_trace()

		print("Training")
		self.set_plasticity(True)
		# self.set_plasticity(False)

		env = gym.make('CartPole-v0')

		try:

			episode_rewards = []

			for eno in range(1, self.args.nepochs+1):
				print("Episode: {eno:d}".format(eno=eno))

				done = False
				observation = env.reset()
				trews = 0

				while not done:

					# print("Observation:", observation)
					indices, times = generate_input_spikes_from_encoding(observation, self._input_size, 40*Hz, 500*ms)

					# inference phase
					self.set_plasticity(False)

					currt = self.network.t
					itimes = times + self.network.t

					self.ilayer.set_spikes(indices, itimes, sorted=True)
					self.network.run(500*ms)

					# import pdb; pdb.set_trace()

					ofreq = np.sum(self.kmon_olayer.t > currt) / (500*ms)
					action = 1 if ofreq > 50*Hz else 0
					# action = np.random.randint(2)

					observation, reward, done, info = env.step(action)
					# reward = 0
					reward = 1 if not done else -1

					if self.args.verbose:
						print(ofreq, action, reward)

					# training phase after a few episodes
					if eno > _LEARNING_STARTS:

						self.set_plasticity(True)

						itimes = times + self.network.t
						self.ilayer.set_spikes(indices, itimes, sorted=True)
						self.hlayer.r = reward
						self.olayer.r = reward

						self.network.run(500*ms)

					# cumulative reward
					trews += reward

					if self.args.verbose:

						env.render()
						# print("Total Spikes: {} {} {}".format(self.kmon_ilayer.num_spikes, self.kmon_hlayer.num_spikes, self.kmon_olayer.num_spikes))
						print("Total Spikes: {} {}".format(self.kmon_hlayer.num_spikes, self.kmon_olayer.num_spikes))
						print("Mean Spiking Rate:", self.kmon_olayer.num_spikes / self.network.t)
						# print("Max/Min Weights: {}/{} {}/{}".format(np.min(self.smon_sih.w[:, -1]), np.max(self.smon_sih.w[:, -1]), np.min(self.smon_sho.w[:, -1]), np.max(self.smon_sho.w[:, -1])))
						# print("H Voltage:", np.mean(self.smon_hlayer.v[:, -20000:]))
						# print("Output Voltage:", np.mean(self.smon_olayer.v[0, -20000:]))

				# import pdb; pdb.set_trace()
				print("Total Episode Reward:", trews)
				episode_rewards.append(trews)

				if eno % self.args.nepochs_per_save == 0:
					self.save_model()
			np.savetxt("outputs/exp3_episode_rewards.csv", episode_rewards)
			
		except KeyboardInterrupt as e:
			print("Training Interrupted. Refer to model saved in {}".format(self.args.model))

	def test(self):

		# import pdb; pdb.set_trace()

		print("Testing")
		self.set_plasticity(False)

		x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
		y = np.array([0, 1, 1, 0])
		rate = np.zeros((self.args.nepochs, y.shape[0]))*Hz

		for q in range(self.args.nepochs):

			# print("Run {}".format(q))

			for j, _input in enumerate(x):
				start = self.network.t

				indices, times = generate_input_spikes(30*np.sum(_input), 40*Hz, 500*ms)
				times += start
				if (_input == [0, 1]).all():
					indices += 30

				self.ilayer.set_spikes(indices, times)
				self.network.run(500*ms)

				end = self.network.t

				# import pdb; pdb.set_trace()

				spikes = self.kmon_olayer.t
				spikes = spikes[spikes <= end]
				spikes = spikes[start <= spikes]

				rate[q, j] =  spikes.size / (500*ms)

			print("Rate", rate[q])

		plot_rates(rate, "outputs/exp{}_{}_rates.png".format(0, self.args.rule))
