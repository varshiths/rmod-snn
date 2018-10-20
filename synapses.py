
from brian2.units import *

# pre -> + and post -> -
# Apre = 1
# Apost = -1
# taupre = 20*ms
# taupost = 20*ms

# tauz = 25*ms

# gamma0 = 0.2 * mV
# gamma1 = 0.1 * mV
# gamma2 = 0.625 * mV

model_stdp = '''
plastic : boolean (shared)
w : volt
wmin : volt
wmax : volt
dapre/dt = int(plastic)*-1*apre/taupre : 1 (event-driven)
dapost/dt = int(plastic)*-1*apost/taupost : 1 (event-driven)
'''
# dapre/dt = -1*apre/taupre : volt (event-driven)
# dapost/dt = -1*apost/taupost : volt (event-driven)
action_prespike_stdp = '''
v_post += w
apre += int(plastic)*Apre
w = clip( w + int(plastic)*gamma0*apost, wmin, wmax )
'''
# v_post -= w
# apre += Apre
# w = clip( w + apost, wmin, wmax )
action_postspike_stdp = '''
apost += int(plastic)*Apost
w = clip( w + int(plastic)*gamma0*apre, wmin, wmax )
'''
# apost += Apost
# w = clip( w + apre, wmin, wmax )

model_mstdp = '''
plastic : boolean (shared)
w : volt
wmin : volt
wmax : volt
dPpre/dt = int(plastic)*-Ppre/taupre : 1 (event-driven)
dPpost/dt = int(plastic)*-Ppost/taupost : 1 (event-driven)
'''
# dPpre/dt = -Ppre/taupre : volt (event-driven)
# dPpost/dt = -Ppost/taupost : volt (event-driven)
action_prespike_mstdp = '''
v_post += w
Ppre += int(plastic)*Apre
w = clip( w + int(plastic)*gamma1*r_post*Ppost, wmin, wmax)
'''
# Ppre += Apre
# w = clip( w + gamma1*r_post*Ppost, wmin, wmax)
action_postspike_mstdp = '''
Ppost += int(plastic)*Apost
w = clip( w + int(plastic)*gamma1*r_post*Ppre, wmin, wmax)
'''
# Ppost += Apost
# w = clip( w + gamma1*r_post*Ppre, wmin, wmax)

# model_mstdpet = '''
# plastic : boolean (shared)
# w : volt
# wmin : volt
# wmax : volt
# dz/dt = int(plastic)*-z/tauz : 1 (event-driven)
# dPpre/dt = int(plastic)*-Ppre/taupre : 1 (event-driven)
# dPpost/dt = int(plastic)*-Ppost/taupost : 1 (event-driven)
# '''
# action_prespike_mstdpet = '''
# v_post += w
# Ppre += int(plastic)*Apre
# z += int(plastic)*Ppost
# w = clip( w + int(plastic)*gamma2*r_post*z, wmin, wmax)
# '''
# action_postspike_mstdpet = '''
# Ppost += int(plastic)*Apost
# z += int(plastic)*Ppre
# w = clip( w + int(plastic)*gamma2*r_post*z, wmin, wmax)
# '''
