
from brian2.units import *

# pre -> + and post -> -
Apre = 1*volt
Apost = -1*volt
taupre = 20*ms
taupost = 20*ms

model_stdp = '''
plastic : boolean (shared)
w : volt
wmin : volt
wmax : volt
dapre/dt = int(plastic)*-1*apre/taupre : volt (event-driven)
dapost/dt = int(plastic)*-1*apost/taupost : volt (event-driven)
'''
action_prespike_stdp = '''
v_post += w
apre += int(plastic)*Apre
w = clip( w + int(plastic)*apost, wmin, wmax )
'''
action_postspike_stdp = '''
apost += int(plastic)*Apost
w = clip( w + int(plastic)*apre, wmin, wmax )
'''
 
gamma = 0.2

model_mstdp = '''
plastic : boolean (shared)
w : volt
wmin : volt
wmax : volt
dPpre/dt = int(plastic)*-Ppre/taupre : volt (event-driven)
dPpost/dt = int(plastic)*-Ppost/taupost : volt (event-driven)
'''
action_prespike_mstdp = '''
v_post += w
Ppre += int(plastic)*Apre
w = clip( w + int(plastic) * gamma * r_post * Ppost, wmin, wmax)
'''
action_postspike_mstdp = '''
Ppost += int(plastic)*Apost
w = clip( w + int(plastic) * gamma * r_post * Ppre, wmin, wmax)
'''

tauz = 25*ms

model_mstdpet = '''
plastic : boolean (shared)
w : volt
wmin : volt
wmax : volt
dz/dt = int(plastic)*-z/tauz : volt (event-driven)
dPpre/dt = int(plastic)*-Ppre/taupre : volt (event-driven)
dPpost/dt = int(plastic)*-Ppost/taupost : volt (event-driven)
'''
action_prespike_mstdpet = '''
v_post += w
Ppre += int(plastic)*Apre
z += int(plastic)*Ppost
w = clip( w + int(plastic) * gamma * r_post * z, wmin, wmax)
'''
action_postspike_mstdpet = '''
Ppost += int(plastic)*Apost
z += int(plastic)*Ppre
w = clip( w + int(plastic) * gamma * r_post * z, wmin, wmax)
'''
