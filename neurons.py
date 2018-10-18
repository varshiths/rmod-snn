
from brian2.units import *

tau = 20*ms
Vr = -70*mV
Vt = -54*mV

IF = '''
dv/dt  = (Vr-v)/tau : volt
'''
IF_out = '''
dv/dt  = (Vr-v)/tau : volt
r : 1
'''
