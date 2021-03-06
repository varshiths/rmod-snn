
# from exp10 import Experiment10
from exp0 import Experiment0
from exp1 import Experiment1
from exp2 import Experiment2
from exp3 import Experiment3
from exp4 import Experiment4

import os
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None,
                    help='Seed for simulation')
parser.add_argument('--conf', type=int, required=True,
                    help='Configuration Identifier')
parser.add_argument('--test', type=int, required=True,
                    help='Train vs Test')
parser.add_argument('--model', type=str, required=True,
                    help='Model Parameters File')
parser.add_argument('--rule', type=str, default="mstdp",
                    help='Configuration Identifier')
parser.add_argument('--nepochs', type=int, default=20,
                    help='Number epochs to train for')
parser.add_argument('--nepochs_per_save', type=int, default=20,
                    help='Number of epochs after which model is saved')
parser.add_argument('--verbose', type=bool, default=False,
                    help='If state is to be monitored')
parser.add_argument('--dummy', type=int, required=True,
                    help='Dummy')

def main(args):

	# create savedmodels
	if not os.path.isdir(os.path.dirname(args.model)):
		os.mkdir(os.path.dirname(args.model))

	assert args.rule in ["mstdp", "mstdpet"]

	# if args.conf == 10:
		# Experiment10(args)
	# elif args.conf == 0:
	if args.conf == 0:
		Experiment0(args)
	elif args.conf == 1:
		Experiment1(args)
	elif args.conf == 2:
		Experiment2(args)
	elif args.conf == 3:
		Experiment3(args)
	elif args.conf == 4:
		Experiment4(args)
	else:
		print("Configuration not available: {}".format(args.conf))
		sys.exit(1)

if __name__ == '__main__':
    main(parser.parse_args())
