
from experiments import *
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--conf', type=int, required=True,
                    help='Configuration Identifier')
parser.add_argument('--test', type=int, required=True,
                    help='Train vs Test')
parser.add_argument('--model', type=str, required=True,
                    help='Model Parameters File')
parser.add_argument('--nepochs', type=int, default=20,
                    help='Number epochs to train for')
parser.add_argument('--nepochs_per_save', type=int, default=5,
                    help='Number of epochs after which model is saved')
parser.add_argument('--dummy', type=int, required=True,
                    help='Dummy')

def main(args):

	# create savedmodels
	if not os.path.isdir(os.path.dirname(args.model)):
		os.mkdir(os.path.dirname(args.model))

	if args.conf == 0:
		Experiment0(args)
	else:
		print("Configuration not available: {}".format(args.conf))
		sys.exit(1)

if __name__ == '__main__':
    main(parser.parse_args())
