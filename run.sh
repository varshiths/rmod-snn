#!/usr/bin/env bash

rm savedmodels/sim3

# python2 main.py \
python main.py \
	--conf 0 \
	--test 0 \
	--model savedmodels/sim3 \
	--nepochs 200 \
	--dummy 0
	# --seed 10 \
	# --verbose True \
