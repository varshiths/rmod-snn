#!/usr/bin/env bash

# rm savedmodels/exp1_sim1

# python2 main.py \
python main.py \
	--conf 0 \
	--test 1 \
	--model savedmodels/exp1_sim1 \
	--nepochs 200 \
	--verbose True \
	--dummy 0
	# --seed 100 \
