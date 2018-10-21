#!/usr/bin/env bash

rm savedmodels/exp0_mstdp_sim2

# python2 main.py \
python main.py \
	--conf 0 \
	--rule mstdp \
	--test 0 \
	--model savedmodels/exp0_mstdp_sim2 \
	--nepochs 400 \
	--seed 0 \
	--dummy 0
	# --verbose True \
