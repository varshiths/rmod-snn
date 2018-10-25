#!/usr/bin/env bash

rm savedmodels/exp2_mstdp_sim1

# python2 main.py \
python main.py \
	--conf 2 \
	--rule mstdp \
	--test 0 \
	--model savedmodels/exp2_mstdp_sim1 \
	--nepochs 2000 \
	--seed 0 \
	--dummy 0
	# --verbose True \
