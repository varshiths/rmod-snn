#!/usr/bin/env bash

# rm savedmodels/exp1_sim1

# python2 main.py \
python main.py \
	--conf 0 \
	--rule mstdp \
	--test 0 \
	--model savedmodels/exp0_mstdp_sim1 \
	--nepochs 200 \
	--verbose True \
	--seed 10 \
	--dummy 0
