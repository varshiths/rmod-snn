#!/usr/bin/env bash

rm savedmodels/exp0_mstdpet_sim1

# python2 main.py \
python main.py \
	--conf 0 \
	--rule mstdpet \
	--test 0 \
	--model savedmodels/exp0_mstdpet_sim1 \
	--nepochs 200 \
	--seed 0 \
	--dummy 0
	# --verbose True \
