#!/usr/bin/env bash

rm savedmodels/exp4_mstdp_sim1

# python2 main.py \
python main.py \
	--conf 4 \
	--rule mstdp \
	--test 0 \
	--model savedmodels/exp4_mstdp_sim1 \
	--nepochs 150 \
	--seed 0 \
	--dummy 0
	# --verbose True \

rm savedmodels/exp4_mstdpet_sim1

# python2 main.py \
python main.py \
	--conf 4 \
	--rule mstdpet \
	--test 0 \
	--model savedmodels/exp4_mstdpet_sim1 \
	--nepochs 150 \
	--seed 0 \
	--dummy 0
	# --verbose True \
