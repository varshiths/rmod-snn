#!/usr/bin/env bash

rm savedmodels/temp

# python2 main.py \
python main.py \
	--conf 10 \
	--rule mstdp \
	--test 0 \
	--model savedmodels/temp \
	--nepochs 200 \
	--verbose True \
	--seed 0 \
	--dummy 0
