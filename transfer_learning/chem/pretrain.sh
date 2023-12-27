#!/bin/bash -ex
CUDA_VISIBLE_DEVICES=$1 python pretrain_html.py --aug1 random --aug2 random --iso_logic --a $2 --subiso_logic --b $3