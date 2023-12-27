#!/bin/bash -ex

for seed in 0 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=$1 python HTML.py --DS $2 --lr 0.01 --local --num-gc-layers 3 --aug $3 --seed $seed --iso_logic --a $4 --subiso_logic --b $5
done


