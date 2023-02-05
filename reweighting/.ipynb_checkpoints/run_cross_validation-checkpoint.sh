#!/bin/bash
REX='../data/exp/ubiquitin/nmr_r1r2r3r4r5_rates_liao.npy'
EEX='../data/exp/ubiquitin/nmr_r1r2r3r4r5_errors_liao.npy'
RMD='../data/calc/ubiquitin/deltak_CgCorr/rmd.pkl'

OUT='../data/calc/ubiquitin/deltak_CgCorr/crossval/'
mkdir ${OUT}
mkdir ${OUT}/training
mkdir ${OUT}/test

for SEED in {1..10}; do
        nohup python -u absurder_cross_validate.py --rex ${REX} --eex ${EEX} --rmd ${RMD} --out ${OUT} --seed ${SEED} --test_size 0.2 > ${OUT}/crossval_${SEED}.out &
        done