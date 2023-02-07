#!/bin/bash
# See the help entries for each script separately to learn about the flags.
# This is just an example to see which options were used. Need to modify according to your local paths etc.
# Here, the input to script is the name of the ff (e.g. 'CgCorr').

NTRAJS=5
SCRIPTS=scripts/
GMX=/path/to/gmx_mpi
OUT=/data/ci2/deltak_${1}/analysis/
B=0
E=1000000
LBLOCKS_BB=1000000
LBLOCKS_M=10000
QUADRIC=/path/to/quadric_diffusion
PDBINERTIA=/path/to/pdbinertia
IN_DIR=$OUT
STAU=4000 #Guess in ps
TRAJNAME=md
LOGS=${IN_DIR}/logs/
CT_LIM=2
WD=115.150 # !Change
TUMBLING=../data/exp/ci2/tauR_exp_isotropic #!Change

export OMP_NUM_THREADS=1
mkdir $OUT
mkdir $LOGS

for s in $(seq 1 ${NTRAJS}); do
	cd /path/to/md/deltak_${1}/5-md/sim${s}/ 
	if [ -f "md_2019.tpr" ]; then
    		echo "md_2019.tpr exists already."
	else 
		gmx_mpi grompp -f mdp_files/md_npt.mdp -c npt.gro -r npt.gro -p topol.top -o md_2019.tpr -maxwarn 2
	fi
done

# Compute the TCFs
for s in $(seq 1 ${NTRAJS}); do
        XTC=/path/to/md/deltak_${1}/5-md/sim${s}/
        TPR=/path/to/md/deltak_${1}/5-md/sim${s}/${TRAJNAME}_2019.tpr
        NOPBC=${XTC}/${TRAJNAME}${s}_prot_nopbc.xtc
        ROT_TRANS=${XTC}/${TRAJNAME}${s}_rot_trans.xtc
        sleep 8s
        nohup python ${SCRIPTS}/compute_tcfs.py --xtc $XTC --tpr $TPR --gmx $GMX --out $OUT --b $B --e $E --lblocks_bb $LBLOCKS_BB --lblocks_m $LBLOCKS_M --nopbc $NOPBC --rot_trans $ROT_TRANS > ${LOGS}/compute_tcfs_${s}.log &
done

wait 

# Compute the tumbling:
nohup python ${SCRIPTS}/compute_tumbling.py --quadric $QUADRIC --pdbinertia $PDBINERTIA --in_dir $IN_DIR --lblocks_bb $LBLOCKS_BB --e $E --b $B --stau $STAU --trajname $TRAJNAME --diff_model "iso" > ${LOGS}/compute_tumbling.log &

wait

# Do spectral density mapping:
for s in $(seq 1 ${NTRAJS}); do
        sleep 5s
        nohup python ${SCRIPTS}/specdens_mapping.py --in_dir $IN_DIR --trj_idx $s --b $B --e $E --lblocks_m $LBLOCKS_M --trajname $TRAJNAME --wD $WD --ct_lim $CT_LIM --tumbling $TUMBLING > ${LOGS}/specdens_mapping_${s}.log & #--tumbling $TUMBLING
done

wait

# Merge results:
nohup python ${SCRIPTS}/merge_results.py --in_dir $IN_DIR --tumbling $TUMBLING > ${LOGS}/merge_results.log & # --tumbling $TUMBLING
