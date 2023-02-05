#!/bin/bash
#SBATCH --job-name=t4l_cghg
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --array=3

echo "========= Job started  at `date` =========="

source /comm/specialstacks/gromacs-volta/bin/modules.sh
module load gromacs-gcc-8.2.0-openmpi-4.0.1-cuda-10.1/2020 

#mkdir 5-md/sim${SLURM_ARRAY_TASK_ID}

# ------------- #
# 5. PRODUCTION #
# ------------- #

# Set up directories
#cd $SLURM_SUBMIT_DIR/4-npt/
#cp -r amber99sb-star-ildn.ff *.itp npt.gro topol.top npt.cpt ../5-md/sim${SLURM_ARRAY_TASK_ID}
cd $SLURM_SUBMIT_DIR/5-md/sim${SLURM_ARRAY_TASK_ID}

#gmx_mpi grompp -f /home/kummerer/OPTIM_FF/T4L/mdp_files/md_npt.mdp -c npt.gro -r npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 2
gmx_mpi mdrun -deffnm md -nb gpu -pme auto -dlb no -npme 0 -cpi -maxh 23.9 -ntomp 18

# ------------- #
# 6. PROCESSING #
# ------------- #

# Remove PBC and center
gmx_mpi trjconv -f md.xtc -s md.tpr -o md${SLURM_ARRAY_TASK_ID}_prot_nopbc.xtc -pbc mol -ur compact -center <<-EOF
Protein
Protein
EOF

# Remove overall tumbling
gmx_mpi trjconv -f md${SLURM_ARRAY_TASK_ID}_prot_nopbc.xtc -s md.tpr -o md${SLURM_ARRAY_TASK_ID}_rot_trans.xtc -fit rot+trans <<-EOF
Backbone
Protein
EOF

# Check RMSD
gmx_mpi rms -f md${SLURM_ARRAY_TASK_ID}_prot_nopbc.xtc -s md.tpr -o rmsd.xvg -xvg none <<-EOF
Protein
Protein
EOF

echo "========= Job finished at `date` =========="

