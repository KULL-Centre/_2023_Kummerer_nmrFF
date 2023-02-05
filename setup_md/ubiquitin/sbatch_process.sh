#!/bin/bash
#SBATCH --job-name=ubi_opt
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
##SBATCH --gres=gpu:v100:1
#SBATCH --array=1-4

echo "========= Job started  at `date` =========="
echo This job is running on the following node\(s\):
echo $SLURM_NODELIST

source /comm/specialstacks/gromacs-volta/bin/modules.sh
module load gromacs-tmpi-plumed2-2.7.3-gcc-8.2.0-openmpi-4.0.3-cuda-10.1/2021.4

cd $SLURM_SUBMIT_DIR/5-md/sim${SLURM_ARRAY_TASK_ID}

# ------------- #
# 6. PROCESSING #
# ------------- #

# Remove PBC and center
gmx trjconv -f md.xtc -s md.tpr -o md${SLURM_ARRAY_TASK_ID}_prot_nopbc.xtc -pbc mol -ur compact -center <<-EOF
Protein
Protein
EOF

# Remove overall tumbling
gmx trjconv -f md${SLURM_ARRAY_TASK_ID}_prot_nopbc.xtc -s md.tpr -o md${SLURM_ARRAY_TASK_ID}_rot_trans.xtc -fit rot+trans <<-EOF
Backbone
Protein
EOF

# Check RMSD
gmx rms -f md${SLURM_ARRAY_TASK_ID}_prot_nopbc.xtc -s md.tpr -o rmsd.xvg -xvg none <<-EOF
Protein
Protein
EOF

echo "========= Job finished at `date` =========="
