#!/bin/bash
#SBATCH --job-name=ubi_opt
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1

echo "========= Job started  at `date` =========="
echo This job is running on the following node\(s\):
echo $SLURM_NODELIST

cd $SLURM_SUBMIT_DIR
source /comm/specialstacks/gromacs-volta/bin/modules.sh
module load gromacs-tmpi-gcc-8.2.0-openmpi-4.0.3-cuda-10.1/2020.4 

PDB=ubi_1xqq_60.pdb
PROT=ubi

# -------------------- #
# 0. PREPARE FILES     #
# -------------------- #

# Create directories
mkdir 1-input 2-emin 3-nvt 4-npt 5-md

cp -r moddihed_ffAMBER.py amber99sb-star-ildn.ff ${PDB} 1-input
cd 1-input

# -------------------- #
# 1. SETUP SYSTEM      #
# -------------------- #

# Generate topology: choose option 1: AMBER99SB*-ILDN and 1: TIP4P/2005
gmx pdb2gmx -f ${PDB} -ignh <<-EOF
1
7
EOF

# Apply methyl fix (Hoffmann et al 2018, PCCP)
mv topol.top topol_orig.top
python moddihed_ffAMBER.py topol_orig.top > topol.top

# Build box
gmx editconf -f conf.gro -o ${PROT}_box.gro -c -d 1.2 -bt dodecahedron

# Solvate the box
gmx solvate -cp ${PROT}_box.gro -cs amber99sb-star-ildn.ff/tip4p2005.gro -p topol.top -o ${PROT}_solvate.gro

# Add ions, removing only SOL molecules
gmx grompp -f ../../mdp_files/minim.mdp -c ${PROT}_solvate.gro -p topol.top -o ${PROT}_grompp.tpr -maxwarn 2
echo SOL | gmx genion -s ${PROT}_grompp.tpr -p topol.top -o ${PROT}_neutral.gro -pname NA -nname CL -neutral

# ---------------------- #
# 2. ENERGY MINIMIZATION #
# ---------------------- #

cp -r ${PROT}_neutral.gro posre.itp amber99sb-star-ildn.ff topol.top ../2-emin
cd ../2-emin

# Prepare the executable and run energy minimization
gmx grompp -f ../../mdp_files/minim.mdp -c ${PROT}_neutral.gro -p topol.top -o emin.tpr -maxwarn 2
gmx mdrun -deffnm emin -nb gpu -pme auto -dlb no -npme 0

# Sanity check
gmx energy -f emin.edr -o energy.xvg -xvg none <<-EOF
Potential

EOF

# ------------------gg #
# 3. NVT EQUILIBRATION #
# -------------------- #

cp -r amber99sb-star-ildn.ff emin.gro topol.top *.itp ../3-nvt
cd ../3-nvt

# Prepare the executable and run the NVT equilibration
gmx grompp -f  ../../mdp_files/nvt.mdp -c emin.gro -o nvt.tpr -p topol.top -r emin.gro
gmx mdrun -deffnm nvt -nb gpu -pme auto -dlb no -npme 0

# Sanity checks
gmx energy -f nvt.edr -o temperature.xvg -xvg none <<-EOF
Temperature

EOF

# -------------------  #
# 4. NPT EQUILIBRATION #
# -------------------- #

cp -r amber99sb-star-ildn.ff nvt.gro topol.top *.itp nvt.cpt ../4-npt
cd ../4-npt/

# Prepare the executable and run the NPT equilibration
gmx grompp -f ../../mdp_files/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -deffnm npt -nb gpu -pme auto -dlb no -npme 0 -maxh 1.9 

# Sanity checks
gmx energy -f npt.edr -o temp_press_dens.xvg -xvg none <<-EOF
Temperature
Pressure
Density

EOF

echo "========= Job finished at `date` =========="

