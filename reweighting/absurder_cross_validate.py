import sys
import ABSURDer as absurder
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import argparse
import os


#####################################
############## Parser ###############
def parse():

    parser = argparse.ArgumentParser( description = '' )

    # Mandatory Inputs
    parser.add_argument( '--rex', type=str, required=True, help='Path to the experimental values.')
    parser.add_argument( '--eex', type=str, required=True, help='Path to the experimental errors.')
    parser.add_argument( '--rmd', type=str, required=True, help='Path to the experimental errors.')
    parser.add_argument( '--out', type=str, required=True, help='Path to the output directory (will be created if not existant).')
    parser.add_argument( '--seed', type=int, required=True, help='Seed for splitting of the dataset.')
    parser.add_argument( '--test_size', type=float, required=False, default=0.2, help='Test set size. Value between 0 and 1 (default=0.2).')
    
    args = parser.parse_args()

    return args.rex, args.eex, args.rmd, args.out, args.seed, args.test_size


########################################
############## Functions ###############
def load( inp ):
    pin = open( inp, "rb" )
    return pkl.load( pin )

def save( outfile, results ):
    with open( outfile + ".pkl", "wb" ) as fp:
        pkl.dump( results, fp )

def mkdir( directory ):
    if not os.path.isdir(directory):
        os.mkdir( directory )
        
        
##################################
############## MAIN ##############
##################################

rexf, eexf, rmdf, outf, seed, test_size = parse()

print('\nInput options:')
print(f'--rex {rexf} --eex {eexf} --rmd {rmdf} --out {outf} --seed {seed} --test_size {test_size}')

thetas = np.geomspace(100,20000, 30)

############## NMR data ###############
# get only WT and get it in right shape:
rex = np.load(rexf) 
eex = np.load(eexf)
print(rex.shape, eex.shape)

############## MD data ###############
rmd = load(rmdf)
# Get a residue list:
residues = load(f'{"/".join(rexf.split("/")[:-1])}/nmr_methyls.pkl')

############## ABSURDer ###############
# Split data into training and test sets:
idx_all = np.arange(0,rmd.shape[1])
idx_train,idx_test, residues_train,residues_test = train_test_split(idx_all, residues, test_size=test_size, random_state=seed)

# Reweight training set:
idx_rm = []
rw_train = absurder.ABSURDer(rex[:,idx_train], rmd[:,idx_train,:], eex[:,idx_train], out=f'{outf}/training/results_{seed}', idx=idx_rm, thetas=thetas, verbose=False, ignore_last=False)
rw_train.reweight(-1)
rw_train.phix2r(-1)
chis_train = np.insert(rw_train.chi,0,rw_train.ix2[-1])
phis_train = np.insert(rw_train.phi,0,1)
thetas_ABSURDer = rw_train.ths

# Apply weights to test set:
idx_rm = []
rw_test = absurder.ABSURDer(rex[:,idx_test], rmd[:,idx_test,:], eex[:,idx_test], idx=idx_rm, thetas=thetas, verbose=False, ignore_last=False)
rw_test.load_results(f'{outf}/training/results_{seed}.pkl')
rw_test.phix2r(-1)
chis_test = np.insert(rw_test.chi,0,rw_test.ix2[-1])
phis_test = np.insert(rw_test.phi,0,1)

# Save stuff:
save( f'{outf}/training/chis_{seed}', np.array(chis_train) )
save( f'{outf}/training/phis_{seed}', np.array(phis_train) )
save( f'{outf}/training/thetas_{seed}', np.array(thetas_ABSURDer) )

save( f'{outf}/test/chis_{seed}', np.array(chis_test) )
save( f'{outf}/test/phis_{seed}', np.array(phis_test) )
