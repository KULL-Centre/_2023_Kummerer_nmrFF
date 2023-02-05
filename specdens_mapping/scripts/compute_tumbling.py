import subprocess
import glob
import os
import mdtraj as md
import numpy as np
import sys
import pickle as pkl
import _pickle as cPickle
import bz2
import re
import argparse
from math import log
from lmfit import minimize, Parameters
from scipy import optimize


########## Functions ##########

def mkdir( directory ):
    if not os.path.isdir(directory):
        os.mkdir( directory )

def load( inp ):
    pin = open( inp, "rb" )
    return pkl.load( pin )

def load_bz2_pkl(inp):
    data = bz2.BZ2File(inp, 'rb')
    data = cPickle.load(data)
    return data

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
def LS3(params,x, data):
   a = params['a'].value
   b = params['b'].value
   c = params['c'].value
   model = np.exp(-x/a)*(b+(1-b)*np.exp(-x/c))
   return model - data

def save( outfile, results ):
    with open( outfile + ".pkl", "wb" ) as fp:
        pkl.dump( results, fp )

def load_and_concat( infile ):
    
    def load( inp ):
        pin = open( inp, "rb" )
        return pkl.load( pin )
    
    arr = []
    for f in glob.glob( infile ):
        arr.append( load(f) )
    arr = np.concatenate( arr, axis = -1 )
    
    return arr

# For sorting:
def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s
        
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
        l.sort(key = alphanum_key)
        return l
    
def parse():
    
    parser = argparse.ArgumentParser( description = '' )

    # Mandatory Inputs
    parser.add_argument( '--quadric', type = str, required = True,
                        help = 'Path to the QUADRIC Diffusion executable.' )
    parser.add_argument( '--pdbinertia', type = str, required = True,
                        help = 'Path to the PDBInertia executable.' )
    parser.add_argument( '--in_dir', type = str, required = True,
                        help = 'Path to input directory.' )
    parser.add_argument( '--b', type = int, required = False, default = 0,
                        help = 'Initial time (in ps) to be used in the trajectory (default: 0 ps).' )
    parser.add_argument( '--e', type = int, required = False, default = 1000000,
                        help = 'Final time (in ps) to be used in the trajectory (default: 1000000 ps, corresponding to 1us at 1ps stride). In case the trajectories have different lengths, this has to be the maximum length of the shortest one.' )
    parser.add_argument( '--lblocks_bb', type = int, required = True,
                        help = 'Length of the blocks (in ps) employed to estimate the backbone tumbling time. Rule of thumb: ~50x the expected tumbling time (if you ran compute_tcfs.py use the same value you specified there).' )
    parser.add_argument( '--stau', type = int, required = True,
                        help = 'Initial guess for the backbone tumbling time (in ps).' )
    parser.add_argument( '--diff_model', type = str, required = True, choices = ['iso', 'axial', 'anis'], help = 'Diffusion model. Choose between "iso" (for isotropic diffusion model), "axial" (for axially symmetric) and "anis" (for fully anisotropic).' )
    parser.add_argument( '--trajname', type = str, required = True,
                        help = 'Common prefix of the trajectories (Ex. sim for sim1.xtc, sim2.xtc etc.).' )
    parser.add_argument('--gen_mlist', action = 'store_true', required = False, default = False, help = 'Get the ordered list of methyls. The rest of the code will not be executed.')

    args = parser.parse_args()
    return args.quadric, args.pdbinertia, args.in_dir, args.b, args.e, args.lblocks_bb, args.stau, args.diff_model, args.trajname, args.gen_mlist    


##########################
########## MAIN ##########
##########################

quadric, pdbinertia, in_dir, b_frame, e_frame, l_blocks_bb, start_taum, diff_model, trajname, gen_mlist = parse()

print( f"# The following arguments were provided:\n --in {in_dir} --quadric {quadric} --pdbinertia {pdbinertia} --b {b_frame} --e {e_frame} --lblocks_bb {l_blocks_bb} --stau {start_taum} --diff_model {diff_model} --trajname {trajname}")


if gen_mlist:
    ''' Get list with methyl names '''
    
    print("# Generating methyl list.")
    mkdir( f'{in_dir}/tau' )
    methyls_carbons = { 'ALA': ['CB'], 'VAL': ['CG1', 'CG2'], 'THR': ['CG2'], 'ILE': ['CG2', 'CD1'], 'LEU': ['CD1', 'CD2'], 'MET': ['CE'] }
    
    struct          = md.load(f'{in_dir}/initial.pdb')
    topology        = struct.topology
    table, bonds    = topology.to_dataframe()
    
    with open( f'{in_dir}/tau/tauR_methyl_specific', 'w' ) as f: # Creates a file listing the methyl names
        for res in methyls_carbons.keys():
            mtable = table[table['resName'] == res]

            for mc in methyls_carbons[res]:
                tmp = list( mtable.loc[mtable['name'] == mc, 'serial'] )

                for c in tmp:

                    a1 = list(mtable.loc[mtable['serial'] == c, 'resName'])
                    a2 = list(mtable.loc[mtable['serial'] == c, 'resSeq'])
                    a3 = list(mtable.loc[mtable['serial'] == c, 'name'])
                    a4 = list(mtable.loc[mtable['serial'] == c+1, 'name'])
                    f.write( f'{a1[0]}{a2[0]}-{a3[0]}{a4[0][:-1]}\n' ) # Ex. format: ALA41-CBHB
    print("# DONE!")
    sys.exit()

len_traj        = e_frame - b_frame
n_blocks_bb     = int( len_traj / l_blocks_bb )
nh_res          = load(f'{in_dir}/nh_residues.pkl')
nh_count        = len( nh_res )
tmp             = glob.glob(f'{in_dir}/tcf_bb/*.pbz2')
ntraj           = int( len( tmp ) )

''' Get list with methyl names '''
struct          = md.load(f'{in_dir}/initial.pdb')
topology        = struct.topology
table, bonds    = topology.to_dataframe()
methyls_carbons = { 'ALA': ['CB'], 'VAL': ['CG1', 'CG2'], 'THR': ['CG2'], 'ILE': ['CG2', 'CD1'], 'LEU': ['CD1', 'CD2'], 'MET': ['CE'] }

methyl_names   = [] # !!! NOT IN THE SAME ORDER AS THE FINAL OUTPUT !!!

for res in methyls_carbons.keys():
    mtable     = table[table['resName'] == res]

    for mc in methyls_carbons[res]:
        tmp    = list( mtable.loc[mtable['name'] == mc, 'serial'] )

        for c in tmp:
            a1 = list(mtable.loc[mtable['serial'] == c, 'resName'])
            a2 = list(mtable.loc[mtable['serial'] == c, 'resSeq'])
            a3 = list(mtable.loc[mtable['serial'] == c, 'name'])
            a4 = list(mtable.loc[mtable['serial'] == c+1, 'name'])
            methyl_names.append(f'{a1[0]}{a2[0]}-{a3[0]}{a4[0][:-1]}')
            #print(f'{a1[0]}{a2[0]}-{a3[0]}{a4[0][:-1]}')

''' Parse bb NH TCFs into np.array '''

bb_pkls = sort_nicely( glob.glob(f'{in_dir}/tcf_bb/{trajname}*_prot_nopbc.pbz2') )
for trj in range( ntraj ): 
    tcfs_bb = []
    
    print(f"# Analysing trajectory {trj+1}/{ntraj}")
    print( "   # Load backbone NH time correlation functions")
    tcfs_bb = load_bz2_pkl( bb_pkls[trj] )        

    ''' Fit bb NH TCFS to LS models '''

    # Starting parameters
    simlength  = l_blocks_bb
    fit_length = int(l_blocks_bb / 2)
    accuracy   = 100

    delta_taum = start_taum
    min_taum   = start_taum - delta_taum
    max_taum   = start_taum + delta_taum

    start_S    = 0.8 
    min_S      = 0
    max_S      = 1

    start_taue = 50
    min_taue   = 5
    max_taue   = 100

    start_Sf   = 1.0
    min_Sf     = 0
    max_Sf     = 1

    t          = np.arange(0, fit_length+1, 1)
    maxlogtime = int( accuracy * np.round( np.log( simlength ) ) )
    tmp        = [ np.exp( i / accuracy ) for i in np.linspace( 1, maxlogtime, maxlogtime ) ] # exp(maxlogtime/accuracy) ~ simlength; exponentially distributed time points
    exp_t      = np.unique( [ int (np.round( tmp[j]) ) for j in range( len(tmp) ) if [ i < fit_length for i in tmp ][j] ] ) # exponentially distributed time points < fit_length

    print("   # Fit backbone time correlation function to LS")
    tau_Ms = []
    for bl in range( n_blocks_bb ):

        tau_tmp = []
        for tcf in range(1, nh_count+1 ):

            params = Parameters()
            params.add( 'a', value = start_taum, min = min_taum, max = max_taum )
            params.add( 'b', value = start_S,    min = min_S,    max = max_S )
            params.add( 'c', value = start_taue, min = min_taue, max = max_taue )

            ct = [ tcfs_bb[bl][ i,tcf ] for i in exp_t ]
            try:
                result = minimize( LS3, params, args=(exp_t, ct) )
                popt=np.zeros(3)
                popt[0]=result.params['a'].value
                popt[1]=result.params['b'].value
                popt[2]=result.params['c'].value
            except:
                popt = [0,0,0]
            
            tau_tmp.append( popt[0] )
        tau_Ms.append( tau_tmp )

    mkdir( f'{in_dir}/tau' )
    with open( f'{in_dir}/tau/{trajname}{trj}_tauM.pkl', 'wb' ) as fp:
                pkl.dump( tau_Ms, fp )

''' Load tauMs and residue numbers & save tm.avg.input '''

print("# Averaging tauMs over trajectories and blocks")
tauMs = []
for f in glob.glob( f'{in_dir}/tau/*_tauM.pkl' ):
    pin      = open( f, "rb" )
    tauMs.append( pkl.load( pin ) )

tauMs         = np.concatenate( tauMs, axis = 0 )
tauM_av       = np.average( tauMs, axis = 0 )
tauM_std      = np.std( tauMs, axis = 0 )

tm_input      = np.empty( (len(nh_res), 3) )
tm_input[:,0] = nh_res
tm_input[:,1] = tauM_av/1000
tm_input[:,2] = tauM_std/1000

np.savetxt(f'{in_dir}/tau/tm.avg.input', tm_input, fmt = '%g')

''' Prepare pdb with initial coordinates '''

print("# Running PDBInertia")
process = f'{pdbinertia} -r {in_dir}/initial.pdb {in_dir}/tau/initial.inertia.pdb > {in_dir}/tau/initial.inertia.output'
p       = subprocess.Popen( process, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True )
p.communicate()

''' Write QUADRIC input file and execute QUADRIC '''

print("# Running QUADRIC Diffusion")
with open( f'{in_dir}/tau/quadric.in', 'w' ) as f:
    f.write( f"# sample control file\n0.8 1.2 10\n1 'N'\n{in_dir}/tau/tm.avg.input\n{in_dir}/tau/initial.inertia.pdb\navg.axial.pdb\navg.anis.pdb\n" )

process = f'{quadric} {in_dir}/tau/quadric.in > {in_dir}/tau/quadric_log.out'

p       = subprocess.Popen( process, shell = True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, universal_newlines = True )
p.communicate()
os.system(f'mv avg.axial.pdb avg.anis.pdb {in_dir}/tau/') # Quadric has problems saving the output in the correct directory, thus they are moved afterwards

''' Calculate distances between C-CH atoms '''

print("# Calculating distances between C-CH atoms")
struct          = md.load(f'{in_dir}/initial.pdb')
topology        = struct.topology
table, bonds    = topology.to_dataframe()

methyls_carbons = { 'ALA': ['CB'], 'VAL': ['CG1', 'CG2'], 'THR': ['CG2'], 'ILE': ['CG2', 'CD1'], 'LEU': ['CD1', 'CD2'], 'MET': ['CE'] }
carbons         = { 'ALA': ['CA'], 'VAL': ['CB', 'CB'],   'THR': ['CB'],  'ILE': ['CB', 'CG1'],  'LEU': ['CG', 'CG'], 'MET': ['SD'] }

if diff_model=='axial' or diff_model=='anis':
    pdb_qu          = md.load(f'{in_dir}/tau/avg.{diff_model}.pdb')
    xyz             = pdb_qu.xyz[0]
    distances_mod   = []
    distances_z     = []
    distances_x     = []
    
    for res in carbons.keys():
        mtable = table[table['resName'] == res]
    
        for x,c in enumerate( carbons[res] ):
            mc   = methyls_carbons[res][x]
            tmp  = list( mtable.loc[mtable['name'] == c, 'serial'] ) # preceeding carbon of methyl carbon
            tmp2 = list( mtable.loc[mtable['name'] == mc, 'serial'] ) # methyl carbons
    
            carbon_indices = []
            for n,i in enumerate(tmp):
                r = - xyz[i-1] + xyz[tmp2[n]-1]
                distances_z.append( r[2] ) # save z component of C-C distance vector
                distances_x.append( r[0] ) # save x component of C-C distance vector
                carbon_indices.append( [i-1, tmp2[n]-1] )
    
            carbon_indices = np.array(carbon_indices)
            distances_mod.append( md.compute_distances( pdb_qu, carbon_indices, periodic = False, opt = False )[0] ) # Computes modules of the distances
    
    distances_mod = np.concatenate( distances_mod )
    
''' Parsing Diso and Dpar from quadric output '''

print(f"# Computing methyl specific tauR. Diffusion model: {diff_model}")
with open(f'{in_dir}/tau/quadric_log.out', 'r') as f:
    lines = f.readlines()

if diff_model == 'iso':
    # Parses the QUADRIC output file to extract Diso and Dpar/Dper:
    for c,l in enumerate(lines):
        if 'Isotropic' in l:
            axc = c
            c  += 1
            break

    for l in lines[axc+1:]:
        if 'Actual' in l:
            axj = c
            break
        c += 1

    Diso     = float( lines[axj+1].split()[2] )
    # Calculate methyl specific tumbling time tauR according to an isotropic tumbling model:
    tauRs = [ (1/(6*Diso))*100 for i in range( len(methyl_names) )]
    
    
elif diff_model == 'axial':
    # Parses the QUADRIC output file to extract Diso and Dpar/Dper:
    for c,l in enumerate(lines):
        if 'Axial' in l:
            axc = c
            c  += 1
            break

    for l in lines[axc+1:]:
        if 'Jacknife' in l:
            axj = c
            break
        c += 1

    Diso     = float( lines[axj+1].split()[2] )
    DparDper = float( lines[axj+2].split()[1] )

    Dzz = 3*Diso/(1+2/DparDper)
    Dyy = 3*Diso/(2+DparDper)
    # Calculate methyl specific tumbling time tauR according to an axially symmetric tumbling model:
    tauRs     = []
    for i in range( len(methyl_names) ):
        ratio = distances_z[i] / distances_mod[i]
        Y20   = ( 3*ratio**2 - 1 )/2
        Di    = Diso - Y20 * ( Dzz-Dyy )/3
        tauRs.append( ( 1 / ( 6*Di ) )*100 )

        
elif diff_model == 'anis':
    # Parses the QUADRIC output file to extract Diso and Dpar/Dper:
    for c,l in enumerate(lines):
        if 'Anisotropic' in l:
            axc = c
            c  += 1
            break

    for l in lines[axc+1:]:
        if 'Actual' in l:
            axj = c
            break
        c += 1

    Diso       = float( lines[axj+1].split()[2] )
    Dzz2DxxDyy = float( lines[axj+2].split()[1] )
    DxxDyy     = float( lines[axj+3].split()[1] )

    # Solve 3 equations, 3 unknowens to get Dzz, Dxx, and Dyy:
    a = np.array([[Dzz2DxxDyy, Dzz2DxxDyy, -2], [1, -DxxDyy ,0], [1, 1, 1]])
    b = np.array([0, 0, 3*Diso])
    Dxxyyzz = np.linalg.solve(a, b)
    Dxx = Dxxyyzz[0]
    Dyy = Dxxyyzz[1]
    Dzz = Dxxyyzz[2]
    # Calculate methyl specific tumbling time tauR according to an anisotropic tumbling model:
    tauRs     = []
    for i in range( len(methyl_names) ):

        cos_theta = distances_z[i]/distances_mod[i] # same as ratio
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta)
        cos_phi = distances_x[i] / distances_mod[i]
        phi = np.arccos(cos_phi)
        Y22p = ((((3/8)**0.5) * sin_theta** 2) * np.exp(+2j * phi)) # Note: imaginary number j
        Y22n = ((((3/8)**0.5) * sin_theta** 2) * np.exp(-2j * phi)) # Note: imaginary number j
        Y20 = (3 * ((cos_theta)**2)-1)/2

        Di = Diso - (Dxx-Dyy) * (Y22p + Y22n) / (24**0.5) - Y20 * (2 * Dzz - Dxx - Dyy)/6
        tauRs.append( float(( 1 / ( 6*Di ) )*100) ) # use float(tauR) to discard imaginary part

pkl_tauRs = []
# Write to human readible file:
with open( f'{in_dir}/tau/tauR_methyl_specific', 'w' ) as f:
    f.write( '#resID tauR[ns]\n' )
    for i,tauR in enumerate( tauRs ):
        pkl_tauRs.append( [methyl_names[i], tauR] )
        f.write( f'{methyl_names[i]} {tauR}\n' )
        
nmr_indices = []
for i in methyl_names:
    nmr_indices.append( methyl_names.index( i ) )

print("# DONE!")
