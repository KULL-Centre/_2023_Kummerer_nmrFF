import glob
import os
import numpy as np
import sys
import pickle as pkl
import _pickle as cPickle
import bz2
import re
import argparse


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

def load_and_concat( infile ):
    
    def load( inp ):
        pin = open( inp, "rb" )
        return pkl.load( pin )
    
    arr = []
    for f in sort_nicely( glob.glob( infile ) ):
        arr.append( load(f) )
    arr = np.concatenate( arr, axis = -1 )
    print(sort_nicely( glob.glob( infile ) ))
    
    return arr

def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

def sort_methyls( array, label_list ): # Sorts a list (methyl names) in natural order
    
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s
     
    def alphanum_key(s):
         return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    
    label_list_sort = label_list[:]
    sort_nicely( label_list_sort )
    array_dict = dict( zip(label_list, array) )
    temp_list  = []
    for l in label_list_sort:
        temp_list.append( array_dict[l] )
        
    return np.array(temp_list), label_list_sort

def sort_arr( array, label_list ): # Sorts an array based the sorting of another list
    arr_new = []
    for r in range( len( array ) ):
        arr_temp, all_sorted = sort_methyls( array[r], label_list )
        arr_new.append( arr_temp )
    
    return np.array( arr_new )

def save( outfile, results ):
    with open( outfile + ".pkl", "wb" ) as fp:
        pkl.dump( results, fp )

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
    
    parser.add_argument( '--in_dir', type = str, required = True,
                        help = 'Path to input directory.' )
    parser.add_argument('--tumbling', type = str, required = False, default = '', 
                       help = 'Path to the file with user defined backbone tumbling times. If specified, the calculation of the backbone tumbling times will be skipped. Use --gen_mlist to get the ordered list of methyls.')
    
    args = parser.parse_args()
    
    return args.in_dir, args.tumbling

''' Load final results, merge trajectories and order methyls in sequence order '''

print("# Merging final results")
in_dir, tumbling = parse()

# Merge the arrays (from different trajectories):
rates = load_and_concat( f'{in_dir}/results/*_rates.pkl' )
J     = load_and_concat( f'{in_dir}/results/*_J.pkl' )
JNMR  = load_and_concat( f'{in_dir}/results/*_JNMR.pkl' )

if tumbling == '':
    with open( f'{in_dir}/tau/tauR_methyl_specific', 'r' ) as f:
        lines = f.readlines()
else:
    with open( tumbling, 'r' ) as f:
        lines = f.readlines()
        
methyl_labels = []
for l in lines:
    l = l.split()
    if '#' in l[0]:
        continue
    else:
        methyl_labels.append( l[0] )
# Sort the arrays in methyl order:      
rates = sort_arr( rates, methyl_labels )
J     = sort_arr( J, methyl_labels )
JNMR  = sort_arr( JNMR, methyl_labels )

save( f'{in_dir}/results/rates', rates )
save( f'{in_dir}/results/J', J )
save( f'{in_dir}/results/JNMR', JNMR )
save( f'{in_dir}/results/methyls', sort_nicely( methyl_labels ) ) # Save sorted list of methyls

print("# DONE!")
