"""
Support Scripts for reading in data for variantNN
"""
import logging
import numpy as np
import intervaltree
import random

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_training_array(aln_tensor_fn, variant_set_fn, mask_bed_fn):
    base2num = dict(zip("ACGT",(0, 1, 2, 3)))
    # Load BED data into interval tree
    tree =  intervaltree.IntervalTree()
    with open(mask_bed_fn) as f:  # make list of positions of interest
        for row in f:
            row = row.strip().split()
            b = int(row[1]) # beginning coordinate
            e = int(row[2]) # end coordinate
            tree.addi(b, e, None)
    # Load variants (Labels)
    Y_intitial = {}
    with open(variant_set_fn) as f: # make dictionary of expected variants
        for row in f:
            row = row.strip().split()
            if row[3] == "0":  # Het base
                het = True
            else:
                het = False
            
            pos = int(row[0])
            if len(tree.search(pos)) == 0: # not included in BED file
                continue
            base_vec = [0,0,0,0,0,0,0,0]  #[A(0) C(1) G(2) T(3) het(4) hom(5) non-variant(6) non-SNP(7)]
            if het: # split coverage over 2 bases
                base_vec[base2num[row[1][0]]] = 0.5
                base_vec[base2num[row[2][0]]] = 0.5
                base_vec[4] = 1.
            else: # homo base
                base_vec[base2num[row[2][0]]] = 1
                base_vec[5] = 1.

            if len(row[1]) > 1 or len(row[2]) > 1 :  # not simple SNP case
                base_vec[7] = 1.
                base_vec[4] = 0.
                base_vec[5] = 0.
        
            Y_intitial[pos] = base_vec
            
    Y_pos = sorted(Y_intitial.keys())
    cpos = Y_pos[0]
    for pos in Y_pos[1:]: #consider variants w/in 12 bases as non-SNPs
        if abs(pos - cpos) < 12:
            Y_intitial[pos][7] = 1
            Y_intitial[cpos][7] = 1
            
            Y_intitial[pos][4] = 0
            Y_intitial[cpos][4] = 0
            Y_intitial[pos][5] = 0
            Y_intitial[cpos][5] = 0
        cpos = pos

    X_intitial = {}  
    # Load alignment data (Features)
    with open(aln_tensor_fn ) as f:  # parse dataset and include only reads at locations of interest
        for row in f:
            row = row.strip().split()
            pos = int(row[0]) #coordinate
            if len(tree.search(pos)) == 0: # check if included in BED
                continue
            ref_seq = row[1] # reference sequence
            if ref_seq[7] not in ["A","C","G","T"]: # non-standard center base
                continue
            vec = np.reshape(np.array([float(x) for x in row[2:]]), (15,3,4))

            vec = np.transpose(vec, axes=(0,2,1))
            if sum(vec[7,:,0]) < 5:
                continue
            
            vec[:,:,1] -= vec[:,:,0] # subtract matrix 0 from 1 and 2
            vec[:,:,2] -= vec[:,:,0]

            
            X_intitial[pos] = vec
            
            if pos not in Y_intitial: # add corrdinate to Y_initial
                base_vec = [0,0,0,0,0,0,0,0]
                base_vec[base2num[ref_seq[7]]] = 1 # take center base from read
                base_vec[6] = 1. # set as non variant
                Y_intitial[pos] = base_vec
                
    all_pos = sorted(X_intitial.keys())
    random.shuffle(all_pos)

    Xarray = [] # Save array in random order (reads shuffled)
    Yarray = []
    pos_array = []
    for pos in all_pos:
        Xarray.append(X_intitial[pos])  # pair up reads and expected variants by position
        Yarray.append(Y_intitial[pos])
        pos_array.append(pos)
    Xarray = np.array(Xarray)
    Yarray = np.array(Yarray)
    Xarray = np.float32(Xarray)
    Yarray = np.float32(Yarray)

    return Xarray, Yarray, pos_array

def get_aln_array( aln_tensor_fn ):

    X_intitial = {}  

    with open( aln_tensor_fn ) as f:  #parse all reads (no target list)
        for row in f:
            row = row.strip().split()
            pos = int(row[0])  # coordinate
            ref_seq = row[1] # reference sequence
            ref_seq = ref_seq.upper()

            if ref_seq[7] not in ["A","C","G","T"]:  #non-standard base at center
                continue

            vec = np.reshape(np.array([float(x) for x in row[2:]]), (15,3,4))

            vec = np.transpose(vec, axes=(0,2,1))
            if sum(vec[7,:,0]) < 5:
                continue
            
            vec[:,:,1] -= vec[:,:,0]
            vec[:,:,2] -= vec[:,:,0]
            
            X_intitial[pos] = vec
                
    all_pos = sorted(X_intitial.keys())

    Xarray = []
    pos_array = []
    for pos in all_pos:
        Xarray.append(X_intitial[pos])
        pos_array.append(pos)
    Xarray = np.array(Xarray)
    Xarray = np.float32(Xarray)

    return Xarray, pos_array

def save_flat_file(Xarray, Yarray, Xfilename, Yfilename):
    with open(Xfilename, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(Xarray.shape))
        for dataslice in Xarray:
            np.savetxt(outfile, dataslice.reshape(15,12), fmt='%-7.2f')
            outfile.write('# New slice\n')
    with open(Yfilename, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(Yarray.shape))
        for dataslice in Yarray:
            np.savetxt(outfile, dataslice, fmt='%-7.2f')
            outfile.write('# New slice\n')

def read_flat_file(Xfilename, Yfilename):
    Xdata = np.loadtxt(Xfilename)
    Xdata = Xdata.reshape((int(Xdata.shape[0]/15), 15, 4, 3))
    Ydata = np.loadtxt(Yfilename)
    Ydata = Ydata.reshape((int(Ydata.shape[0]/8), 8))
    Xdata = np.float32(Xdata)
    Ydata = np.float32(Ydata)
    assert Ydata.shape[0]==Xdata.shape[0]
    return Xdata, Ydata