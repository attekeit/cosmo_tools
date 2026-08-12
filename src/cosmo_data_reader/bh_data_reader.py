import numpy as np
import os
import glob
import pickle
import h5py
import ketjugw
import re

__all__ = ['load_allbh_data', 
           'load_nonketju_merger_info', 
           'load_ketju_merger_info',
           'load_combined_merger_info',
           'load_ketju_bhs_of_mergers']

def load_allbh_data(allbhs_file = None, recalc: bool=False):
    """
    Read the BH data from allbhs.hdf5 file. This can take a while with files having 
    >1000 BHs. For faster reloading, the loaded data is saved into a pickle file.

    Args:

        allbhs_file (int): The hdf5 file. Note that if you are analysing a simulation
                           that is still running, it is better to copy the hdf5 file
                           somewhere else and use that file.

        recalc (bool): go through the hdf5 file again even if pkl file exists


    Returns:
        allbhs
    """

    pkl_fname = allbhs_file[:-5] + '.pkl'
    
    if not recalc:
        try:
            with open(pkl_fname, 'rb') as f:
                allbhs = pickle.load(f)
                return allbhs
        except:
            return load_allbh_data(allbhs_file, recalc=True)

    print('Reading BH data from', allbhs_file)
    allbhs = ketjugw.load_hdf5(allbhs_file, enforce_mass_limit=False, all_bhs_output=True)

    print('Saving data to', pkl_fname)
    with open(pkl_fname, 'wb') as f:
        pickle.dump(allbhs, f, protocol=pickle.HIGHEST_PROTOCOL)

    return allbhs

def load_nonketju_merger_info(directory: str, recalc: bool = False):
    """
    Read the BH merger data from blackhole_details files. This can take a while with files 
    having >1000 BHs. Thus, the loaded data is saved into a pickle file.
    TODO parallelized version?

    Args:
        directory (str): Path to the output directory where the folder blackhole_details is.

        recalc (bool): go through the hdf5 file again even if pkl file exists


    Returns:
        merger_data (dict)
    """

    pkl_fname = directory + '/nonketju_merger_info.pkl'
    if not recalc:
        try:
            with open(pkl_fname, 'rb') as f:
                merger_data = pickle.load(f)
                return merger_data
        except:
            return load_nonketju_merger_info(directory, recalc=True)

    

    # word matches 'swallows' as a whole word, case-insensitive
    pattern = re.compile(r'\bswallows\b', re.IGNORECASE)

    N_merge = 0

    directory = directory + '/blackhole_details/'

    print('Looping through blackhole_details files in ', directory)
    # Find files like blackhole_details_*.txt
    file_pattern = os.path.join(directory, "blackhole_details_*.txt")
    z_merge = np.zeros(0)
    merger_id_pairs = np.empty((0,2))

    #example of a line which states that a merger has occurred
    #ThisTask=746, time=0.134105: id=8159924 swallows 7729381 (0.000105827 0.000878516)

    #TODO also save the masses (last two numbers on the "swallows" lines),
    #note that these seem to be a bit buggy :/
    lines_of_mergers = []

    files_checked = 0

    for filepath in glob.glob(file_pattern):
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if pattern.search(line):
                        words = line.split()
                        id1 = words[2][3:]
                        id2 = words[4]

                        #id check: if a run has crashed at some point, mergers
                        #have possibly ween written multiple times
                        merger_copy = False

                        #if only one merger has been saved:
                        if N_merge == 1:
                            if id1 in merger_id_pairs and id2 in merger_id_pairs:
                                merger_copy = True
                                break
                        else:
                            for binary_ids in merger_id_pairs:
                                if id1 in binary_ids and id2 in binary_ids:
                                    merger_copy = True
                                    break

                        if not merger_copy:
                            N_merge += 1
                            a= float(words[1][5:-1])
                            
                            z = 1/a-1
                            z_merge = np.append(z_merge, z)
                            
                            id_pair = np.array([[int(id1),int(id2)]])
                            merger_id_pairs = np.append(merger_id_pairs, id_pair, axis=0)
                            lines_of_mergers.append(line)
                    
        except OSError as e:
            print(f"Skipping {filepath}: {e}")
            continue

        files_checked += 1
        if files_checked % 50 == 0:
            print(files_checked, ' files done')

    if files_checked == 0:
        print('We did not find any files!')
        exit()

    merger_data = dict(
        merger_ids = merger_id_pairs,
        z = z_merge,
        N_merged = int(N_merge),
        lines_of_mergers = lines_of_mergers
    )

    with open(pkl_fname, 'wb') as f:
        pickle.dump(merger_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return merger_data

def load_ketju_merger_info(ketju_file: str, h = 0.674):
    """
    Read the merger data from the ketju_bhs files.
    

    Args:

        ketju_file (str): The hdf5 file. Note that if you are analysing a simulation
                           that is still running, it is better to copy the hdf5 file
                           somewhere else and use that file.

    Returns:
        N_merge (int): Total count of mergers

        z_merge (np.ndarray): Numpy array of merger redshifts

        merger_id_pairs (np.ndarray): array of merger id pairs
        


    Returns:
        ketju_merger_data (dict)
    """

    if ketju_file is None:
        ketju_file = directory + '/ketju_bhs.hdf5'

    with h5py.File(ketju_file, 'r') as f:
        #we have a gadget 3 run, so lower case 
        mergers = f['mergers'] 
        z_merge = mergers['merger_redshift']
        N_merge = len(z_merge)
        merger_id_pairs = np.zeros((N_merge,2), dtype=int)
        id1 = mergers["ID1"]
        id2 = mergers['ID2']
        id_remnant = mergers['ID_remnant']
        #in solar mass
        m1 = mergers['m1'] * 1e10/h
        m2 = mergers['m2'] * 1e10/h
        
        merger_id_pairs[:,0] = id1[:]
        merger_id_pairs[:,1] = id2[:]
            
    #If a simulation crashed for some reason, some mergers might be
    #written multiple times. Cut these from the data
    _, mask = np.unique(merger_id_pairs, axis=0, return_index=True)

    ketju_merger_data = dict(
        z = z_merge[mask],
        merger_ids = merger_id_pairs[mask],
        remnant_ids = id_remnant[mask],
        N_merged = int(len(z_merge[mask])),
        m1 = m1[mask],
        m2 = m2[mask]
    )
    return ketju_merger_data


def load_combined_merger_info(directory, recalc = False):
    print('Not implemented yet!')
    return -1

#Using find_binaries of ketju on the list of bhs that are a part of a 
#merger also finds binaries that do not merge. This functions only
#returns binaries that actually merge, same structure as find_binaries
#returns in ketjugw
def load_binaries_of_ketju_mergers(ketju_file: str, enforce_mass_limit=False):

    binaries = dict()

    ketju_merger_info = load_ketju_merger_info(ketju_file)
    ids = ketju_merger_info['merger_ids']
    

    merging_bhs = load_ketju_bhs_of_mergers(ketju_file, enforce_mass_limit)
    merged_binaries = dict()
    
    #TODO surely this calculation can be done better
    for idpair in ids:
        id1 = idpair[0]
        id2 = idpair[1]

        bh1_found = False
        bh2_found = False

        for bhid, bh in merging_bhs.items():
            if int(bhid) == int(id1):
                bh1 = bh
                bh1_found = True
            elif int(bhid) == int(id2):
                bh2 = bh
                bh2_found = True
            
            if bh1_found and bh2_found:
                break

        merger_pair = dict()
        merger_pair[id1] = bh1
        merger_pair[id2] = bh2
        #Should find exactly 1 binary
        merged_binary = ketjugw.find_binaries(merger_pair, remove_unbound_gaps=True, 
                            mass_limit = ketjugw.get_mass_limit(ketju_file))
        if len(merged_binary) != 1:
            print('load_binaries_of_ketju_mergers did not find one binary!')
            quit()

        #add the found binary to merged_binaries
        #this might work?
        merged_binaries[(str(id1), str(id2))] = merged_binary[(id1, id2)]
            

    return merged_binaries

#Get BHs that have been a part of a ketju integrated merger.
def load_ketju_bhs_of_mergers(ketju_file: str, enforce_mass_limit=False):

    ketju_merger_info = load_ketju_merger_info(ketju_file)
    ids = ketju_merger_info['merger_ids'].flatten()
    idlist = np.unique(ids)

    #could be made faster by only loading those BHs that belong to idlist
    ketju_bhs = ketjugw.load_hdf5(ketju_file, enforce_mass_limit=enforce_mass_limit)

    merging_bhs = dict()
    
    for bhid, bh in ketju_bhs.items():
        if int(bhid) in idlist:
            merging_bhs[bhid] = bh 
            
    #TODO maybe change this to return the actual binaries?

    return merging_bhs

