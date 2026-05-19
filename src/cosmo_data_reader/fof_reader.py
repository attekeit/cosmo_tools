import numpy as np
import os
import glob
import re
import pickle
import h5py
import pygad

__all__ = ['read_fof_files']

class Fof():

    def __init__(self, mdm, mstar, nbh, max_mbh, mlowres, x, y, z,
                bh_masses = None, bh_separations = None, bhids=None):
        self.mdm = mdm
        self.mstar=mstar
        self.nbh=nbh
        self.max_mbh = max_mbh
        self.x = x
        self.y = y
        self.z = z
        self.mlowres = mlowres

        if bh_masses is None:
            self.bh_masses = np.zeros(max(1,nbh))
        elif nbh > 0:
            self.bh_masses = np.array(bh_masses, dtype=float)
        else:
            self.bh_masses = None

        if bh_separations is None:
            self.bh_separations = np.zeros(max(1,nbh))
        elif nbh > 0:
            self.bh_separations = np.array(bh_separations, dtype=float)
        else:
            self.bh_separations = None

        if bhids is None:
            self.bhids = np.zeros(max(1,nbh))
        elif nbh > 0:
            self.bhids = np.array(bhids, dtype=int)
        else:
            self.bhids = None

class Subhalo_bh_info():
    '''
    Contains info about the IDs, masses and separation from centre for BH in subhalo.
    Also includes the subhalo dark matter (total and lowres) and stellar masses.
    
    '''
    def __init__(self, bhids, bh_m, bh_m_sph, bh_r, mstar, mdm, mlowres):
        self.mdm = mdm
        self.mstar=mstar
        self.mlowres = mlowres
        self.bhids = np.array(bhids, dtype=int)
        self.bh_r = np.array(bh_r, dtype=float)
        self.bh_m = np.array(bh_m, dtype=float)
        self.bh_m_sph = np.array(bh_m_sph, dtype=float)


def read_fof_files(folder_path: str, snap_i: int,
                        recalc: bool = False):
    
    """
    NOTE: This assumes that subfind has been compiled (at least) with the following flags:
          SUBFIND
          WRITE_SUB_IN_SNAP_FORMAT     # Save subfind results in snap format

    Read the FoF structure data from SUBFIND files. Saves the info to a pkl file
    which will be used if this is called again. 

    TODO known issue(s): Run and creates an empty pkl file even if files
                         are not found

    Args:
        folder_path (str): Path to the directory to scan (output directory, 
        not the /groups_iii folder in it).

        snap_i (int): Number of snapshot

        recalc (bool): loop through subfind files again even if pkl file exists

    Returns:
        structure_data (dict): Includes:
            - mdm 
            - mstar 
            - max_mbh 
            - Nbh 
            - mlowres 
            - pos_x (NOTE! In comoving units)
            - pos_y (NOTE! In comoving units)
            - pos_z (NOTE! In comoving units)
    """

    subfind_fname_start = folder_path + f"/groups_{snap_i:03d}/sub_{snap_i:03d}"
    
    pkl_fname = folder_path + f"/FoF_data_{snap_i:03d}.pkl"
    #check if data is already saved. If not, we will loop through subfind files.
    if not recalc:
        try:
            with open(pkl_fname, 'rb') as f:
                data_stucture = pickle.load(f)
        #This most likely is not best practise but whatever
        except FileNotFoundError:
            data_stucture = read_fof_files(folder_path, snap_i, recalc=True)
        return data_stucture


    print('Looping through subfind files in ', folder_path, 'for snapshot ', snap_i)

    a = None
    h = None
   
    # Find files like groups folder
    file_pattern = subfind_fname_start + '.*.hdf5'

    z_merge = np.zeros(0)
    merger_id_pairs = np.empty((0,2))

    files_checked = 0

    fof_mass = np.zeros(100000)
    mvir = np.zeros_like(fof_mass)
    rvir = np.zeros_like(fof_mass)
    mvir_mean = np.zeros_like(fof_mass)
    rvir_mean = np.zeros_like(fof_mass)
    pos_x = np.zeros_like(fof_mass)
    pos_y = np.zeros_like(fof_mass)
    pos_z = np.zeros_like(fof_mass)
    fof_indices = np.zeros(len(fof_mass), dtype=int)

    fof_total = 0

    len_start = len(subfind_fname_start) + 1
    for subfind_fname in sorted(glob.glob(file_pattern), key=lambda x: int(x[len_start:-5])):
        try:
            with h5py.File(subfind_fname, 'r') as f:
                header_info = f['Header'].attrs
                if a is None:
                    a = header_info['Time']
                if h is None:
                    h = header_info['HubbleParam']

                subfind_ids = f['IDs']
                groups = f['Group']
                n_halos = len(groups['GroupLen'])

                fof_pos = groups['GroupPos']
                
                fof_mass[fof_total:fof_total+n_halos] = groups['GroupMass']/h*1e10
                mvir[fof_total:fof_total+n_halos] = groups['Group_M_Crit200']/h*1e10
                rvir[fof_total:fof_total+n_halos] = groups['Group_M_Mean200']/h*1e10
                rvir[fof_total:fof_total+n_halos] = groups['Group_R_Crit200']/h*a
                rvir_mean[fof_total:fof_total+n_halos] = groups['Group_R_Mean200']/h*a
                pos_x[fof_total:fof_total+n_halos] = fof_pos[:,0]
                pos_y[fof_total:fof_total+n_halos] = fof_pos[:,1]
                pos_z[fof_total:fof_total+n_halos] = fof_pos[:,2]
                fof_indices[fof_total:fof_total+n_halos] = np.linspace(fof_total, fof_total+n_halos, n_halos, dtype=int)
                
                fof_total += n_halos
    
        except OSError as e:
            print(f"Error trying to read file {filepath}: {e}!")
            quit()

        files_checked += 1

    fof_mass = fof_mass[:fof_total]
    mvir = mvir[:fof_total]
    rvir = rvir[:fof_total]
    mvir_mean = mvir_mean[:fof_total]
    rvir_mean = rvir_mean[:fof_total]
    pos_x = pos_x[:fof_total]
    pos_y = pos_y[:fof_total]
    pos_z = pos_z[:fof_total]
    fof_indices = fof_indices[:fof_total]
        
    structure_data = dict(
        fof_mass = fof_mass,
        mvir = mvir,
        rvir = rvir,
        mvir_mean = mvir_mean,
        rvir_mean = rvir_mean,
        pos_x = pos_x,
        pos_y = pos_y,
        pos_z = pos_z,
        fof_indices = fof_indices
    )
    

    #save FoF data to pkl format for faster access
    print('Saving the read FoF data to ', pkl_fname)
    with open(pkl_fname, 'wb') as f:
        pickle.dump(structure_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return structure_data
