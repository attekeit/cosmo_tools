import numpy as np
import os
import glob
import re
import pickle
import h5py
import pygad

__all__ = ['read_subfind_files',
            'get_bh_ids_in_each_subsystem']

#TODO add info about FoF host group nuymber
class Subhalo():

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
    def __init__(self, bhids, bh_m, bh_m_sph, bh_r, mstar, mdm, mlowres, host_ind, sub_centre, rvir=None):
        self.mdm = mdm
        self.mstar=mstar
        self.mlowres = mlowres
        self.bhids = np.array(bhids, dtype=int)
        self.bh_r = np.array(bh_r, dtype=float)
        self.bh_m = np.array(bh_m, dtype=float)
        self.bh_m_sph = np.array(bh_m_sph, dtype=float)
        self.host_ind = host_ind
        self.centre = sub_centre
        if rvir is None:
            self.rvir = -1
        else:
            self.rvir = rvir #kpc

    def compute_rvir(self, snap):
        rvir, _ = pygad.analysis.halo.virial_info(
            snap, center=self.centre
        )
        self.rvir = rvir

        return



def read_subfind_files(folder_path: str, snap_i: int, contamination_exclude_frac = None,
                        recalc: bool = False):
    
    """
    NOTE: This assumes that subfind has been compiled with the following flags:
          SUBFIND
          DENSITY_SPLIT_BY_TYPE=1+2+4+16+32  # Split gas, DM, lowres DM, stars and BH
          WRITE_SUB_IN_SNAP_FORMAT           # Save subfind results in snap format
          SAVE_MASS_TAB
          SUBFIND_BH_INFO
          SUBFIND_NO_UNBIND_CHECK_FOR_BH
          Some of these we added in the commit 

    Read the structure data from SUBFIND files. Saves the info to a pkl file
    which will be used if this is called again. Note that your subfind fields may
    not match what this functions expects them to have.

    TODO known issue(s): Run and creates an empty pkl file even if files
                         are not found

    Args:
        folder_path (str): Path to the directory to scan (output directory, 
        not the /groups_iii folder in it).

        snap_i (int): Number of snapshot
        
        contamination_exclude_frac (float): if a structure has a mass fraction higher
        than this amount of dark matter in low-res particles, the structure will be
        excluded

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
    
    pkl_fname = folder_path + f"/subfind_data_{snap_i:03d}.pkl"
    #check if data is already saved. If not, we will loop through subfind files.
    if not recalc:
        try:
            with open(pkl_fname, 'rb') as f:
                data_stucture = pickle.load(f)
        #This most likely is not best practise but whatever
        except FileNotFoundError:
            data_stucture = read_subfind_files(folder_path, snap_i, contamination_exclude_frac,
                        recalc=True)
        return data_stucture

    print('Looping through subfind files in ', folder_path, 'for snapshot ', snap_i)

   
    # Find files like groups folder
    file_pattern = subfind_fname_start + '.*.hdf5'

    z_merge = np.zeros(0)
    merger_id_pairs = np.empty((0,2))

    files_checked = 0

    Nbh = np.zeros(100000)
    Max_mbh = np.zeros_like(Nbh)
    mstar = np.zeros_like(Nbh)
    mdm = np.zeros_like(Nbh)
    mlowres = np.zeros_like(Nbh)
    pos_x = np.zeros_like(Nbh)
    pos_y = np.zeros_like(Nbh)
    pos_z = np.zeros_like(Nbh)
    subhalo_indices = np.zeros(len(Nbh), dtype=int)

    structures_to_include = 0
    tot_n_subhalos = 0

    a = None
    h = None

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
                
                subhalos = f['Subhalo']
                
                n_groups = len(groups['GroupLen'])
                n_subhalos = len(subhalos['SubhaloLen'])
                sub_offsets = subhalos['SubhaloOffset']
                particles_per_subhalo = subhalos['SubhaloLen']
                
                #TODO make sure units are correct!
                n_tot = 0
                for i_subhalo in range(n_subhalos):
                    subhalo_mass_by_type = subhalos['SMST'][i_subhalo]
                    subhalo_mstar = subhalo_mass_by_type[4]/h*1e10
                    
                    n_bh_subhalo = subhalos['SubhaloNBH'][i_subhalo]
                    
                    #TODO what limit to use? only write data for systems with BH(s)?
                    if subhalo_mstar/h > 0.001:
                        
                        max_mbh_subhalo = subhalos['SubhaloMaxMBH'][i_subhalo]/h*1e10
                        subhalo_mdm = subhalo_mass_by_type[1]/h*1e10
                        subhalo_mlowres = subhalo_mass_by_type[2]/h*1e10
                        subhalo_pos = subhalos['SubhaloPos'][i_subhalo]/h
                        pos_x[structures_to_include] = subhalo_pos[0]
                        pos_y[structures_to_include] = subhalo_pos[1]
                        pos_z[structures_to_include] = subhalo_pos[2]
                        #actually, dm mass should be sum of type 1 + type 2
                        subhalo_mdm += subhalo_mlowres
                        subhalo_mbh = subhalo_mass_by_type[5]/h*1e10
                        Nbh[structures_to_include] = n_bh_subhalo
                        mstar[structures_to_include] = subhalo_mstar
                        mdm[structures_to_include] = subhalo_mdm
                        mlowres[structures_to_include] = subhalo_mlowres
                        Max_mbh[structures_to_include] = max_mbh_subhalo
                        subhalo_indices[structures_to_include] = i_subhalo + tot_n_subhalos
                        #Max_mbh[structures_to_include] = subhalo_mbh

                        structures_to_include += 1
                    
                    elif n_bh_subhalo > 0:
                        max_mbh_subhalo = subhalos['SubhaloMaxMBH'][i_subhalo]/h*1e10
                        #print('No stellar matter in structure but ', n_bh_subhalo, 'BHs, max mass is ', max_mbh_subhalo)
                
                tot_n_subhalos += n_subhalos
    
        except OSError as e:
            print(f"Skipping {filepath}: {e}")
            continue

        files_checked += 1

    #cut

    mdm = mdm[:structures_to_include]
    Nbh = Nbh[:structures_to_include]
    mstar = mstar[:structures_to_include]
    Max_mbh = Max_mbh[:structures_to_include]
    mlowres = mlowres[:structures_to_include]
    pos_x = pos_x[:structures_to_include]
    pos_y = pos_y[:structures_to_include]
    pos_z = pos_z[:structures_to_include]
    subhalo_indices = subhalo_indices[:structures_to_include]

    if contamination_exclude_frac is not None:
        contamination_mask = mlowres/mdm < contamination_exclude_frac
        mdm = mdm[contamination_mask]
        Nbh = Nbh[contamination_mask]
        mstar = mstar[contamination_mask]
        Max_mbh = Max_mbh[contamination_mask]
        pos_x = pos_x[contamination_mask]
        pos_y = pos_y[contamination_mask]
        pos_z = pos_z[contamination_mask]
        
    structure_data = dict(
        mdm = mdm,
        mstar = mstar,
        max_mbh = Max_mbh,
        Nbh = Nbh,
        mlowres = mlowres,
        pos_x = pos_x,
        pos_y = pos_y,
        pos_z = pos_z,
        subhalo_ind = subhalo_indices
    )
        
    subhalo_data = np.ndarray(len(Nbh), dtype=np.object)
    

    #save subfind data to pkl format for faster access
    print('Saving the read subfind data to ', pkl_fname)
    with open(pkl_fname, 'wb') as f:
        pickle.dump(structure_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    

    return structure_data

def get_bh_ids_in_each_subsystem(folder_path: str, snap_base_fname: str, snap_i: int, 
                                    contamination_exclude_frac = None, recalc: bool = False):
    
    """
    NOTE: This assumes that subfind has been compiled with the following flags:
          SUBFIND
          DENSITY_SPLIT_BY_TYPE=1+2+4+16+32  # Split gas, DM, lowres DM, stars and BH
          WRITE_SUB_IN_SNAP_FORMAT           # Save subfind results in snap format
          SAVE_MASS_TAB
          SUBFIND_BH_INFO
          SUBFIND_NO_UNBIND_CHECK_FOR_BH
          Some of these we added in the commit AAA

    This would be easiest if subfind is compiled with SUBFIND_SAVE_PARTICLELISTS,
    but can be done using the snapshot and the current saved subfind info.
    Loop through ID info for each particle using subfind data.
    For each BH, save the number of substructure it belongs to.

    TODO improve the input params (snap info given akwardly)

    Args:
        folder_path (str): Path to the directory to scan (output directory, 
        not the /groups_iii folder in it).

        snap_base_fname (str): Filename of snapshot (needed to get info wabout mass and position based on IDs)

        snap_i (int): Number of snapshot
        
        contamination_exclude_frac (float): if a structure has a mass fraction higher
        than this amount of dark matter in low-res particles, the structure will be
        excluded

        recalc (bool): loop through subfind files again even if pkl file exists

    Returns:
        Currently returns for each bh mass, separation from centre and host id.
        Would be better to return data for each subhalo (their BH ids, masses and separations)
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
    
    pkl_fname = folder_path + f"/subfind_bh_data_{snap_i:03d}.pkl"
    #check if data is already saved. If not, we will loop through subfind files.
    if not recalc:
        try:
            with open(pkl_fname, 'rb') as f:
                data_stucture = pickle.load(f)
        #This most likely is not best practise but whatever
        except FileNotFoundError:
            data_stucture = get_bh_ids_in_each_subsystem(folder_path, snap_base_fname, snap_i, 
                                    contamination_exclude_frac, recalc=True)
        return data_stucture


   
    # Find files like groups folder
    file_pattern = subfind_fname_start + '.*.hdf5'

    z_merge = np.zeros(0)
    merger_id_pairs = np.empty((0,2))

    files_checked = 0

    #TODO can just BHs be loaded instead of the full snap?
    snap = pygad.Snapshot(snap_base_fname + f'_{snap_i:03d}.hdf5')
    snap.to_physical_units()
    bhs = snap.bh
    #bhs = pygad.Snapshot(snap_base_fname + f'_{snap_i:03d}.hdf5').bh
    #Default value for host number is -1 (BHs not assigned to any system will have this value)
    subnum = np.ones(len(bhs), dtype=int)
    subnum = subnum * -1
    #separation from the centre (units? default would be ckpc/h)
    r_centre = np.zeros(len(bhs))

    fof_offset_this_file = 0
    subfind_offset_this_file = 0

    max_part_num = 0
    max_offset = 0
    total_number_of_ids = 0
    total_number_of_parts_in_structures = 0

    #let's do this with one loop, nees to allocate quite big arrays,
    #will NOT work with very big simulation outputs
    #might need two loops over subfind files, as
    #needs also number of substructures
    subfind_full_idlist = np.zeros(len(snap), dtype=int)

    hostlist=dict()
    #need also amount
    subfind_total = 0

    max_subs = 100000
    pos_x = np.zeros(max_subs)
    pos_y = np.zeros(max_subs)
    pos_z = np.zeros(max_subs)
    offsets = np.zeros(max_subs, dtype=int)
    lens = np.zeros(max_subs, dtype=int)
    Nbhs = np.zeros(max_subs, dtype=int)
    mstar = np.zeros_like(pos_x)
    mdm = np.zeros_like(pos_x)
    mlowres = np.zeros_like(pos_x)
    host_ind = np.zeros(max_subs, dtype=int)

    a = None
    h = None

    print('Looping through subfind files in ', folder_path, 'for snapshot ', snap_base_fname, 'number ', snap_i)

    i = 0
    
    #we have folder/sub_snap_i.NUMBER.hdf5
    #subfind_fname_start has everything before the number we sort with except the final dot
    len_start = len(subfind_fname_start) + 1
    for subfind_fname in sorted(glob.glob(file_pattern), key=lambda x: int(x[len_start:-5])):
        try:
            with h5py.File(subfind_fname, 'r') as f:
                header_info = f['Header'].attrs
                if a is None:
                    a = header_info['Time']
                if h is None:
                    h = header_info['HubbleParam']
                ids = f['IDs']['PID ']

                subfind_full_idlist[total_number_of_ids:total_number_of_ids+len(ids)] = ids
                groups = f['Group']
                
                subhalos = f['Subhalo']
                
                n_groups = len(groups['GroupLen'])
                fof_offsets = groups['GroupOffset']
                sub_offsets = subhalos['SubhaloOffset']
                particles_per_subhalo = subhalos['SubhaloLen']
                n_subhalos = len(subhalos['SubhaloLen'])
                nbhs_subhalos = subhalos['SubhaloNBH']
                subhalo_host_numbers =subhalos['SubhaloGrNr']

                sub_pos = subhalos['SubhaloPos']
                subhalos_mass_by_type = subhalos['SMST']
                offsets[subfind_total:subfind_total+n_subhalos] = sub_offsets
                lens[subfind_total:subfind_total+n_subhalos] = particles_per_subhalo
                pos_x[subfind_total:subfind_total+n_subhalos] = sub_pos[:,0]
                pos_y[subfind_total:subfind_total+n_subhalos] = sub_pos[:,1]
                pos_z[subfind_total:subfind_total+n_subhalos] = sub_pos[:,2]
                Nbhs[subfind_total:subfind_total+n_subhalos] = nbhs_subhalos
                host_ind[subfind_total:subfind_total+n_subhalos] = subhalo_host_numbers
                
                mstar[subfind_total:subfind_total+n_subhalos] = subhalos_mass_by_type[:,4]/h*1e10
                mdm[subfind_total:subfind_total+n_subhalos] = (subhalos_mass_by_type[:,1]+subhalos_mass_by_type[:,2])/h*1e10
                mlowres[subfind_total:subfind_total+n_subhalos] = subhalos_mass_by_type[:,2]/h*1e10


                max_part_num = max(max_part_num, max(particles_per_subhalo))
                max_offset = max(max_offset, max(np.diff(sub_offsets)))
                total_number_of_ids += len(ids)
                total_number_of_parts_in_structures += sum(particles_per_subhalo)
                subfind_total += n_subhalos
                
        except OSError as e:
            print(f"Something went wrong reading file {filepath}: {e}")
            quit()

    subfind_full_idlist = subfind_full_idlist[:total_number_of_ids]

    substruct_info = dict()
    #we now have the position, offsets and length for each structure, now we can get the
    #IDs for each substructure, and check if any IDs match BH IDs.
    for i_subhalo in range(subfind_total):
        #print(i_subhalo)
        offset = offsets[i_subhalo]
        npart = lens[i_subhalo]
        ids_subhalo = subfind_full_idlist[offset:offset+npart]
        
        #check if we have bhs whose id matches any of the ids_subhalo
        sub_x = pos_x[i_subhalo]
        sub_y = pos_y[i_subhalo]
        sub_z = pos_z[i_subhalo]
        sub_mdm = mdm[i_subhalo]
        sub_mstar = mstar[i_subhalo]
        sub_lowres = mlowres[i_subhalo]
        sub_host_ind = host_ind[i_subhalo]

        if sub_mstar < 1e7:
            continue

        if contamination_exclude_frac is not None:
            if sub_lowres/sub_mdm > contamination_exclude_frac:
                continue
        

        subhalo_center = pygad.UnitArr([sub_x, sub_y, sub_z], 'ckpc h_0**-1').in_units_of('kpc', subs=snap)
        bhs_this_sub = bhs[np.in1d(bhs['ID'], ids_subhalo)]
        if len(bhs_this_sub) != Nbhs[i_subhalo]:
            print('Mismatch in BH number!',len(bhs_this_sub),  Nbhs[i_subhalo], i_subhalo)
            print(offset, npart)
            quit()
        if len(bhs_this_sub) > 0:
            bhids = bhs_this_sub['ID']
            #If we go with the substruct_info
            #def __init__(self, bhids, bh_m, bh_r, mstar, mdm, mlowres):
            subhalo_bh_positions = bhs_this_sub['pos']
            subhalo_bh_masses = bhs_this_sub['mass']
            subhalo_bh_sph_masses = bhs_this_sub['BH_Mass']*1e10/h
            dists = np.linalg.norm(subhalo_bh_positions-subhalo_center[np.newaxis], axis=-1)
            
            substruct_info[i_subhalo] = Subhalo_bh_info(bhids, subhalo_bh_masses, subhalo_bh_sph_masses, 
                                        dists, sub_mstar, sub_mdm, sub_lowres, sub_host_ind, subhalo_center)

    print('Saving the read subfind data to ', pkl_fname)
    with open(pkl_fname, 'wb') as f:
        pickle.dump(substruct_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    del snap
    pygad.gc_full_collect()

    return substruct_info


#Subfind in gadget3 does not save/calculate virial radius for each subhalo.
#Instead, it is saved for each FoF group.
#Calculate the virial radius for each subhalo and save the info
#This might take a while to run :)
def add_rvir_info_for_each_subhalo(folder_path: str, snap_i: int, 
                        snap_base_fname: str, recalc: bool = False):
    """
    Calculate virial radius from the snapshot for each subhalo from SUBFIND files. 
    Saves the info to a pkl file which will be used if this is called again. 
    Note that your subfind fields may not match what this functions expects them to have.

    TODO known issue(s): Run and creates an empty pkl file even if files
                         are not found

    Args:
        folder_path (str): Path to the directory to scan (output directory, 
        not the /groups_iii folder in it).

        snap_i (int): Number of snapshot

        snap_base_fname (str): Filename of snapshot (needed to get info about mass and position based on IDs)
        
        recalc (bool): loop through subfind files again even if pkl file exists


    Returns:
        
    """

    snap = pygad.Snapshot(snap_base_fname + f'_{snap_i:03d}.hdf5')
    snap.to_physical_units()
    z = snap.redshift
    a = 1/(z+1)

    #the saved subfind data has positions saved in ckpc
    #Let's add info to the files which also have BH-specific info,
    #since we most likely want to look at r_wander/r_vir for the BHs
    #subfind_data = read_subfind_files(folder_path, snap_i, recalc=False)
    #for now, hardcoded exclusion of >1% contaminated systems

    #Load saved info
    bh_host_info =  get_bh_ids_in_each_subsystem(folder_path, snap_base_fname, snap_i, contamination_exclude_frac = 0.01, recalc = recalc)

    print('Looping through subhalos to add virial radii info, may take a while...')

    #add virial radius to the info
    for subhalo_id, subhalo_info in bh_host_info.items():
        print(subhalo_id)
        #If the info has already been added, we can just return
        if subhalo_info.rvir > 0:
            print('Saved data already has virial radii info included, no need to loop over again.')
            del snap
            pygad.gc_full_collect()
            return
        
        subhalo_info.compute_rvir(snap)
        #center = subhalo_info.centre #in units of kpc
        #rvir, _ = pygad.analysis.halo.virial_info(snap, center=center)
        #subhalo_info.rvir = rvir

    #overwrite previous saved data with the new data
    pkl_fname = folder_path + f"/subfind_bh_data_{snap_i:03d}.pkl"
    with open(pkl_fname, 'wb') as f:
        pickle.dump(bh_host_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Virial radii info added to ', pkl_fname)

    del snap
    pygad.gc_full_collect()

    return
