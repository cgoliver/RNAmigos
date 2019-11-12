"""
    Picking ligands from set of sphere selections.
"""
from collections import defaultdict
import pickle

def ligand_filter(lig_name):
    if lig_name in ['UNX', 'OHX', 'MPD', 'SO4', 'IRI', 'PG4']:
        return False
    return True

def get_valids(d, max_dist, min_conc, min_size=4):
    num_OK = 0
    ok_ligs = defaultdict(list)
    for pdb, ligands in d.items():
        for lig_id,lig_cuts in ligands:
            lig_name = lig_id.split(":")[2]
            for c in lig_cuts:
                tot = c['rna'] + c['protein']
                if tot == 0:
                    continue
                if c['rna'] < min_size:
                    continue
                rna_conc = c['rna'] / tot
                if (rna_conc >= min_conc and c['cutoff'] == max_dist) and ligand_filter(lig_name):
                    num_OK += 1
                    ok_ligs[pdb].append(lig_id)
                    continue

    return ok_ligs

def ligs_to_txt(d, dest="../data/ligs.txt"):
    """
        Write selected ligands to text file for Chimera.
    """
    with open(dest, "w") as o:
        for pdb, ligs in d.items():
            o.write(" ".join([pdb, *ligs]) + "\n")
    pass
def lig_stats(d):
    """
        Compute stats on resulting ligand dictionary.
    """
    unique_ligs = set()
    total_sites = 0
    one_lig_per_pdb = 0
    for pdb, ligands in d.items():
        ligands_per_pdb = set()
        for l in ligands:
            lig_id = l.split(":")[2]
            ligands_per_pdb.add(lig_id)
            unique_ligs.add(lig_id)
            total_sites += 1
        one_lig_per_pdb += len(ligands_per_pdb)
    print(f">>> PDBs: {len(d)}")
    print(f">>> Total sites: {total_sites}")
    print(f">>> Unique ligands: {len(unique_ligs)}")
    print(f">>> Unique ligand per PDB: {one_lig_per_pdb}")
    pass
def lig_dict_filter(d):
    """
        Keep one copy of each unique ligand per PDB.
    """
    filtered = defaultdict(list)
    for pdb, ligands in d.items():
        seen_ligs = set() 
        for l in ligands:
            lig_id = l.split(":")[2]
            if lig_id not in seen_ligs:
                filtered[pdb].append(l)
                seen_ligs.add(lig_id)
    return filtered

if __name__ == "__main__":
    d = pickle.load(open('../data/lig_dict_c.p', 'rb'))
    c =   8
    conc = 0.6
    ligs = get_valids(d, c, conc)
    ligs = lig_dict_filter(ligs)
    print(ligs)
    pickle.dump(ligs, open('../data/lig_dict_c_8A_06rna.p', 'wb'))
