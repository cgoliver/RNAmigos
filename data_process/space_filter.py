"""
    Picking ligands from set of sphere selections.
"""
from collections import defaultdict
import pickle

def ligand_filter(name):
    if name in ['IRI', 'UNX']:
        return False
    return True

def get_valids(d, max_dist, min_conc, min_size=4):
    num_OK = 0
    ok_ligs = defaultdict(list)
    for pdb, ligands in d.items():
        for lig_id,lig_cuts in ligands:
            lig_name = lig_id.split(":")[1]
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
if __name__ == "__main__":
    d = pickle.load(open('../data/lig_dict_c.p', 'rb'))
    c = 10
    conc = .8
    ligs = get_valids(d, c, conc)
    pickle.dump(ligs, open('../data/lig_dict_c_10A_08rna.p', 'wb'))
