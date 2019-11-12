"""
    Functions for generated decoys.
"""

import pickle

def smiles_2_text(smiles_dict):
    with open("decoys_plz.txt", "w") as sm:
        for _, v in smiles_dict.items():
            sm.write(f"{v} \n")
    pass

if __name__ == "__main__":
    smiles_dict = pickle.load(open("../data/smiles_ligs_dict.p", "rb"))
    smiles_2_text(smiles_dict)
    pass
