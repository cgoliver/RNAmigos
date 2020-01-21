"""
    Swap the fingerprints around from an annotated folder.
"""
import os, pickle


def scramble_fingerprints(anot_dir, dump_dir):
    """
        Assign a fingerprint to each graph chosen at random from all other
        fingerprints in annot_dir.
    """

    datas = os.listdir(annot_dir)
    indices = range(len(datas))
    fps = []
    for i,g in datas:
        g,tree,ring,fp = pickle.load(open(os.path.join(annot_dir, g)))
        fps.append(fp)
    for i,g in datas:
        pickle.dump((g,tree,ring,fps[random.choice(indices)]), os.path.join(dump_dir, g))
    pass
