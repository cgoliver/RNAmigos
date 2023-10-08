import os
import pickle
from pathlib import Path

from rnaglib.prepare_data import fr3d_to_graph
from tools.learning_utils import inference_on_dir

pdb_dir = "data/my_pdbs"
graph_dir = "data/my_graphs"

# make the graphs and dump them
for p in os.listdir(pdb_dir):
    g = fr3d_to_graph(Path(pdb_dir, p))
    pickle.dump(g, open(Path(graph_dir, g.graph['pdbid'] + '.p'), 'wb'))

fp_pred,_ = inference_on_dir("rnamigos", graph_dir, pocket_only=True)
graphs = os.listdir(graph_dir)

for i in range(len(graphs)):
    print(f"Graph {graphs[i]} predicted fingerprint: {fp_pred[0][i]}")
