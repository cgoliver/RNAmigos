# RNAmigos: RNA Small Molecule Ligand Prediction

This repository is an implementation of ligand prediction from an RNA base pairing network.


> :warning: If you just want to obtain the data used in the paper, use the dataset on [Zenodo](https://zenodo.org/record/8338267).

Please cite:

```
@article{oliver2020augmented,
  title={Augmented base pairing networks encode RNA-small molecule binding preferences},
  author={Oliver, Carlos and Mallet, Vincent and Gendron, Roman Sarrazin and Reinharz, Vladimir and Hamilton, William L and Moitessier, Nicolas and Waldisp{\"u}hl, J{\'e}r{\^o}me},
  journal={Nucleic acids research},
  volume={48},
  number={14},
  pages={7690--7699},
  year={2020},
  publisher={Oxford University Press}
}
```

If you just want to use the tool non-programmatically or don't need to train your own model you can check the online web-server implementation [here](http://rnamigos.cs.mcgill.ca/).


![](images/rnamigos.png)

We implement three major components:

* `data_processor`: building a training set
* `learning`: RGCN training
* `tools`: general tools such as graph drawing, loading, etc.
* `post`: validation and visualization of results 

See README in each folder for details on how to use each component.

## Requirements

* Python >= 3.6
* Pytorch
* dgl
* OpenBabel
* BioPython
* tqdm
* networkx >= 2.1


You can automatically install all the dependencies with Anaconda using the following command:

```
conda env create -f environment.yml
```

## Usage

### Extracting the annotated data

Extract the annotated data and place it in the annotations folder if it has not yet been created.

```
cd data
tar -xzvf pockets_nx_symmetric_orig.tar.gz
cd ..
mkdir data/annotated
mv data/pockets_nx_symmetric_orig data/annotated
```

### Loading a trained model 


* You can use the model used for the paper, or load a trained model you trained yourself (see next section)

Making predictions for every graph in a folder.

Create a script in the root of the repository with the following code:

```
from tools.learning_utils import inference_on_dir

graph_dir = "data/annotated/pockets_nx_symmetric_orig"

fp_pred,_ = inference_on_dir("rnamigos", graph_dir)
```

`fp_pred` is a N x 166 matrix where N is the number of graphs in `graph_dir` and each column corresponds to a fingerprint index.
The raw output is a probability, so if you want a binary fingerprint, do as above and use the `>0.5` filter. 

### Training your own model

A basic example is training on the annotated graphs inside `data/annotated` on default settings.

The trained model and logs will be saved under the ID specified with the `-n` flag.

```
$ python learning/main.py -da <name of annotated data folder> -n <run_id>
```

For a full list of command-line options:

```
$ python learning/main.py -h
```


### Inference on your own structures

If you have your own RNA structures you want to run the model on, check out `inference.py`.

0. Make sure you have rnaglib installed (`pip install -r requirements.txt`)
1. Create a folder where you store the .cif files for your RNAs. (we have some sample ones in `data/my_pdbs`
2. `mkdir data/my_graphs` where we will store the annotated graphs
3. Convert the structures to graphs using `fr3d_to_graph()` from [rnaglib](https://rnaglib.readthedocs.io)
4. Launch the model. You will get one fingerprint prediction for each graph. You can use this fingerprint to screen for similar ones in a ligand database.

> :warning: RNAmigos is **not** a pocket finding method. We assume that the .cif you pass only contains residues you consider to belong to the binding site of interest.

> :warning: This model was trained as a proof of concept and has not been evaluated on PDBs outside the publicly available ones at the time of publication. We therefore cannot guarantee the quality of the predictions. If you want to use the latest models currently under development please contact me (oliver@biochem.mpg.de)

```python

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

```
