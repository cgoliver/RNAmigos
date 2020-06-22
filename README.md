# RNAmigos: RNA Small Molecule Ligand Prediction

This repository is an implementation of ligand prediction from an RNA base pairing network.

> Augmented base pairing networks encode RNA-small molecule binding preferences. 
> Oliver C., Mallet V., Sarrazin Gendron, R., Reinharz V., Hamilton L W., Moitessier N., Waldispuhl J.
> BiorXiv, 2020.
> [[Paper]](https://www.biorxiv.org/content/10.1101/701326v3)

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

Loading the fully trained RNAmigos model and using standard pytorch API:

```

from tools.learning_utils import load_model

model,meta = load_model('data/rnamigos')
nx_graph, dgl_graph = nx_to_dgl(g, edge_map, nucs=nucs)
with torch.no_grad():
	predicted_fingerprint,_ = model(graph)
fp_pred = fp_pred.detach().numpy() > 0.5
```

Making predictions for every graph in a folder.

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


