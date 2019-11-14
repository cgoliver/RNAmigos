import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parallel", default=True, help="If we don't want to run thing in parallel", action='store_false')
parser.add_argument("-da", "--annotated_data", default='samples')
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="choose the batch size")
parser.add_argument("-nw", "--workers", type=int, default=20, help="Number of workers to load data")
parser.add_argument("-wt", "--wall_time", type=int, default=None, help="Max time to run the model")
parser.add_argument("-n", "--name", type=str, default='default_name', help="Name for the logs")
parser.add_argument("-t", "--timed", help="to use timed learning", action='store_true')
parser.add_argument("-ep", "--num_epochs", type=int, help="number of epochs to train", default=3)
parser.add_argument("-ml", "--motif_lam", type=float, help="motif lambda", default=1.0)
parser.add_argument("-rl", "--reconstruction_lam", type=float, help="reconstruction lambda", default=1.0)
parser.add_argument('-ad','--attributor_dims', nargs='+', type=int, help='Dimensions for attributor.', default=[128,64,32])
parser.add_argument('-ed','--embedding_dims', nargs='+', type=int, help='Dimensions for embeddings.', default=[128,128,128,128,128,128])
parser.add_argument('-sf','--sim_function', type=str, help='Node similarity function (Supported Options: R_1, IDF).', default="R_1")
parser.add_argument('-eo','--embed_only', type=int, help='Number of epochs to train embedding network before starting attributor. If -eo > num_epochs, no attributions trained. If < 0, attributor and embeddings always on. If 0 <= -eo <- num_epochs, switch attributor ON and embeddings OFF.', default=-1)
parser.add_argument('-w', '--warm_start', type=str, default=None, help='Path to pre-trained model.')
args = parser.parse_args()

print(f"OPTIONS USED: {args}")
# Torch impors
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# Homemade modules
if args.timed:
    import learning.timed_learning as learn
else:
    import learning.learning as learn
from learning.loader import Loader
from learning.rgcn import Model

from learning.utils import mkdirs


print('Done importing')

'''
Hardware settings
'''

# torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# This is to create an appropriate number of workers, but works too with cpu
if args.parallel:
    used_gpus_count = torch.cuda.device_count()
else:
    used_gpus_count = 1

print(f'Using {used_gpus_count} GPUs')

'''
Dataloader creation
'''

annotated_file = 'data/annotated'
annotated_name = args.annotated_data
annotated_path = os.path.join(annotated_file, annotated_name)
print(f'Using {annotated_path} as the pocket inputs')

# batch_size = 8
batch_size = args.batch_size
num_workers = args.workers

dims = args.embedding_dims
attributor_dims = args.attributor_dims

print(f'Using batch_size of {batch_size}')

loader = Loader(emb_size=dims[0], annotated_path=annotated_path,
                batch_size=batch_size, num_workers=num_workers,
                sim_function=args.sim_function)

train_loader, _, test_loader = loader.get_data()

print('Created data loader')

if len(train_loader) == 0 & len(test_loader) == 0:
    raise ValueError('there are not enough points compared to the BS')

# a = time.perf_counter()
# for batch_idx, (inputs, labels) in enumerate(train_loader):
#     if not batch_idx % 20:
#         print(batch_idx, time.perf_counter() - a)
#         a = time.perf_counter()
# print('Done in : ', time.perf_counter() - a)

'''
Model loading
'''


#sanity checks
if dims[-1] != attributor_dims[0]:
    raise ValueError(f"Final embedding size must match first attributor dimension: {dims[-1]} != {attributor_dims[0]}")


motif_lam = args.motif_lam
reconstruction_lam = args.reconstruction_lam

model = Model(dims=dims, attributor_dims=attributor_dims,
              num_rels=loader.num_edge_types,
              num_bases=-1)

#if pre-trained initialize matching layers
if args.warm_start:
    print("warm starting")
    m = torch.load(args.warm_start)['model_state_dict']
    #remove keys not related to embeddings
    for k in list(m.keys()):
        if 'layers' not in k:
            del m[k]
    # print(m['model_state_dict']['layers.2.weight'])
    missing = model.load_state_dict(m, strict=False)
    print(missing)

model = model.to(device)

print(f'Using {model.__class__} as model')

if used_gpus_count > 1:
    model = torch.nn.DataParallel(model)

'''
Optimizer instanciation
'''

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters())
# print(list(model.named_parameters()))
# raise ValueError
# optimizer.add_param_group({'param': model.embeddings})

# optimizer = optim.SGD(model.parameters(), lr=1)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

'''
Experiment Setup
'''

name = args.name
result_folder, save_path = mkdirs(name)
writer = SummaryWriter(result_folder)
print(f'Saving result in {name}')


print(test_loader.dataset.indices)
print(dir(test_loader.dataset.dataset))
print(test_loader.dataset.dataset.all_graphs)

import numpy as np

all_graphs = np.array(test_loader.dataset.dataset.all_graphs)
test_inds = test_loader.dataset.indices
train_inds = train_loader.dataset.indices

pickle.dump(({'test': all_graphs[test_inds], 'train': all_graphs[train_inds]}),
                open(os.path.join('results', name, 'splits.p'), 'wb'))

wall_time = args.wall_time

'''
Run
'''
num_epochs = args.num_epochs

learn.train_model(model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  device=device,
                  train_loader=train_loader,
                  test_loader=test_loader,
                  save_path=save_path,
                  writer=writer,
                  num_epochs=num_epochs,
                  wall_time=wall_time,
                  reconstruction_lam=reconstruction_lam,
                  motif_lam=motif_lam,
                  embed_only=args.embed_only)
