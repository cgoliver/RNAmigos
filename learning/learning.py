import time
import torch
import torch.nn.functional as F
import sys
import dgl

#debug modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
##

if __name__ == '__main__':
    sys.path.append('../')

from learning.attn import get_attention_map 

def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)

    return g

def set_gradients(model, embedding=True, attributor=True):
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        name = name.split('.')[0]
        if name in ['embeddings', 'attributor']:
            p.requires_grad = attributor
        if name == 'layers':
            p.requires_grad = embedding
    pass

def print_gradients(model):
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        print(name, p.grad)
    pass
def test(model, test_loader, device, reconstruction_lam, motif_lam):
    """
    Compute accuracy and loss of model over given dataset
    :param model:
    :param test_loader:
    :param test_loss_fn:
    :param device:
    :return:
    """
    model.eval()
    test_loss,  motif_loss_tot, recons_loss_tot = (0,) * 3
    test_size = len(test_loader)
    for batch_idx, (graph, K, fp) in enumerate(test_loader):
        # Get data on the devices
        K = K.to(device)
        fp = fp.to(device)
        K = torch.ones(K.shape).to(device) - K
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        out, attributions = model(graph)
        loss, reconstruction_loss, motif_loss = compute_loss(model=model, attributions=attributions, fp=fp,
                                                                         out=out, K=K, device=device,
                                                                         reconstruction_lam=reconstruction_lam,
                                                                         motif_lam=motif_lam)
        del K
        del fp
        del graph

        recons_loss_tot += reconstruction_loss.item()
        motif_loss_tot += motif_loss.item()
        test_loss += loss.item()

        del loss
        del reconstruction_loss
        del motif_loss

    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    return test_loss / test_size, motif_loss_tot / test_size, recons_loss_tot / test_size

def compute_loss(model, attributions, out, K, fp,
        device, reconstruction_lam, motif_lam):
    """
    Compute the total loss and returns scalar value for each contribution of each term. Avoid overwriting loss terms
    :param model:
    :param attributions:
    :param out:
    :param K:
    :param device:
    :param reconstruction_lam:
    :param motif_lam:
    :return:
    """

    # reconstruction loss
    K_predict = torch.norm(out[:, None] - out, dim=2, p=2)
    reconstruction_loss = torch.nn.MSELoss()
    reconstruction_loss = reconstruction_loss(K_predict, K)
    motif_loss = torch.nn.BCELoss()
    motif_loss = motif_loss(attributions, fp)


    loss = reconstruction_lam * reconstruction_loss + motif_lam * motif_loss
    return loss, reconstruction_loss, motif_loss


def train_model(model, criterion, optimizer, device, train_loader, test_loader, save_path,
                writer=None, num_epochs=25, wall_time=None,
                reconstruction_lam=1, motif_lam=1, embed_only=-1):
    """
    Performs the entire training routine.
    :param model: (torch.nn.Module): the model to train
    :param criterion: the criterion to use (eg CrossEntropy)
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param device: the device on which to run
    :param train_loader: dataloader for training
    :param test_loader: dataloader for validation
    :param save_path: where to save the model
    :param writer: a Tensorboard object (defined in utils)
    :param num_epochs: int number of epochs
    :param wall_time: The number of hours you want the model to run
    :param reconstruction_lam: how much to enforce pariwise similarity conservation
    :param motif_lam: how much to enforce motif assignment
    :param embed_only: number of epochs before starting attributor training.
    :return:
    """

    epochs_from_best = 0
    early_stop_threshold = 80

    start_time = time.time()
    best_loss = sys.maxsize

    motif_lam_orig = motif_lam
    reconstruction_lam_orig = reconstruction_lam

    #if we delay attributor, start with attributor OFF
    #if <= -1, both always ON.
    if embed_only > -1:
        print("Switching attriutor OFF. Embeddings still ON.")
        set_gradients(model, attributor=False)
        motif_lam = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Training phase
        model.train()

        #switch off embedding grads, turn on attributor
        if epoch == embed_only:
            print("Switching attributor ON, embeddings OFF.")
            set_gradients(model, embedding=False, attributor=True)
            reconstruction_lam = 0
            motif_lam = motif_lam_orig

        running_loss = 0.0

        time_epoch = time.perf_counter()

        num_batches = len(train_loader)

        for batch_idx, (graph, K, fp) in enumerate(train_loader):

            # Get data on the devices
            batch_size = len(K)
            K = K.to(device)
            fp = fp.to(device)
            K = torch.ones(K.shape).to(device) - K
            graph = send_graph_to_device(graph, device)

            # Do the computations for the forward pass
            out, attributions = model(graph)

            # Compute the loss with proper summation, solves the problem ?
            loss, reconstruction_loss, motif_loss = compute_loss(model=model, attributions=attributions, fp=fp,
                                                                             out=out, K=K, device=device,
                                                                             reconstruction_lam=reconstruction_lam,
                                                                             motif_lam=motif_lam)
            if(batch_idx==0):
                # Att has shape h, dest_nodes, src_nodes
                # Sum of attention[1]=1 (attn weights sum to one for destination node)
                
                # Transform graph to RDKit molecule for nice visualization
                graphs = dgl.unbatch(graph)
                g0=graphs[0]
                n_nodes = len(g0.nodes)
                att= get_attention_map(g0, src_nodes=g0.nodes(), dst_nodes=g0.nodes(), h=1)
                att_g0 = att[0] # get attn weights only for g0
                
                # Select atoms with highest attention weights and plot them 
                tops = np.unique(np.where(att_g0>0.55)) # get top atoms in attention
                print(tops)
                # mol = nx_to_mol(g0, rem, ram, rchim, rcham)
                # img=highlight(mol,list(tops))f batch_idx == 0:

            del K
            del fp
            del graph
            # Backward
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Metrics
            batch_loss = loss.item()
            running_loss += batch_loss



            # running_corrects += labels.eq(target.view_as(out)).sum().item()
            if batch_idx % 20 == 0:
                time_elapsed = time.time() - start_time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Time: {:.2f}'.format(
                    epoch + 1,
                    (batch_idx + 1) * batch_size,
                    num_batches * batch_size,
                    100. * (batch_idx + 1) / num_batches,
                    batch_loss,
                    time_elapsed))

                # tensorboard logging
                writer.add_scalar("Training batch loss", batch_loss,
                                  epoch * num_batches + batch_idx)
                writer.add_scalar("Training reconstruction loss", reconstruction_loss.item(),
                                  epoch * num_batches + batch_idx)
                writer.add_scalar("Training motif loss", motif_loss.item(),
                                  epoch * num_batches + batch_idx)

            del loss
            del reconstruction_loss
            del motif_loss

        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        # Log training metrics
        train_loss = running_loss / num_batches
        writer.add_scalar("Training epoch loss", train_loss, epoch)

        # train_accuracy = running_corrects / num_batches
        # writer.log_scalar("Train accuracy during training", train_accuracy, epoch)

        # Test phase
        test_loss, motif_loss, reconstruction_loss = test(model, test_loader, device, reconstruction_lam, motif_lam)

        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        writer.add_scalar("Test loss during training", test_loss, epoch)
        writer.add_scalar("Test reconstruction loss", reconstruction_loss, epoch)
        writer.add_scalar("Test motif loss", motif_loss, epoch)

        # writer.log_scalar("Test accuracy during training", test_accuracy, epoch)

        # Checkpointing
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_from_best = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion
            }, save_path)

        # Early stopping
        else:
            epochs_from_best += 1
            if epochs_from_best > early_stop_threshold:
                print('This model was early stopped')
                break

        # Sanity Check
        if wall_time is not None:
            # Break out of the loop if we might go beyond the wall time
            time_elapsed = time.time() - start_time
            if time_elapsed * (1 + 1 / (epoch + 1)) > .95 * wall_time * 3600:
                break
        del test_loss
        del reconstruction_loss
        del motif_loss

    return best_loss


def make_predictions(data_loader, model, optimizer, model_weights_path):
    """
    :param data_loader: an iterator on input data
    :param model: An empty model
    :param optimizer: An empty optimizer
    :param model_weights_path: the path of the model to load
    :return: list of predictions
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    predictions = []

    for batch_idx, inputs in enumerate(data_loader):
        inputs = inputs.to(device)
        predictions.append(model(inputs))
    return predictions


if __name__ == "__main__":
    pass
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='../data/testset')
# parser.add_argument('--out_dir', default='Submissions/')
# parser.add_argument(
#     '--model_path', default='results/base_wr_lr01best_model.pth')
# args = parser.parse_args()
# make_predictions(args.data_dir, args.out_dir, args.model_path)
