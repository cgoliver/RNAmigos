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
from learning.utils import dgl_to_nx
from post.drawing import rna_draw

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
def test(model, test_loader, device):
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
        fp_pred = model(graph)
        loss = model.compute_loss(fp, fp_pred)
        del K
        del fp
        del graph

        test_loss += loss.item()

        del loss

    return test_loss / test_size
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize( test_loss / test_size

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

    edge_map = train_loader.dataset.dataset.edge_map

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
            fp = fp.to(device)
            graph = send_graph_to_device(graph, device)

            # Do the computations for the forward pass
            fp_pred = model(graph)

            # Compute the loss with proper summation, solves the problem ?
            loss = model.compute_loss(fp, fp_pred)

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

            del loss

        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        # Log training metrics
        train_loss = running_loss / num_batches
        writer.add_scalar("Training epoch loss", train_loss, epoch)

        # train_accuracy = running_corrects / num_batches
        # writer.log_scalar("Train accuracy during training", train_accuracy, epoch)

        # Test phase
        test_loss = test(model, test_loader, device)
        print(">> test loss ", test_loss)

        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        writer.add_scalar("Test loss during training", test_loss, epoch)

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

