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
def test(model, test_loader, device, reconstruction_lam, motif_lam, ortho_lam):
    """
    Compute accuracy and loss of model over given dataset
    :param model:
    :param test_loader:
    :param test_loss_fn:
    :param device:
    :return:
    """
    model.eval()
    test_loss, ortho_loss_tot, motif_loss_tot, recons_loss_tot = (0,) * 4
    test_size = len(test_loader)
    for batch_idx, (graph, K) in enumerate(test_loader):
        # Get data on the devices
        K = K.to(device)
        K = torch.ones(K.shape).to(device) - K
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        out, attributions = model(graph)
        loss, reconstruction_loss, motif_loss, ortho_loss = compute_loss(model=model, attributions=attributions,
                                                                         out=out, K=K, device=device,
                                                                         reconstruction_lam=reconstruction_lam,
                                                                         motif_lam=motif_lam, ortho_lam=ortho_lam)

        ortho_loss_tot += ortho_loss
        recons_loss_tot += reconstruction_loss
        motif_loss_tot += motif_loss
        test_loss += loss.item()

    return test_loss / test_size, ortho_loss_tot / test_size, motif_loss_tot / test_size, recons_loss_tot / test_size


def motif_embedding(model, attributions, out, device, normalize=False):
    """
        Compute normalized E_prime = \Sigma^T Z
    """

    E_prime = torch.mm(attributions.t(), out)
    E_prime = E_prime.to(device)
    if normalize:
        attr_norm = torch.norm(attributions, dim=0)
        attr_norm = attr_norm.view(1, attr_norm.shape[0])
        attr_norm = attr_norm.to(device)

        # emb = model.embeddings.to(device)
        # motif_loss = criterion(torch.mm(attributions.t(), out), model.embeddings)
        E_prime = torch.addcdiv(torch.zeros(E_prime.shape).to(device), E_prime, attr_norm.t())

    return E_prime


def compute_loss(model, attributions, out, K,
        device, reconstruction_lam, motif_lam, ortho_lam):
    """
    Compute the total loss and returns scalar value for each contribution of each term. Avoid overwriting loss terms
    :param model:
    :param attributions:
    :param out:
    :param K:
    :param device:
    :param reconstruction_lam:
    :param motif_lam:
    :param ortho_lam:
    :return:
    """

    # reconstruction loss
    K_predict = torch.norm(out[:, None] - out, dim=2, p=2)
    reconstruction_loss = torch.nn.MSELoss()
    reconstruction_loss = reconstruction_loss(K_predict, K)

    """
    IDEA:
        - get graph-level embedding for each graph a = mean(Z, axis=0)
        - get graph-level motif assignments for each graph, m = mean(\Sigma, axis=0)
        - maybe add attention term to means
        - train || DM(a) - DM(m) || -> 0
    """

    if attributions is not None:
        E_prime = motif_embedding(model, attributions, out, device, normalize=model.motif_norm)

        motif_loss = torch.nn.MSELoss()
        if not model.motif_norm:
            ref = torch.ones((E_prime.shape[0]))
            motif_loss = motif_loss(F.cosine_similarity(E_prime,model.embeddings), ref)
        else:
            ref = torch.zeros_like(model.embeddings)
            motif_loss = motif_loss(E_prime - model.embeddings, ref)

        # push dictionary to be orthogonal
        ortho_loss = torch.nn.MSELoss()
        ortho_loss = ortho_loss(torch.mm(model.embeddings, model.embeddings.t()), torch.eye(model.num_modules))

        # total loss
        loss = reconstruction_lam * reconstruction_loss + motif_lam * motif_loss + ortho_lam * ortho_loss
        return loss, reconstruction_loss.item(), motif_loss.item(), ortho_loss.item(),
    return reconstruction_loss, reconstruction_loss.item(), 0, 0


def train_model(model, criterion, optimizer, device, train_loader, test_loader, save_path,
                writer=None, num_epochs=25, wall_time=None, ortho_lam=1,
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
    :param ortho_lam: how much to enforce orthogonality between motifs
    :param reconstruction_lam: how much to enforce pariwise similarity conservation
    :param motif_lam: how much to enforce motif assignment
    :param embed_only: number of epochs before starting attributor training.
    :return:
    """

    epochs_from_best = 0
    early_stop_threshold = 60

    start_time = time.time()
    best_loss = sys.maxsize

    motif_lam_orig = motif_lam
    reconstruction_lam_orig = reconstruction_lam
    ortho_lam_orig = ortho_lam

    #if we delay attributor, start with attributor OFF
    #if <= -1, both always ON.
    if embed_only > -1:
        print("Switching attriutor OFF. Embeddings still ON.")
        set_gradients(model, attributor=False)
        motif_lam, ortho_lam = (0,0)

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
            motif_lam, ortho_lam = (motif_lam_orig, ortho_lam_orig)

        running_loss = 0.0

        time_epoch = time.perf_counter()

        num_batches = len(train_loader)

        for batch_idx, (graph, K) in enumerate(train_loader):

            # Get data on the devices
            batch_size = len(K)
            K = K.to(device)
            K = torch.ones(K.shape).to(device) - K
            graph = send_graph_to_device(graph, device)

            # Do the computations for the forward pass
            out, attributions = model(graph)

            # Compute the loss with proper summation, solves the problem ?
            loss, reconstruction_loss, motif_loss, ortho_loss = compute_loss(model=model, attributions=attributions,
                                                                             out=out, K=K, device=device,
                                                                             reconstruction_lam=reconstruction_lam,
                                                                             motif_lam=motif_lam, ortho_lam=ortho_lam)
            # Backward
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Metrics
            batch_loss = loss.item()
            running_loss += batch_loss
            del loss

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
                writer.add_scalar("Training reconstruction loss", reconstruction_loss,
                                  epoch * num_batches + batch_idx)
                writer.add_scalar("Training motif loss", motif_loss,
                                  epoch * num_batches + batch_idx)
                writer.add_scalar("Training ortho loss", ortho_loss,
                                  epoch * num_batches + batch_idx)

        # Log training metrics
        train_loss = running_loss / num_batches
        writer.add_scalar("Training epoch loss", train_loss, epoch)

        # train_accuracy = running_corrects / num_batches
        # writer.log_scalar("Train accuracy during training", train_accuracy, epoch)

        # Test phase
        test_loss, ortho_loss, motif_loss, recons_loss = test(model, test_loader, device, reconstruction_lam, motif_lam,
                                                              ortho_lam)

        writer.add_scalar("Test loss during training", test_loss, epoch)
        writer.add_scalar("Test reconstruction loss", recons_loss, epoch)
        writer.add_scalar("Test motif loss", motif_loss, epoch)
        writer.add_scalar("Test ortho loss", ortho_loss, epoch)

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
