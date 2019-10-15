import os
import io
import numpy as np


def mkdirs(name, permissive=True):
    """
    Try to make the logs folder
    :param name:
    :param permissive: If True will overwrite existing files (good for debugging)
    :return:
    """
    log_path = os.path.join('results', name)
    save_path = os.path.join('trained_models', name)
    try:
        os.makedirs(log_path)
        os.makedirs(save_path)
    except FileExistsError:
        if not permissive:
            raise ValueError('This name is already taken !')
    save_name = os.path.join(save_path, name + '.pth')
    return log_path, save_name


def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape), o.size())
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))


if __name__ == '__main__':
    pass

    # for key, value in labels.items():
    #     tensor = torch.from_numpy(value)
    #     labels[key] = tensor
    #     tensor.requires_grad = False
