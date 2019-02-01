from distances import square_distance_loss, square_distance_loss_soft, square_distance_loss_distr
import torch.nn.functional as F

# This is used to read legacy models.
OLD_NET_IDS = {
    # Relu and sigmoid.
    'std':        ('net_relu', 'ReLU'),
    'StdNet':     ('net_relu', 'ReLU'),
    'ReLU':       ('net_relu', 'ReLU'),

    # MWD
    'rbfi':       ('net_mwd', 'MWDNet'),
    'RBFINet':    ('net_mwd', 'MWDNet'),
    'RBFNet':     ('net_mwd', 'MWDNet'),
}

LOSS = {
    'nll': F.nll_loss,
    'sdl': square_distance_loss,
    'sdl_soft': square_distance_loss_soft,
    'sdl_distr': square_distance_loss_distr,
}
