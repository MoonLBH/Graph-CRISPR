import torch

def get_activation_function(name, alpha=None):
    if name == 'ReLU':
        return torch.nn.ReLU()
    elif name == 'LeakyReLU':
        if alpha is not None:
            return torch.nn.LeakyReLU(negative_slope=alpha)
        else:
            return torch.nn.LeakyReLU()
    elif name == 'ELU':
        return torch.nn.ELU()
    elif name == 'GELU':
        return torch.nn.GELU()
    elif name == 'Tanh':
        return torch.nn.Tanh()
    elif name == 'Sigmoid':
        return torch.nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation function: {name}")

def get_conv_layer(name, in_channels, out_channels):
    if name == 'GCNConv':
        from torch_geometric.nn import GCNConv
        return GCNConv(in_channels, out_channels)
    elif name == 'GATConv':
        from torch_geometric.nn import GATConv
        return GATConv(in_channels, out_channels)
    elif name == 'SAGEConv':
        from torch_geometric.nn import SAGEConv
        return SAGEConv(in_channels, out_channels)
    elif name == 'GraphConv':
        from torch_geometric.nn import GraphConv
        return GraphConv(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown convolution layer: {name}")

def get_pool_layer(name, in_channels):
    if name == 'TopKPooling':
        from torch_geometric.nn import TopKPooling
        return TopKPooling(in_channels, ratio=0.8)
    elif name == 'SAGPooling':
        from torch_geometric.nn import SAGPooling
        return SAGPooling(in_channels, ratio=0.8)
    # elif name == 'EdgePooling':
    #     from torch_geometric.nn import EdgePooling
    #     return EdgePooling(in_channels)
    else:
        raise ValueError(f"Unknown pooling layer: {name}")

def get_global_pool_layer(name):
    if name == 'global_add_pool':
        from torch_geometric.nn import global_add_pool
        return global_add_pool
    elif name == 'global_mean_pool':
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool
    elif name == 'global_max_pool':
        from torch_geometric.nn import global_max_pool
        return global_max_pool
    else:
        raise ValueError(f"Unknown global pooling layer: {name}")
