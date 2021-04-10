import torch


def weights_init(layer):
    """ Initialize weights using Xavier uniform """
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias.data)

    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias.data)

    if isinstance(layer, torch.nn.Embedding):
        torch.nn.init.xavier_uniform_(layer.weight.data)
    
    if isinstance(layer, torch.nn.LSTM):
        for name, param in layer.named_parameters():
            if param.requires_grad:
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.constant(param, 0)

    if isinstance(layer, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias.data)
