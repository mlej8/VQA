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

    if isinstance(layer, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias.data)
