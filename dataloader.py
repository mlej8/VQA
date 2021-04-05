from torch.utils.data import DataLoader

def preprocess(batch):
    # TODO custom batch preprocessing
    pass 

def get_data_loader(dataset, batch_size, num_workers):
    return Dataloader(
        dataset,
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=preprocess,
        num_workers=num_workers
        )