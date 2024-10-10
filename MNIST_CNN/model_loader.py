import torch


def load_checkpoint(model_path, model, optimizer) -> torch.nn.Module:
    """
    Load the model and optimizer from the checkpoint file.
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    return model, optimizer, epoch, loss