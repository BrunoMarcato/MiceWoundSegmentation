import torch
from model import UNet

# -----------------------------------------------------------------------------------------------

def save_model(model, optimizer, epoch, loss, filename = "/model"):
  print("... Saving model ...")
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, filename)

# -----------------------------------------------------------------------------------------------

def load_model(filename, learning_rate, device):
    print("... Loading model ...")
    model = UNet(in_channels = 3, out_channels = 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


    return model, optimizer, epoch, loss

# -----------------------------------------------------------------------------------------------