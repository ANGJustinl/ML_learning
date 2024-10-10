import torch
import logger
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cnn import CNN
from model_test import check_accuracy
from model_loader import load_checkpoint


checkpoint_path = None #"mnist_cnn.checkpoint"

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.logger.info(f"Using {device}")

input_size = 28 * 28  # 784 pixels in one image (28x28)
num_classes = 10  # 10 digits (0-9)
learning_rate = 0.001  # Learning rate for the optimizer
batch_size = 64  # Number of samples to be considered in each iteration
num_epochs = 10  # Number of times the model will be trained on the entire dataset

# Load the MNIST dataset
train_dataset = datasets.MNIST(root="MNIST_CNN/data/", download=True, train=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="MNIST_CNN/data/", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Define the model for MNIST dataset
model = CNN(in_channels=1, num_classes=num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the Tensorboard writer
writer = SummaryWriter('MNIST_CNN/logs/')

if checkpoint_path is None:
    # Train the model
    logger.logger.info("No model checkpoint provided, Training the model")
    for epoch in range(num_epochs):
        logger.logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            # Move data to GPU if available
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute predicted outputs by passing inputs to the model
            scores = model(data)
            loss = criterion(scores, targets)

            running_loss = loss.item()
            writer.add_scalar("loss", running_loss, epoch * len(train_loader) + batch_index)

            # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.zero_grad()
            loss.backward()

            # Update the model parameters
            optimizer.step()

    writer.close()
    logger.logger.info("Training completed")
    
    # save the model checkpoint
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, "MNIST_CNN/mnist_cnn.checkpoint")
    logger.logger.info("Model saved successfully")
else:
    # Load the model checkpoint
    model, optimizer, start_epoch, loss = load_checkpoint(checkpoint_path, model, optimizer)
    logger.logger.info(f"Model loaded successfully from epoch {start_epoch} with loss {loss}")

check_accuracy(train_loader, model, device)
check_accuracy(test_loader, model, device)
