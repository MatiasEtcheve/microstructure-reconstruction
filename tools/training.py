import torch
import wandb
from tqdm import tqdm


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for data, target in tqdm(train_loader, total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_loader)


def compute_outputs(model, device, val_loader):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if idx == 0:
                outputs = output
                targets = target
            else:
                outputs = torch.cat((outputs, output), dim=0)
                targets = torch.cat((targets, target), dim=0)
    return outputs, targets


def compute_errors(outputs, targets, device, min_val, max_val):
    min_val = torch.FloatTensor(min_val).to(device)
    max_val = torch.FloatTensor(max_val).to(device)
    outputs = outputs.to(device)
    targets = targets.to(device)
    return (outputs - targets) / (targets + (min_val / (max_val - min_val)))


def validate(model, device, val_loader, criterion, min, max):
    model.eval()
    val_loss = 0
    example_images = []
    min = torch.FloatTensor(min).to(device)
    max = torch.FloatTensor(max).to(device)
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            example_images.append(wandb.Image(data[0]))
            error = (output - target) / (target + (min / (max - min)))
            if idx == 0:
                errors = error
            else:
                errors = torch.cat((errors, error), dim=0)
    return val_loss / len(val_loader), example_images, errors.max()
