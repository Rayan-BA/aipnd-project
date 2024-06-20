import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
import argparse
from tqdm import tqdm
from pathlib import Path

def model_setup(args):
    '''Loads model from torchvision api and replaces classifier, returns PyTorch model'''
    if args.arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if args.hidden_units:
        if args.arch == 'resnet50':
            model.fc = nn.Sequential(nn.Linear(2048, args.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(args.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
        elif args.arch == 'densenet121':
            model.classifier = nn.Sequential(nn.Linear(1024, args.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(args.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    else:
        if args.arch == 'resnet50':
            model.fc = nn.Sequential(nn.Linear(2048, 102), nn.LogSoftmax(dim=1))
        elif args.arch == 'densenet121':
            model.classifier = nn.Sequential(nn.Linear(1024, 102), nn.LogSoftmax(dim=1))
    
    return model

def load_data(data_dir):
    '''Loads and transforms data for training, returns Dataset objects for training and validation'''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_transforms = transforms.Compose([transforms.Resize((255,255)),
                                       transforms.RandomResizedCrop((224, 224)), 
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(30),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize((255,255)),
                                                transforms.CenterCrop((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    return train_dataset, valid_dataset

def save_checkpoint(model, optimizer, class_to_idx, args):
    '''Saves a PyTorch model into a .pth file, no returns'''
    model.class_to_idx = class_to_idx
    checkpoint = {'arch': args.arch,
                  'classifier_arch': model.fc if args.arch == 'resnet50' else model.classifier,
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': args.epochs}
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
    else:
        torch.save(checkpoint, 'checkpoint.pth')

def train(model, trainloader, validloader, args):
    '''Trains a PyTorch model, returns model and optimizer'''
    if args.gpu: device = 'cuda'
    else: device = 'cpu'
    
    model.to(device)
    
    criterion = nn.NLLLoss()
    params = model.fc.parameters() if args.arch == 'resnet50' else model.classifier.parameters()
    optimizer = optim.Adam(params, lr=args.learning_rate)
    
    for e in range(args.epochs):
        train_loss = 0
        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        else:
            model.eval()
            valid_loss = 0
            accuracy = 0
            for images, labels in tqdm(validloader):
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                valid_loss += loss.item()
                top_p, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

            model.train()
            print('Epoch: {}/{}'.format(e+1, args.epochs)
                  + '\nTraining Loss: {:.2f}'.format(train_loss/len(trainloader))
                  + '\nValidation Loss: {:.2f}'.format(valid_loss/len(validloader))
                  +'\nAccuracy: {:.2f}'.format(accuracy/len(validloader)))
    
    return model, optimizer

def parse_args():
    '''Parses command line arguments, returns parsed args'''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', help='Training data directory.', type=str)
    parser.add_argument('--save_dir', help='Where training checkpoints will be saved.', type=str)
    parser.add_argument('--arch', choices=['resnet50', 'densenet121'], default='resnet50', help='Choose between two models, resnet50 and densenet121.', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--hidden_units', type=int)
    parser.add_argument('--epochs', default=8, help='Number of training iterations.', type=int)
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    train_dataset, valid_dataset = load_data(args.data_dir)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    
    model = model_setup(args)
    
    model, optimizer = train(model, trainloader, validloader, args)
    
    save_checkpoint(model, optimizer, train_dataset.class_to_idx, args)

if __name__ == '__main__': main()