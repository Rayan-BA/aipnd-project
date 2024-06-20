import argparse
import json
import torch
from torchvision import models
import numpy as np
from PIL import Image

def load_checkpoint(args):
    '''Loads a saved checkpoint from .pth file, returns a PyTorch model'''
    chkpnt = torch.load(args.checkpoint, map_location=torch.device('cuda') if args.gpu else torch.device('cpu'))
    
    if chkpnt['arch'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = chkpnt['classifier_arch']
    elif chkpnt['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = chkpnt['classifier_arch']
        
    model.load_state_dict(chkpnt['model_state_dict'])
    model.class_to_idx = chkpnt['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array'''
    img = Image.open(image)
    img = img.resize((256, 256))
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    np_img = np.array(img)
    np_img = np_img / 255.0
    np_img = (np_img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_img = np_img.transpose((2,0,1))
    return np_img

def predict(model, processed_image, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    with torch.no_grad():
        ps = torch.exp(model(torch.tensor(processed_image).type(torch.FloatTensor).unsqueeze(0)))
    return ps.topk(top_k, dim=1)

def parse_args():
    '''Parses command line arguments, returns parsed args'''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', help='Path of image to be predicted.', type=str)
    parser.add_argument('checkpoint', help='Path of model checkpoint.', type=str)
    parser.add_argument('--top_k', default=3, help='Number of likely classes predicted.', type=int)
    parser.add_argument('--category_names', help='Path of mapping of categories to real names.', type=str)
    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    model = load_checkpoint(args)
    
    processed_image = process_image(args.image_path)
    
    probs, classes = predict(model, processed_image, args.top_k)
    classes, probs = classes[0].numpy(), probs[0].numpy()
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(idx)].title() for idx in classes]
    
    print('(Class, Probability):\n')
    for cls, prb in zip(classes, probs):
        print('({}, {:.2f})'.format(cls, prb))

if __name__ == '__main__': main()