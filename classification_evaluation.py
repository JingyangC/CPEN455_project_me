'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)
    """
    #OG
    answer = model(model_input, device)
    return answer
    #"""
    #"""
    # classification don't seem to be the issue
    with torch.no_grad():
        batch_size = model_input.size(0)
        answer = torch.zeros(batch_size, device=device)
        for b in range(batch_size):
            # Make sure we start from a very small number:
            max_p = float('-inf')       # not 0
            # -discretized_mix_logistic_loss(...) is typically negative, (the loss is positive, so its negative is usually below zero)
            corresponding_label = 0
            
            #print(f"model input min: {model_input.min()} and max: {model_input.max()}")
            sample = model_input[b].unsqueeze(0)  # Shape: (1, C, H, W)
            for i in range(NUM_CLASSES):
                class_tensor = torch.tensor([i], device=device)
                model_out =  model(x=sample, class_labels=class_tensor) #return model return x_out 
                #print(f"Label {i} -> Output norm: {model_out.norm().item()}")

                #print(sample)      #sample is within range
                #print(model_out)    #model_out is not within range
                log_likelihood = -discretized_mix_logistic_loss(sample, model_out)      # for classification this is good enough (proportion), for actual prob need to follow tutorial slide
                #print(f"loglikelihood for label {i}:", log_likelihood.item(), "\n")
                if log_likelihood > max_p:
                    max_p = log_likelihood
                    corresponding_label = i
                    
            answer[b] = corresponding_label
    #"""
    
    
    return answer

    
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':2, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    """
    #OG code
    model = random_classifier(NUM_CLASSES)
    #"""
    model = PixelCNN(nr_resnet=3, nr_filters=250, 
                input_channels=3, nr_logistic_mix=150)
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        