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

import pandas as pd

#TODO: Begin of your code
def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)
    """
    #OG
    answer = model(model_input, device)
    return answer
    #"""
    """
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

                #tutorial thought, has error
                #log_likelihood = torch.log(model_out) + discretized_mix_logistic_loss(sample, model_out)
                #gpt recommend
                #print(sample)      #sample is within range
                #print(model_out)    #model_out is not within range
                log_likelihood = -discretized_mix_logistic_loss(sample, model_out)      # for classification this is good enough (proportion), for actual prob need to follow tutorial slide
                print(f"loglikelihood for label {i}:", log_likelihood.item(), "\n")
                if log_likelihood > max_p:
                    max_p = log_likelihood
                    corresponding_label = i
                    
            answer[b] = corresponding_label
    #"""
    
    #"""
    batch_size = model_input.size(0)
    log_likelihoods = torch.zeros(batch_size, NUM_CLASSES, device=device)
        
    with torch.no_grad():
        for class_idx in range(NUM_CLASSES):
            label = torch.full((batch_size,), class_idx, dtype=torch.long, device=device)
                
            output = model(model_input, label, sample=False)
            #print(output)
                
            for i in range(batch_size):
                single_input = model_input[i:i+1]
                single_output = output[i:i+1]
                nll = discretized_mix_logistic_loss(single_input, single_output)
                log_likelihoods[i, class_idx] = -nll / np.prod(single_input.shape[1:])
        
    _, answer = torch.max(log_likelihoods, dim=1)
    #print(answer)
    #"""
    
    return answer

    
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    #acc_tracker = ratio_tracker()
    # Load CSV into DataFrame
    csv_file = "test_submission.csv"
    
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(csv_file, header=None)
    df.columns = ['filename', 'prediction']

    predictions = []
    
    # Loop over the data loader batches.
    for item in tqdm(data_loader, desc="Processing batches"):
        model_input, categories = item
        model_input = model_input.to(device)
        
        # Get predictions for the current batch.
        answer = get_label(model, model_input, device)
        
        print(answer)
        # Check if answer contains more than one prediction.
        # If it's a tensor (or a list) representing multiple values,
        # extend the list with the individual predictions.
        if hasattr(answer, "tolist"):
            # Convert to a list (e.g., for a PyTorch tensor).
            answer_list = answer.tolist()
            predictions.extend(answer_list)
        elif isinstance(answer, (list, tuple)):
            predictions.extend(answer)
        else:
            predictions.append(answer)

    # Verify that the predictions count matches the CSV rows.
    if len(predictions) != len(df):
        raise ValueError("The number of predictions ({}) does not match the number of rows in the CSV ({})."
                         .format(len(predictions), len(df)))
    
    # Replace the 'prediction' column with the new predictions.
    df['prediction'] = predictions

    # Save the updated DataFrame back to the CSV file.
    df.to_csv(csv_file, index=False, header=False)
    
    return 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    #"""
    #added for convenient
    parser.add_argument('-l', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    
    parser.add_argument('-q', '--nr_resnet', type=int, default=1,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=40,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-r', '--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    #"""
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}
    
    device = "mps"              #add for mps

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    """
    #OG code
    model = random_classifier(NUM_CLASSES)
    #"""
    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=3, nr_logistic_mix=args.nr_logistic_mix)
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    
    #OG
    #model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    #NEW, For simplicity for now
    model_path = args.load_params
    
    if os.path.exists(model_path):
        #OG
        #model.load_state_dict(torch.load(model_path))
        #new
        model.load_state_dict(torch.load(model_path, map_location=device))

        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        