import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR  #added for learning rate

NUM_CLASSES = 4
TEST_ACC_EPOCH = 10
def train_or_test(model, data_loader, optimizer, loss_op, device, args, epoch, mode = 'training'):
    if mode == 'training':
        model.train()
    else:
        model.eval()
        
    deno =  args.batch_size * np.prod(args.obs) * np.log(2.)        
    loss_tracker = mean_tracker()
    
    #new
    classification_acc_tracker = ratio_tracker()
    
    for batch_idx, item in enumerate(tqdm(data_loader)):
        """
        #OG
        model_input, _ = item                  #OG
        model_input = model_input.to(device)        #accomdate the device using 
        model_output = model(model_input)      #OG
        #"""
        #"""
        # new code passing class label, conditional
        model_input, class_labels = item        #new
        model_input = model_input.to(device)        #accomdate the device using 
        #class_labels = class_labels.to(device)    

        # Convert the list of string labels to a tensor of integer indices.
        # This assumes that my_bidict is accessible here.
        # class_indices = torch.tensor([my_bidict[label] for label in class_labels]).to(device)
        class_indices = torch.tensor([my_bidict[label] for label in class_labels], dtype=torch.long).to(device)
        # Forward pass: pass both image and class_labels to the model.
        model_output = model(model_input, class_indices, sample=False)
        #"""
        
        loss = loss_op(model_input, model_output)
        loss_tracker.update(loss.item()/deno)
        if mode == 'training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            #update optimizer, must be at the end
            
        if epoch % TEST_ACC_EPOCH == 0 and mode == "val":
            model.eval()
            answer = get_label(model, model_input, device)
            correct_num = torch.sum(answer == class_indices)
            classification_acc_tracker.update(correct_num.item(), model_input.shape[0])
        
    if args.en_wandb:
        wandb.log({mode + "-Average-BPD" : loss_tracker.get_mean()})
        wandb.log({mode + "-epoch": epoch})
        if epoch % TEST_ACC_EPOCH == 0 and mode == "val":
             wandb.log({mode + "classification acc": classification_acc_tracker.get_ratio()})
             
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-w', '--en_wandb', type=bool, default=False,
                            help='Enable wandb logging')
    parser.add_argument('-t', '--tag', type=str, default='default',
                            help='Tag for this run')
    
    # sampling
    parser.add_argument('-c', '--sampling_interval', type=int, default=5,
                        help='sampling interval')
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-sd', '--sample_dir',  type=str, default='samples',
                        help='Location for saving samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='cpen455', help='Can be either cifar|mnist|cpen455')
    parser.add_argument('-st', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('--obs', type=tuple, default=(3, 32, 32),
                        help='Observation shape')
    
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=1,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=40,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=5,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-sb', '--sample_batch_size', type=int, default=32,
                        help='Batch size during sampling per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=5000, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    check_dir_and_create(args.save_dir)
    
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model_name = 'pcnn_' + args.dataset + "_"
    model_path = args.save_dir + '/'
    if args.load_params is not None:
        model_name = model_name + 'load_model'
        model_path = model_path + model_name + '/'
    else:
        model_name = model_name + 'from_scratch'
        model_path = model_path + model_name + '/'
    
    job_name = "PCNN_Training_" + "dataset:" + args.dataset + "_" + args.tag
    
    if args.en_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set entity to specify your username or team name
            # entity="qihangz-work",
            # set the wandb project where this run will be logged
            project="CPEN455HW",
            # group=Group Name
            name=job_name,
        )
        wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        wandb.config.update(args)

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Reminder: if you have patience to read code line by line, you should notice this comment. here is the reason why we set num_workers to 0:
    #In order to avoid pickling errors with the dataset on different machines, we set num_workers to 0.
    #If you are using ubuntu/linux/colab, and find that loading data is too slow, you can set num_workers to 1 or even bigger.
    kwargs = {'num_workers':3, 'pin_memory':True, 'drop_last':True}

    #newly added for mps
    #device = "mps"
    
    # set data
    if "mnist" in args.dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), rescaling, replicate_color_channel])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                            train=True, transform=ds_transforms), batch_size=args.batch_size, 
                                shuffle=True, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    elif "cifar" in args.dataset:
        ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
        if args.dataset == "cifar10":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        elif args.dataset == "cifar100":
            train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=True, 
                download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
            
            test_loader  = torch.utils.data.DataLoader(datasets.CIFAR100(args.data_dir, train=False, 
                        transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            raise Exception('{} dataset not in {cifar10, cifar100}'.format(args.dataset))
    
    elif "cpen455" in args.dataset:
        """
        # OG transform
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
        #"""
        # New data transformation
        train_transforms = transforms.Compose([
            # The order of transformations matters:
            transforms.Resize((32, 32)),             # or your desired size
            #transforms.RandomHorizontalFlip(p=0.1),  # randomly flip images
            #transforms.RandomRotation(degrees=10),   # randomly rotate images
            transforms.RandomApply([transforms.RandomCrop(32, padding=4, padding_mode='reflect')], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                saturation=0.2, hue=0.1), 
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))], p = 0.2),     #edge translation too much?
            rescaling  # whatever your custom function is for re-scaling
        ])
        test_transform = transforms.Compose([transforms.Resize((32, 32)), rescaling])
         
        train_loader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                   mode = 'train', 
                                                                   transform=train_transforms), 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    **kwargs)
        test_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                   mode = 'test', 
                                                                   transform=test_transform), 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    **kwargs)
        val_loader  = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                                   mode = 'validation', 
                                                                   transform=test_transform), 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    **kwargs)
    else:
        raise Exception('{} dataset not in {mnist, cifar, cpen455}'.format(args.dataset))
    
    args.obs = (3, 32, 32)
    input_channels = args.obs[0]
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    model = model.to(device)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params))
        print('model parameters loaded')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    """
    # OG scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    #"""
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=15)

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs-10)

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[10]
    )
    
    for epoch in tqdm(range(args.max_epochs)):
        train_or_test(model = model, 
                      data_loader = train_loader, 
                      optimizer = optimizer, 
                      loss_op = loss_op, 
                      device = device, 
                      args = args, 
                      epoch = epoch, 
                      mode = 'training')
        
        # decrease learning rate
        scheduler.step()
        """
        #I don't think we should have this in training, nor useful at the moment?
        train_or_test(model = model,
                      data_loader = test_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'test')
        """
        
        train_or_test(model = model,
                      data_loader = val_loader,
                      optimizer = optimizer,
                      loss_op = loss_op,
                      device = device,
                      args = args,
                      epoch = epoch,
                      mode = 'val')
        
        if epoch % args.sampling_interval == 0:
            print('......sampling......')
            # new, added for condition
            class_labels = torch.full((args.sample_batch_size, ), 0, dtype=torch.int64, device=device)
            sample_t = sample(model, args.sample_batch_size, args.obs, sample_op, class_labels)
            #sample_t = sample(model, args.sample_batch_size, args.obs, sample_op)   #OG
            sample_t = rescaling_inv(sample_t)
            save_images(sample_t, args.sample_dir)
            sample_result = wandb.Image(sample_t, caption="epoch {}".format(epoch))
            
            gen_data_dir = args.sample_dir
            ref_data_dir = args.data_dir +'/test'
            paths = [gen_data_dir, ref_data_dir]
            try:
                fid_score = calculate_fid_given_paths(paths, 32, device, dims=192)
                print("Dimension {:d} works! fid score: {}".format(192, fid_score))
            except:
                print("Dimension {:d} fails!".format(192))
                
            if args.en_wandb:
                wandb.log({"samples": sample_result,
                            "FID": fid_score})
        
        if (epoch) % args.save_interval == 0 and epoch != 0: 
            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
