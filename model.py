import torch.nn as nn
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None
        
        
        # FiLM initialize
        embedding_dim = 16
        self.embedding = nn.Embedding(4, embedding_dim)     # 4 classes, 16 based on gpt recommendation
        self.film_gen = FiLMGenerator(embedding_dim, n_channels=self.nr_filters)     #set 16 for now
        

    #def forward(self, x, sample=False):    #OG
    def forward(self, x, class_labels, sample=False):
        """
        #added early fusion
        if sample == False or sample != False:
            class_embedding = self.embedding(class_labels)
            B, C, H, W = x.shape
            class_embedding = class_embedding.view(B, 3, 1, 1)      # reshape embedding for broadcasting
            #print(class_labels.shape)
            #print(class_embedding.shape)    #[16, 3]
            x = x + class_embedding                                  # Add to feature maps
        #"""
        
        # similar as done in the tf repo :
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding
            self.init_padding = self.init_padding.to(x.device)  #added for mps

        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            padding = padding.to(x.device)  #added for mps
            x = torch.cat((x, padding), 1)
            
        #print(f"input x norm: {x.norm().item()}")

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()               #stored features? why
        ul = ul_list.pop()
        
        #print(f"pre FiLM u norm: {u.norm().item()}")
        #print(f"pre FiLM ul x norm: {ul.norm().item()}")
        
        #"""
        # FiLM layer
        # In forward():
        class_embedding = self.embedding(class_labels)  # (B, embedding_dim)

        # Suppose after the up pass you have a feature map `u` shaped (B, self.nr_filters, H, W).
        gamma, beta = self.film_gen(class_embedding)  # each (B, nr_filters)
        
        #print(f"gamma norm: {gamma.norm().item()}")
        #print(f"beta norm: {beta.norm().item()}")
        u = apply_film(u, gamma, beta)
        ul = apply_film(ul, gamma, beta)
        #"""
        
        #print(f"post u norm: {u.norm().item()}")
        #print(f"post ul x norm: {ul.norm().item()}")
        #print("post FiLM u min:", u.min().item(), "ul max:", u.max().item())
        #print("post FiLM ul min:", ul.min().item(), "ul max:", ul.max().item())

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        #print(ul)
        #print("before elu: ul min:", ul.min().item(), "ul max:", ul.max().item())
        #print("after elu norm", F.elu(ul).norm().item())       #ISSUE: ul value is too extreme that no matter whether apply FiLM or not, or apply different input, after elu, full of -1
        x_out = self.nin_out(F.elu(ul))
        #print("x_out min:", ul.min().item(), " max:", ul.max().item())

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
    
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if os.path.join(os.path.dirname(__file__), 'models') not in os.listdir():
            os.mkdir(os.path.join(os.path.dirname(__file__), 'models'))
        torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth'))
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
    

# FiLM layer

class FiLMGenerator(nn.Module):
    def __init__(self, embedding_dim, n_channels):
        super().__init__()
        # MLP that outputs gamma and beta for each channel
        self.film_fc = nn.Linear(embedding_dim, 2 * n_channels)

    def forward(self, embedding):
        # embedding is (B, embedding_dim)
        gamma_beta = self.film_fc(embedding)  # shape (B, 2*C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  # each is (B, C)
        return gamma, beta

def apply_film(feature_map, gamma, beta):
    # feature_map: (B, C, H, W)
    B, C, H, W = feature_map.shape

    # Reshape gamma, beta for broadcast: (B, C, 1, 1)
    gamma = gamma.view(B, C, 1, 1)
    beta = beta.view(B, C, 1, 1)

    # scale and shift
    return gamma * feature_map + beta
