import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

# F.max_pool2d needs kernel_size and stride. If only one argument is passed, 
# then kernel_size = stride

import requests
import torchvision.models as models
class VGG16(BaseModel):
    def __init__(self, classes=None, config=None, state_dict=None):

        super(VGG16, self).__init__(mode)
        self.classes = classes
        self.model = models.vgg16(pretrained=True)

        tl_mode = config['net_mode']

        if tl_mode == 'init':
            self._modify_last_layer(len(self.classes))
        elif tl_mode == 'freeze':
            self._tl_freeze()
            self._modify_last_layer(len(self.classes))
        elif tl_mode == 'random':
            self._tl_random()
            self._modify_last_layer(len(self.classes))
        elif state_dict is not None:
            out_size, _ = self._get_shape_from_sd(state_dict)
            self._modify_last_layer(out_size)
        else:
            self.classes = self.__get_labels()


    def __get_labels(self):
        LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
        index_to_class = {int(key):value for (key, value)
           in requests.get(LABELS_URL).json().items()}
        return [index_to_class[k] for k in sorted(list(index_to_class.keys()))]


    def _modify_last_layer(self, num_classes):
        in_dim = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_dim, num_classes)
    

    def _tl_freeze(self):

        for param in self.model.parameters():
            param.requires_grad = False

    def _tl_random(self):
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def _get_shape_from_sd(self, dic):
        '''
        Get last layer shape when loading from a state dict.
        '''
        return dic['model.classifier.6.weight'].shape
    
    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        out_raw = self.forward( x.unsqueeze(0) )
        out = F.softmax(out_raw, dim=1)
        max_ind = out.argmax().item()

        return self.classes[max_ind], out[:,max_ind].item()


from .audio import Melspectrogram

from utils import plot_heatmap, CFGParser


# Architecture inspiration from: https://github.com/keunwoochoi/music-auto_tagging-keras
class AudioCRNN(BaseModel):
    # add some require grad false in some palces no?
    def __init__(self, classes, config={}, state_dict=None):
        super(AudioCRNN, self).__init__(config)
        
        in_chan = 2 if config['transforms']['args']['channels'] == 'stereo' else 1

        self.classes = classes
        self.lstm_units = 64 #200 # 400
        self.lstm_layers = 2
        self.spec = Melspectrogram(hop=None, 
                                n_mels=128, 
                                n_fft=2048, 
                                norm='whiten', 
                                stretch_param=[0.5, 0.3])
        
        #self.bn0 = nn.BatchNorm2d(in_chan)

        # shape -> (batch, channel, time, freq)
        # = (batch, 2, max_hop, spec.n_mel)

        self.cfg = CFGParser(config['cfg'])

        self.convs = self.cfg.get_modules([in_chan, 500, self.spec.n_mels])
        freq_out = self.cfg.get_spatial_shapes([self.spec.n_mels])[0]

        lstm_inp = int(self.convs.conv2d_3.out_channels*freq_out)

        self.lstm = nn.LSTM(lstm_inp, self.lstm_units, self.lstm_layers, batch_first=True)
    
        self.bn1 = nn.BatchNorm1d(self.lstm_units)
        #self.drop1 = nn.Dropout(0.3)
        self.lin1 = nn.Linear(self.lstm_units, len(classes))

        #self.lin1 = nn.Linear(self.lstm_units, self.lin1_out)
        #self.bn2 = nn.BatchNorm1d(self.lin1_out)
        #self.drop2 = nn.Dropout(0.3)
        #self.lin2 = nn.Linear(self.lin1_out, len(classes))
        

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]


    def forward(self, batch):    
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs

        # x-> (batch, channel, time)
        xt = x.float().transpose(1,2)

        # xt -> (batch, channel, freq, time)
        xt, lengths = self.spec(xt, lengths)                
        #print(xt.mean(), xt.std())

        #print(xt.shape, lengths)
        '''
        self.spec.prob = 1
        xt2, lengths2 = self.spec(xt, lengths)
        self.spec.prob = 0
        xt3, lengths3 = self.spec(xt, lengths)
        for i in range(16):
            plot_heatmap(xt2[i][...,:lengths2[i]].cpu().numpy(), 'plots/plots_pv/%s.png'%i)

        for i in range(16):
            plot_heatmap(xt3[i][...,:lengths3[i]].cpu().numpy(), 'plots/plots_pv/%s_pv.png'%i)
        
        exit()
        print(xt.shape, lengths)
        '''

        # (batch, channel, freq, time)
        #xt = self.bn0(xt)
        xt = self.convs(xt)


        lengths = [self.cfg.get_spatial_shapes([l])[0] for l in lengths]
        lengths = [l if l > 0 else 1 for l in lengths]

        #print(lengths)
        #print('--------')
        # xt -> (batch, time, freq, channel)
        x = xt.transpose(1, -1)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)

        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.lstm(x_pack)
        x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        
        x = self.bn1(x)
        #x = self.drop1(x)
        x = self.lin1(x)

        x = F.log_softmax(x, dim=1)

        return x

    def predict(self, x):
        out_raw = self.forward( x )
        out = torch.exp(out_raw)
        max_ind = out.argmax().item()        
        return self.classes[max_ind], out[:,max_ind].item()




class AudioRNN(BaseModel):
    # add some require grad false in some palces no?
    def __init__(self, classes, config={},state_dict=None):
        super(AudioRNN, self).__init__()

        self.classes = classes
        self.lstm_units = 400 # 500
        self.lstm_layers = 2 # 2
        self.lin1_out = 200 # 200
        self.spec = Melspectrogram()

        lstm_inp = self.spec.n_mels * 2
        #self.bn0 = nn.BatchNorm1d(lstm_inp)
        self.lstm = nn.LSTM(lstm_inp, self.lstm_units, self.lstm_layers, batch_first=True)

        self.lin1 = nn.Linear(self.lstm_units, self.lin1_out)
        self.drop1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(self.lin1_out)

        self.lin2 = nn.Linear(self.lin1_out, len(classes))
        self.drop2 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(len(classes))
        #self.lin2 = nn.Linear(self.lstm_units, len(classes))


    def _init_hidden(self, batch_size):
        # Remember to us e nn.Parameter here to
        # not run into cuda() problems.
        ha = torch.randn(self.lstm_layers, batch_size, self.lstm_units)
        hb = torch.randn(self.lstm_layers, batch_size, self.lstm_units)

        # So this isn't working cause the params are not on cuda.
        # look into https://discuss.pytorch.org/t/correct-way-to-declare-hidden-and-cell-states-of-lstm/15745
        # for more details
        return (nn.Parameter(ha), nn.Parameter(hb))


    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]


    def forward(self, batch):
    
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs

        # x-> (batch, channel, time)
        xt = x.float().transpose(1,2)

        # xt -> (batch, channel, time, freq)
        xt, lengths = self.spec(xt, lengths)
        #xt = amplitude_to_db(xt)
        
        '''
        for i in range(16):
            plot_heatmap(xt[i][:,:lengths[i]].cpu().numpy(), 'plots/%s_amp.png'%i)

        for i in range(16):
            plot_heatmap(xt2[i][:,:lengths[i]].cpu().numpy(), 'plots/%s_db.png'%i)

        print(xt.shape)
        exit()
        '''

        # xt -> (batch, time, channel, freq)
        x = xt.transpose(1,2)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.lstm(x_pack)
        x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.bn1(x)

        x = self.lin2(x)
        x = self.drop2(x)
        x = self.bn2(x)

        x = F.log_softmax(x, dim=1)

        return x



