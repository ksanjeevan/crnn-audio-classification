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
    def __init__(self, classes, config={}, state_dict=None):
        super(AudioCRNN, self).__init__(config)
        
        in_chan = 2 if config['transforms']['args']['channels'] == 'stereo' else 1

        self.classes = classes
        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = Melspectrogram(hop=None, 
                                n_mels=128, 
                                n_fft=2048, 
                                norm='whiten', 
                                stretch_param=[0.5, 0.3])
        
        
        # shape -> (batch, channel, time, freq)
        # = (batch, 2, max_hop, spec.n_mel)

        self.cfg = CFGParser(config['cfg'])
        self.net = self.cfg.get_modules([in_chan, self.spec.n_mels, 400])

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        
        for name, layer in self.net.convs.named_children():
            if name.startswith(('conv2d','maxpool2d')):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride]) 
                lengths = (lengths + 2*p - k)//s + 1
        return lengths


    def forward(self, batch):    
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs

        # x-> (batch, channel, time)
        xt = x.float().transpose(1,2)

        # xt -> (batch, channel, freq, time)
        xt, lengths = self.spec(xt, lengths)                

        # (batch, channel, freq, time)
        xt = self.net.convs(xt)

        lengths = self.modify_lengths(lengths)
        lengths = [l if l > 0 else 1 for l in lengths]

        # xt -> (batch, time, freq, channel)
        x = xt.transpose(1, -1)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)

        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net.recur(x_pack)

        x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        
        # (batch, classes)
        x = self.net.dense(x)

        x = F.log_softmax(x, dim=1)


        return x

    def predict(self, x):
        out_raw = self.forward( x )
        out = torch.exp(out_raw)
        max_ind = out.argmax().item()        
        return self.classes[max_ind], out[:,max_ind].item()

