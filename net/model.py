import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

# F.max_pool2d needs kernel_size and stride. If only one argument is passed, 
# then kernel_size = stride

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

        self.cfg = CFGParser(config['cfg'])
        net = self.cfg.get_modules([in_chan, self.spec.n_mels, 400])

        self.convs = net.convs
        self.recur = net.recur
        self.dense = net.dense
     

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        
        for name, layer in self.convs.named_children():
            #if name.startswith(('conv2d','maxpool2d')):
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride]) 
                lengths = (lengths + 2*p - k)//s + 1

        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))

    def forward(self, batch):    
        # x-> (batch, time, channel)
        x, lengths, _ = batch # unpacking seqs, lengths and srs

        # x-> (batch, channel, time)
        xt = x.float().transpose(1,2)

        # xt -> (batch, channel, freq, time)
        xt, lengths = self.spec(xt, lengths)                

        # (batch, channel, freq, time)
        xt = self.convs(xt)
        lengths = self.modify_lengths(lengths)

        # xt -> (batch, time, freq, channel)
        x = xt.transpose(1, -1)


        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.recur(x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
        
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.dense(x)

        x = F.log_softmax(x, dim=1)

        return x

    def predict(self, x):
        out_raw = self.forward( x )
        out = torch.exp(out_raw)
        max_ind = out.argmax().item()        
        return self.classes[max_ind], out[:,max_ind].item()

