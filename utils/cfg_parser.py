
import torch
import pandas as pd
import torch.nn as nn
from collections import OrderedDict
import configparser


def padding_type(h, pad, params):

    if isinstance(pad, list):
        return torch.tensor(pad)

    elif pad == 'same':

        k = torch.tensor(params['kernel'])
        s = torch.tensor(params['stride'])

        return (h*(s-1)-1+k)//2

    elif pad == 'valid':
        return torch.zeros(h.shape).long()
    else:
        raise ValueError('Pad type is invalid')


def safe_conversion(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


class CFGParser(object):

    def __init__(self, cfg_fname):

        self.config = configparser.ConfigParser(strict=False,
            allow_no_value=True)

        self.config.read(cfg_fname)
        self.curr_shape = None

        self.modules = {
                    'conv2d' : nn.Conv2d,
                    'maxpool2d' : nn.MaxPool2d,
                    'relu' : nn.ReLU,
                    'batchnorm1d':nn.BatchNorm1d,
                    'batchnorm2d':nn.BatchNorm2d,
                    'dropout':nn.Dropout,
                    'elu':nn.ELU,
                    'lstm':nn.LSTM,
                    'gru':nn.GRU,
                    'linear':nn.Linear
        }
        self.seq_ind = 2
        self.on_convs = True
        self.on_recur = False

    def _adjust_dim(self, vals, shape):
        ret = []
        for v in vals:
            if isinstance(v, int):
                ret.append(v*torch.ones(shape).long())
            else:
                ret.append(v)
        return ret

    def _is_dense(self, name):
        return name in ['linear', 'lstm', 'gru']

    def _spatial_shape(self, p, k, s):

        spatial = self.curr_shape[1:]
        if k is None:
            return spatial
        p, k, s = self._adjust_dim([p,k,s], spatial.shape)
        return (spatial + 2*p - k)//s + 1


    def _to_dict(self, conf, name):
        dic = {k:safe_conversion(v) for k,v in conf.items()}
        if self._is_dense(name):
            hidden_size = dic.get('hidden_size', None)
            new_shape = dic.get('out_features', None) if hidden_size is None else hidden_size
            new_shape = torch.tensor([new_shape])
        else:
            channel = self.curr_shape[0]
            padding = dic.pop('padding', 0)
            kernel = dic.get('kernel_size', None)
            stride = dic.get('stride',None)

            if padding:
                padding = padding_type(self.curr_shape[1:], padding, dic)
                dic.update({'padding':tuple(padding.numpy())})
            
            out_channels = dic.get('out_channels', channel.item())
            out_channels = torch.tensor([out_channels])
            params = [padding, kernel, stride]
            new_spatial = self._spatial_shape(*params)

            new_shape = torch.cat([out_channels, new_spatial])

        return dic, new_shape

    def _build_layer(self, name, dic):

        assert name in self.modules, "Not yet defined layer type."

        if name.startswith('conv'):
            dic.update({'in_channels':self.curr_shape[0]})

        if name.startswith('batchnorm'):
            # For some reason batchnorm doesn't like 0-d longtensor ??
            dic.update({'num_features':self.curr_shape[0].item()})

        if name.startswith(('lstm', 'gru', 'linear')):
            self.on_convs = False

            non_seq = [0,1,2]
            del non_seq[self.seq_ind]

            size = self.curr_shape[non_seq[0]]
            if self.curr_shape.size(0) > 2:
                size *= self.curr_shape[non_seq[1]]

            if name.startswith('linear'):
                dic.update({'in_features':size})
            else:
                self.on_recur = True
                dic.update({'input_size':size})
                dic.update({'batch_first' : True})

        return self.modules[name](**dic).cuda()


    def _flow(self, in_shape, build_model):

        self.curr_shape = torch.tensor(in_shape) 

        convs = OrderedDict()
        recur = None
        dense = OrderedDict()
        
        for l in self.config.sections():
            
            name = l.split('_')[0]
            dic, new_shape = self._to_dict(self.config[l], name)

            layer = self._build_layer(name, dic)
            if self.on_convs:
                convs[l] = layer
            elif self.on_recur:
                recur = layer
                self.on_recur = False
            else:
                dense[l] = layer

            self.curr_shape = new_shape
        
        return convs, recur, dense


    def get_modules(self, in_shape):

        convs, recur, dense = self._flow(in_shape, build_model=True)
        out = nn.Sequential()

        if convs:
            out.convs = nn.Sequential(convs)
        if recur:
            out.recur = recur
        if dense:
            out.dense = nn.Sequential(dense)
        return out




if __name__ == '__main__':
   pass

