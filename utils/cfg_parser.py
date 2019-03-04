
import torch
import pandas as pd
import torch.nn as nn
from collections import OrderedDict
import configparser


# SHould store the configs in the checkpoints too
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
                    'elu':nn.ELU
        }


    def _adjust_dim(self, vals, shape):
        ret = []
        for v in vals:
            if isinstance(v, int):
                ret.append(v*torch.ones(shape).long())
            else:
                ret.append(v)
        return ret


    def _spatial_shape(self, p, k, s):

        spatial = self._get_to_spatial()
        
        if k is None:
            return spatial
        p, k, s = self._adjust_dim([p,k,s], spatial.shape)
        return (spatial + 2*p - k)//s + 1


    def _to_dict(self, conf):
        return {k:safe_conversion(v) for k,v in conf.items()}

    def _get_to_spatial(self):
        size = self.curr_shape.size()[0]

        if size <= 2:
            return self.curr_shape
        else: 
            return self.curr_shape[1:]


    def _build_layer(self, name, dic):
        channel = self.curr_shape[0]

        assert name in self.modules, "Not yet defined layer type."

        if name.startswith('conv'):
            dic.update({'in_channels':channel})

        if name.startswith('batchnorm'):
            # For some reason batchnorm doesn't like 0-d longtensor ??
            dic.update({'num_features':channel.item()})

        out_channels = dic.get('out_channels', channel.item())
        return self.modules[name](**dic), torch.tensor([out_channels]) 


    def _flow(self, in_shape, build_model):

        self.curr_shape = torch.tensor(in_shape) 

        layers = []
        shapes = [list(self.curr_shape.numpy())]
        
        for layer in self.config.sections():
            
            name = layer.split('_')[0]
            dic = self._to_dict(self.config[layer])
            padding = dic.pop('padding', 0)
            kernel = dic.get('kernel_size', None)
            stride = dic.get('stride',None)

            if padding:
                padding = padding_type(self._get_to_spatial(), padding, dic)
                dic.update({'padding':tuple(padding.numpy())})

            params = [padding, kernel, stride]
            new_spatial = self._spatial_shape(*params)

            if build_model:
                layer, new_channels = self._build_layer(name, dic)
                new_spatial = torch.cat([new_channels, new_spatial])
                layers.append(layer)
            else:
                
                shapes.append(list(new_spatial.numpy()))

            self.curr_shape = new_spatial

        if build_model:
            return layers
        else:
            return shapes


    def get_modules(self, in_shape):

        layers = self._flow(in_shape, build_model=True)
        return nn.Sequential(
                OrderedDict(
                    zip(self.config.sections(), layers)))


    def get_spatial_shapes(self, in_shape, last=True):
        shapes = self._flow(in_shape, build_model=False)
        if last:
            return shapes[-1]
        return shapes


if __name__ == '__main__':
   pass

