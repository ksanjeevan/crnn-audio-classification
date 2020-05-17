'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform, HalfNormal

from torchaudio_contrib import STFT, TimeStretch, MelFilterbank, ComplexNorm, ApplyFilterbank
'''


from torchaudio.transforms import Spectrogram, MelSpectrogram , ComplexNorm

def _num_stft_bins(lengths, fft_length, hop_length, pad):
    return (lengths + 2 * pad - fft_length + hop_length) // hop_length

class MelspectrogramStretch(MelSpectrogram):

    def __init__(self, hop_length=None, 
                       sample_rate=44100, 
                       num_mels=128, 
                       fft_length=2048, 
                       norm='whiten', 
                       stretch_param=[0.4, 0.4]):

        super(MelspectrogramStretch, self).__init__(sample_rate=sample_rate, 
                                                    n_fft=fft_length, 
                                                    hop_length=hop_length, 
                                                    n_mels=num_mels)
        
        self.stft = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length, pad=self.pad, 
                                       power=None, normalized=False)

        # Augmentation
        self.prob = stretch_param[0]
        self.random_stretch = RandomTimeStretch(stretch_param[1], 
                                                self.hop_length, 
                                                self.n_fft//2+1, 
                                                fixed_rate=None)
        
        # Normalization (pot spec processing)
        self.complex_norm = ComplexNorm(power=2.)
        self.norm = SpecNormalization(norm)

    def forward(self, x, lengths=None):
        x = self.stft(x)

        if lengths is not None:
            lengths = _num_stft_bins(lengths, self.n_fft, self.hop_length, self.n_fft//2)
            lengths = lengths.long()
        
        if torch.rand(1)[0] <= self.prob and self.training:
            # Stretch spectrogram in time using Phase Vocoder
            x, rate = self.random_stretch(x)
            # Modify the rate accordingly
            lengths = (lengths.float()/rate).long()+1
        
        x = self.complex_norm(x)
        x = self.mel_scale(x)

        # Normalize melspectrogram
        x = self.norm(x)

        if lengths is not None:
            return x, lengths        
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


import numpy as np
import torch
import torch.nn as nn

from torchaudio.transforms import TimeStretch, AmplitudeToDB 
from torch.distributions import Uniform

class RandomTimeStretch(TimeStretch):

    def __init__(self, max_perc, hop_length=None, n_freq=201, fixed_rate=None):

        super(RandomTimeStretch, self).__init__(hop_length, n_freq, fixed_rate)
        self._dist = Uniform(1.-max_perc, 1+max_perc)

    def forward(self, x):
        rate = self._dist.sample().item()
        return super(RandomTimeStretch, self).forward(x, rate), rate


class SpecNormalization(nn.Module):

    def __init__(self, norm_type, top_db=80.0):

        super(SpecNormalization, self).__init__()

        if 'db' == norm_type:
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            self._norm = lambda x: self.z_transform(x)
        else:
            self._norm = lambda x: x
        
    
    def z_transform(self, x):
        # Independent mean, std per batch
        non_batch_inds = [1, 2, 3]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean)/std 
        return x

    def forward(self, x):
        return self._norm(x)


'''
def amplitude_to_db(spec, ref=1.0, amin=1e-10, top_db=80):
    """
    Amplitude spectrogram to the db scale
    """
    power = spec**2
    return power_to_db(power, ref, amin, top_db)

def power_to_db(spec, ref=1.0, amin=1e-10, top_db=80.0):
    """
    Power spectrogram to the db scale

    spec -> (*, freq, time)
    """
    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if callable(ref):
        ref_value = ref(spec)
    else:
        ref_value = torch.tensor(ref)

    log_spec = 10*torch.log10( torch.clamp(spec, min=amin) )
    log_spec -= 10*torch.log10( torch.clamp(ref_value, min=amin) )
    
    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        
        log_spec = torch.clamp(log_spec, min=(log_spec.max() - top_db))

    #log_spec /= log_spec.max()
    return log_spec
    
def spec_whiten(spec, eps=1):    
    
    along_dim = lambda f, x: f(x, dim=-1).view(-1,1,1,1)
    
    lspec = torch.log10(spec + eps)

    batch = lspec.size(0)

    mean = along_dim(torch.mean, lspec.view(batch, -1))
    std = along_dim(torch.std, lspec.view(batch, -1))

    resu = (lspec - mean)/std

    return resu


class MelspectrogramStretch(nn.Module):

    def __init__(self, hop_length=None, num_mels=128, fft_length=2048, norm='whiten', stretch_param=[0.4, 0.4]):

        super(MelspectrogramStretch, self).__init__()
        
        self.prob = stretch_param[0]
        self.dist = Uniform(-stretch_param[1], stretch_param[1])
        self.norm = {
            'whiten':spec_whiten,
            'db' : amplitude_to_db
            }.get(norm, None)

        self.stft = STFT(fft_length=fft_length, hop_length=fft_length//4)
        self.pv = TimeStretch(hop_length=self.stft.hop_length, num_freqs=fft_length//2+1)
        self.cn = ComplexNorm(power=2.)

        fb = MelFilterbank(num_mels=num_mels, max_freq=1.0).get_filterbank()
        self.app_fb = ApplyFilterbank(fb)

        self.fft_length = fft_length
        self.hop_length = self.stft.hop_length
        self.num_mels = num_mels
        self.stretch_param = stretch_param

        self.counter = 0



    def forward(self, x, lengths=None):
        x = self.stft(x)

        if lengths is not None:
            lengths = _num_stft_bins(lengths, self.fft_length, self.hop_length, self.fft_length//2)
            lengths = lengths.long()
            
        if torch.rand(1)[0] <= self.prob and self.training:
            rate = 1 - self.dist.sample()
            x = self.pv(x, rate)
            lengths = (lengths.float()/rate).long()+1

        x = self.app_fb(self.cn(x))
        
        if self.norm is not None:
            x = self.norm(x)

        if lengths is not None:
            return x, lengths
        
        return x

    def __repr__(self):
        param_str = '(num_mels={}, fft_length={}, norm={}, stretch_param={})'.format(
                        self.num_mels, self.fft_length, self.norm.__name__, self.stretch_param)
        return self.__class__.__name__ + param_str
'''