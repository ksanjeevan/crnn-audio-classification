import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Uniform, HalfNormal


def torch_angle(t):
    return torch.atan2(t[...,1], t[...,0])


def spectrogram(sig, n_fft=2048, hop=None, window=None, **kwargs):
    """
    sig -> (batch, channel, time) or (channel, time)
    stft -> (batch, channel, freq, hop, complex) or (channel, freq, hop, complex)
    """
    if hop is None:
        hop = n_fft // 4
    if window is None:
        window = torch.hann_window(n_fft)

    if sig.dim() == 3:
        batch, channel, time = sig.size()
        out_shape = [batch, channel, n_fft//2+1, -1, 2]
    elif sig.dim() == 2:
        channel, time = sig.size()
        out_shape = [channel, n_fft//2+1, -1, 2]
    else:
        raise ValueError('Input tensor dim() must be either 2 or 3.')

    sig = sig.reshape(-1, time)

    stft = torch.stft(sig, n_fft, hop, window=window, **kwargs)

    stft = stft.reshape(out_shape)
    return stft


def _hertz_to_mel(f):
    '''
    Converting frequency into mel values using HTK formula
    '''
    return 2595. * torch.log10(torch.tensor(1.) + (f / 700.))


def _mel_to_hertz(mel):
    '''
    Converting mel values into frequency using HTK formula
    '''
    return 700. * (10**(mel / 2595.) - 1.)


def create_mel_filter(n_stft, sr, n_mels=128, f_min=0.0, f_max=None):
    '''
    Creates filter matrix to transform fft frequency bins into mel frequency bins.
    Equivalent to librosa.filters.mel(sr, n_fft, htk=True, norm=None).
    '''
    # Convert to find mel lower/upper bounds
    f_max = f_max if f_max else sr // 2   
    m_min = 0. if f_min == 0 else _hertz_to_mel(f_min)
    m_max = _hertz_to_mel(f_max)

    # Compute stft frequency values
    stft_freqs = torch.linspace(f_min, f_max, n_stft)

    # Find mel values, and convert them to frequency units
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hertz(m_pts)

    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)  # (n_stft, n_mels + 2)
    
    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_stft, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_stft, n_mels)
    fb = torch.clamp(torch.min(down_slopes, up_slopes), min=0.)

    return fb, f_pts[:-2]


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


class Spectrogram(nn.Module):
    """
    Module that outputs the spectrogram
    of an audio signal with shape (batch, channel, time_hop, frequency_bins).

    Its implemented as a layer so that the computation can be faster (done dynamically
    on GPU) and no need to store the transforms. More information:
        - https://github.com/keunwoochoi/kapre
        - https://arxiv.org/pdf/1706.05781.pdf
    
    Args:
     * hop: int > 0
       - Hop length between frames in sample,  should be <= n_fft.
       - Default: None (in which case n_fft // 4 is used)
     * n_fft: int > 0 
       - Size of the fft.
       - Default: 2048
     * pad: int >= 0
       - Amount of two sided zero padding to apply.
       - Default: 0
     * window: torch.Tensor,
       -  Windowing used in the stft.
       -  Default: None (in which case torch.hann_window(n_fft) is used)
     * sr: int > 0
       -  Sampling rate of the audio signal. This may not be the same in all samples (?)
       -  Default: 44100
     * spec_kwargs: 
       -  Any named arguments to be passed to the stft

    """

    def __init__(self, hop=None, n_fft=2048, pad=0, window=None, sr=44100, stretch_param=None, **spec_kwargs):
        
        super(Spectrogram, self).__init__()

        if window is None:
            window = torch.hann_window(n_fft)

        self.window = self._build_window(window)
        self.hop = n_fft // 4 if hop is None else hop
        self.n_fft = n_fft
        self.pad = pad

        # Not all samples will have the same sr
        self.sr = sr
        self.spec_kwargs = spec_kwargs

        self.stretch_param = stretch_param
        self.prob = 0
        if self.stretch_param is not None:
            self._build_pv()

    def _build_pv(self):
        fft_size = self.n_fft//2 + 1
        self.phi_advance = nn.Parameter(torch.linspace(0, 
            math.pi * self.hop, 
            fft_size)[..., None], requires_grad=False)

        self.prob = self.stretch_param[0]
        self.dist = Uniform(-self.stretch_param[1], self.stretch_param[1])


    def _get_rate(self):
        return 1 - self.dist.sample()


    def phase_vocoder(self, D):
        # phase_vocoder
        # D -> (freq, old_time, 2)
        # D -> (batch, channel, freq, old_time, 2)
        rate = self._get_rate()
        time_steps = torch.arange(0, D.size(3), rate, device=D.device) # (new_time)
        
        alphas = (time_steps % 1)#.unsqueeze(1) # (new_time)

        phase_0 = torch_angle(D[:,:,:,:1])

        # Time Padding
        pad_shape = [0,0]+[0,2]+[0]*6
        D = F.pad(D, pad_shape)

        D0 = D[:,:,:,time_steps.long()] # (new_time, freq, 2)
        D1 = D[:,:,:,(time_steps + 1).long()] # (new_time, freq, 2)

        D0_angle = torch_angle(D0) # (new_time, freq)
        D1_angle = torch_angle(D1) # (new_time, freq)

        D0_norm = torch.norm(D0, dim=-1) # (new_time, freq)
        D1_norm = torch.norm(D1, dim=-1) # (new_time, freq)

        Dphase = D1_angle - D0_angle - self.phi_advance # (new_time, freq)
        Dphase = Dphase - 2 * math.pi * torch.round(Dphase / (2 * math.pi)) # (new_time, freq)

        # Compute Phase Accum
        phase = Dphase + self.phi_advance # (new_time, freq)

        phase = torch.cat([phase_0, phase[:,:,:,:-1]], dim=-1)
        
        phase_acc = torch.cumsum(phase, -1) # (new_time, freq)

        mag = alphas * D1_norm + (1-alphas) * D0_norm # (new_time, freq)
        
        Dstretch_real = mag * torch.cos(phase_acc) # (new_time, freq)
        Dstretch_imag = mag * torch.sin(phase_acc) # (new_time, freq)
        
        Dstretch = torch.stack([Dstretch_real, Dstretch_imag], dim=-1)

        return Dstretch, rate


    def _build_window(self, window):
        if window is None:
            window = torch.hann_window(n_fft)
        if not isinstance(window, torch.Tensor):
            raise TypeError('window must be a of type torch.Tensor')
        # In order for the window to be added as one of the Module's
        # parameters it has to be a nn.Parameter
        return nn.Parameter(window, requires_grad=False)
    

    def __dim_stft_mod(self, arr):
        if arr is None:
            return None
        return arr//self.hop+1

    def __dim_pv_mod(self, arr, rate):
        if arr is None:
            return None
        return (arr.float()/rate).long()+1

    def _out_dims(self, arr, rate=None):
        if arr is None: 
            return None
        new_arr = self.__dim_stft_mod(arr)
        if rate is None: 
            return new_arr
        return self.__dim_pv_mod(new_arr, rate)


    def _norm(self, stft):
        #return stft.pow(2).sum(-1).pow(1.0 / 2.0)
        return torch.norm(stft, dim=-1, p=2)

    def forward(self, x, lengths=None):
        """
        If x is a padded tensor then lengths should have the 
        corresponding sequence length for every element in the batch.

        Input: (batch, channel, signal)
        Output:(batch, channel, frequency_bins, time_hop)
        """
        with torch.no_grad():
            assert x.dim() == 3

            if self.pad > 0:
                with torch.no_grad():
                    x = F.pad(x, (self.pad, self.pad), "constant")

            spec = spectrogram(x,
                n_fft=self.n_fft,
                hop=self.hop, 
                window=self.window,
                **self.spec_kwargs)

            rate = None

            if torch.rand(1)[0] <= self.prob and self.training:
                spec, rate = self.phase_vocoder(spec)
                #print(rate)

            lengths = self._out_dims(lengths, rate)

            spec = self._norm(spec)

            if lengths is not None:            
                assert spec.size(0) == lengths.size(0)
                return spec, lengths

            return spec


class Melspectrogram(Spectrogram):

    def __init__(self, hop=None, n_mels=128, n_fft=2048, pad=0, window=None, sr=44100, norm=None, **spec_kwargs):
        
        super(Melspectrogram, self).__init__(hop, n_fft, pad, window, sr, **spec_kwargs)

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.mel_fb, self.mel_freq_vals = self._build_filter()
        self.norm = {
            'whiten':spec_whiten,
            'db' : amplitude_to_db
            }.get(norm, None)

    def _build_filter(self):
        # Get the mel filter matrix and the mel frequency values
        mel_fb, mel_f = create_mel_filter(
                                    self.n_fft//2 + 1,
                                    self.sr, 
                                    n_mels=self.n_mels)
        # Cast filter matrix as nn.Parameter so it's loaded on model's device 
        return nn.Parameter(mel_fb, requires_grad=False), mel_f

    def forward(self, x, lengths=None):

        spec = super(Melspectrogram, self).forward(x, lengths)
        if isinstance(spec, tuple):
            spec, lengths = spec

        spec = torch.matmul(spec.transpose(2,3), self.mel_fb).transpose(2,3)

        if self.norm is not None:
            spec = self.norm(spec)

        if lengths is not None:
            return spec, lengths
        
        return spec

    def __repr__(self):
        param_str = '(n_mels={}, n_fft={}, norm={}, stretch_param={})'.format(
                        self.n_mels, self.n_fft, self.norm.__name__, self.stretch_param)
        return self.__class__.__name__ + param_str