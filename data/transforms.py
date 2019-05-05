

import numpy as np

import torch
from torchvision import transforms



class ImageTransforms(object):

    def __init__(self, name, size, scale, ratio, colorjitter):
        self.transfs = {
            'val': transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size=size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size, scale=scale, ratio=ratio),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=colorjitter[0], 
                    contrast=colorjitter[1], 
                    saturation=colorjitter[2]),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }[name]


    def apply(self, data, target):
        return self.transfs(data), target


class AudioTransforms(object):

    def __init__(self, name, args):
        
        self.transfs = {
            'val': transforms.Compose([
                ProcessChannels(args['channels']),
                ToTensorAudio()
            ]),
            'train': transforms.Compose([
                ProcessChannels(args['channels']),
                AdditiveNoise(*args['noise']),
                RandomCropLength(*args['crop']),
                ToTensorAudio()
            ])
        }[name]
        
    def apply(self, data, target):
        audio, sr = data
        # audio -> (time, channel)
        return self.transfs(audio), sr, target
        
    def __repr__(self):
        return self.transfs.__repr__()


class ProcessChannels(object):

    def __init__(self, mode):
        self.mode = mode

    def _modify_channels(self, audio, mode):
        if mode == 'mono':
            new_audio = audio if audio.ndim == 1 else audio[:,:1]
        elif mode == 'stereo':
            new_audio = np.stack([audio]*2).T if audio.ndim == 1 else audio
        elif mode == 'avg':
            new_audio= audio.mean(axis=1) if audio.ndim > 1 else audio
            new_audio = new_audio[:,None] 
        else:
            new_audio = audio
        return new_audio

    def __call__(self, tensor):
        return self._modify_channels(tensor, self.mode)

    def __repr__(self):
        return self.__class__.__name__ + '(mode={})'.format(self.mode)


class ToTensorAudio(object):

    def __call__(self, tensor):
        return torch.from_numpy(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AugmentationTransform(object):

    def __init__(self, prob=None, sig=None, dist_type='uniform'):
        self.sig, self.dist_type = sig, dist_type 
        self.dist = self._get_dist(sig, dist_type)
        self.prob = prob

    def _get_dist(self, sig, dist_type):
        dist = None
        if dist_type == 'normal':
            dist = lambda x: np.random.normal(0, sig, x) 
        elif dist_type == 'uniform':
            dist = lambda x: np.random.uniform(-sig, sig, x)
        elif dist_type == 'half':
            dist = lambda x: np.clip(
                                np.abs(
                                    np.random.normal(0, sig, x)),
                                a_min=0.0,
                                a_max=0.8)
        else:
            raise ValueError('Unimplemented distribution')
        return dist

    def __call__(self, tensor):
        if np.random.rand() <= self.prob:
            return self.transform(tensor)
        return tensor

    def transform(self, tensor):
        raise NotImplementedError

    def __repr__(self):
        param_str = '(prob={}, sig={}, dist_type={})'.format(
                        self.prob, self.sig, self.dist_type)
        return self.__class__.__name__ + param_str
    

class AdditiveNoise(AugmentationTransform):

    def  __init__(self, prob, sig, dist_type='normal'):
        super(AdditiveNoise, self).__init__(prob, sig, dist_type)


    def _noise(self, length):
        return self.dist(length)

    def transform(self, tensor):
        noise = self._noise(tensor.shape[0])[:,None]
        return tensor + noise


class RandomCropLength(AugmentationTransform):

    def __init__(self, prob, sig, dist_type='half'):
        super(RandomCropLength, self).__init__(prob, sig, dist_type)

    def transform(self, tensor):
        ind_start, ind_end, perc = self._crop_inds(tensor.shape[0])
        return self._check_zero(tensor[ind_start:ind_end])

    def _check_zero(self, tensor):
        return tensor + 1e-8 if tensor.sum() == 0 else tensor

    def _crop_inds(self, length):
        d = self.dist(1)[0]
        assert d < 0.9

        perc = 1 - d
        new_length = np.round(length * perc).astype(int)
        max_start = length - new_length + 1
        ind_start = np.random.randint(0, max_start)
        ind_end = ind_start + new_length
        return ind_start, ind_end, perc




class ModifyDuration(object):

    def __init__(self, duration):
        self.duration = duration

    def __call__(self, tensor):
        return self._modify_duration(tensor, self.duration)

    def __repr__(self):
        return self.__class__.__name__ + '(duration={})'.format(self.duration)


    def _modify_duration(self, audio, dur):
        
        if dur < len(audio):
            max_index_start = len(audio) - dur
            index_start = np.random.randint(0,max_index_start)
            index_end = index_start + dur
            new_audio = audio[index_start:index_end]
        else:
            ratio = dur/len(audio)
            full_reps = [audio]*int(ratio)
            padd_reps = [audio[:round(len(audio)*(ratio%1))]]
            new_audio = np.concatenate(full_reps + padd_reps, axis=0)
            
        return new_audio 






