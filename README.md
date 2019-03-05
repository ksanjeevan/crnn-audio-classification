

# PyTorch Audio Classification: Urban Sounds


Classification of audio with variable length using a CNN + LSTM architecture on the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset.


Example results:
<p align="center">
<img src="result_plots/specs.png" width="850px"/>
</p>


### Features
- Easily define CRNN in .cfg format
- Spectrogram computation on GPU
- Audio data augmentation: Cropping, White Noise, Time Stretching (using phase vocoder on GPU!)


### Simple Usage
// document


### Results


Printing example model:
```
AudioCRNN(
  (spec): Melspectrogram(n_mels=128, n_fft=2048, norm=spec_whiten, stretch_param=[0.5, 0.3])
  (convs): Sequential(
    (conv2d_1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    (batchnorm2d_1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (elu_1): ELU(alpha=1.0)
    (maxpool2d_1): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (conv2d_2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (batchnorm2d_2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (elu_2): ELU(alpha=1.0)
    (maxpool2d_2): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
    (conv2d_3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (batchnorm2d_3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (elu_3): ELU(alpha=1.0)
    (maxpool2d_3): MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
  )
  (lstm): LSTM(128, 64, num_layers=2, batch_first=True)
  (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (lin1): Linear(in_features=64, out_features=10, bias=True)
)
Trainable parameters: 139786
```

##### Augmentation
// document

##### 10-Fold Cross Validation
// need to run this


### Usage


#### Training
```bash
./run.py train -c config.json --cfg conv.cfg
```

##### Config file explanation

```bash
{
    "name"          :   "Urban Testing", # Name of the run
    "data"          :   {
                            "type"      :   "CSVDataManager", # Type of data manager
                            "path"      :   "/home/kiran/Documents/DATA/UrbanSound8K", # path to dataset
                            "format"    :   "audio", # data format
                            "loader"    :   { # Data loader information
                                                "batch_size"    : 24,
                                                "num_workers"   : 4,
                                                "shuffle"       : true,
                                                "drop_last"     : true
                                            },
                            "splits"    :   { # split configuration
                                                "train" : [1,2,3,4,5,6,7,8,9], 
                                                "val"   : [10]                                            }
                        },
    "transforms"    :   {
                            "type"      :   "AudioTransforms", 
                            "args"      :   {
                                                "channels"       : "avg", # how to treat mono, stereo
                                                "noise"    : [0.5, 0.005], # [prob of augment, param]
                                                "crop"     : [0.4, 0.25] # [prob of augmentat, param]
                                            }
                        },
    "optimizer"     :   { # Optimizer config
                            "type"      :   "Adam",
                            "args"      :   {
                                                "lr"            : 0.0005,
                                                "weight_decay"  : 0.02,
                                                "amsgrad"       : true
                                            }
                        },
    "lr_scheduler"   :   { # Learning rate schedule
                            "type"      :   "StepLR",
                            "args"      :   {
                                                "step_size" : 10,
                                                "gamma"     : 0.5
                                            }
                        },
    "model"         :   { # Model type
                            "type"      :   "AudioCRNN"
                        },
    "train"         :   { # Training parameters
                            "loss"      :   "nll_loss",
                            "epochs"    :   100,
                            "save_dir"  :   "saved_testing/",
                            "save_p"    :   1,
                            "verbosity" :   2,
                            
                            "monitor"   :   "min val_loss",
                            "early_stop":   8,
                            "tbX"       :   true
                        },
    "metrics"       :   "classification_metrics" # Metrics to use (defined in net/metric.py)

}
```

##### TensorboardX
// document

#### Evaluation


```bash
./run.py eval -r /saved/0303_151917/checkpoints/model_best.pth
```

Then obtain defined metrics:
```bash
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 34/34 [00:03<00:00, 12.68it/s]
{'avg_precision': '0.725', 'avg_recall': '0.719', 'accuracy': '0.804'}
```

### Other

Visualizing architecture used: 
<p align="center">
<img src="result_plots/arch.png" width="400px"/>
</p>


### To Do
- [ ] commit jupyter notebook dataset exploration
- [x] CRNN entirely defined in .cfg
- [ ] Some bug in 'infer'
- [ ] Run 10-fold Cross Validation
- [ ] Comment things




