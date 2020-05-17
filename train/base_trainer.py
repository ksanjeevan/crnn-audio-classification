import os
import math
import json
import logging
import datetime
import torch
import pandas as pd
from utils.util import mkdir_p
from utils.visualization import WriterTensorboardX

# Structure based off https://github.com/victoresque/pytorch-template
class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.train_logger = train_logger

        cfg_trainer = config['train']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_p']
        self.verbosity = cfg_trainer['verbosity']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = cfg_trainer.get('early_stop', math.inf)
        
        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], start_time, 'checkpoints')
        self.log_dir = os.path.join(cfg_trainer['save_dir'], start_time, 'logs')

        self.writer = WriterTensorboardX(self.log_dir, self.logger, cfg_trainer['tbX'])

        # Save configuration file into checkpoint directory:
        mkdir_p(self.checkpoint_dir)
        
        if self.config.get('cfg', None) is not None:
            cfg_save_path = os.path.join(self.checkpoint_dir, 'model.cfg')
            with open(cfg_save_path, 'w') as fw:
                fw.write(open(self.config['cfg']).read())
            self.config['cfg'] = cfg_save_path


        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)
    

    def train(self):
        """
        Full training logic
        """
        best_df = None
        not_improved_count = 0


        #f = open(os.path.join(self.log_dir, 'lr.txt'), 'w')


        for epoch in range(self.start_epoch, self.epochs + 1):
            
            # _train_epoch returns dict with train metrics ("metrics"), validation
            # metrics ("val_metrics") and other key,value pairs. Store/update them in log.
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__ : value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            c_lr = self.optimizer.param_groups[0]['lr']
            
            # print logged informations to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:

                    df = pd.DataFrame.from_dict([log]).T
                    df.columns = ['']
                    #self.logger.info('Epoch: {}'.format(epoch))
                    self.logger.info('{}'.format(df.loc[df.index!='epoch']))
                    self.logger.info('lr_0: {}'.format(c_lr) )
                    

            #f.write('%.5f\t%.5f\t%.5f\n'%(c_lr, result['loss'], result['metrics'][0]))
            #f.flush()
            self.writer.add_scalar('lr', c_lr)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    best_df = df
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    self.logger.info('Final:\n{}'.format(best_df.loc[best_df.index!='epoch']))
                    break

            if len(self.writer) > 0:
                self.logger.info(
                    '\nRun TensorboardX:\ntensorboard --logdir={}\n'.format(self.log_dir))
            
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
                #self.logger.info('\n\n\tTensorboardX Path: {}\n'.format(self.log_dir))
            
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'classes':self.model.classes
        }

        filename = os.path.join(self.checkpoint_dir, 'checkpoint-current.pth')
        #filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))
            self.logger.info("[IMPROVED]")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed. 

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

        
