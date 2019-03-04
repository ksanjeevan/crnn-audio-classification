#import importlib
import os
import tensorboardX

class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writers = {}
        if enable:
            train_log_path = os.path.join(writer_dir, 'train')
            valid_log_path = os.path.join(writer_dir, 'valid')
            
            self.writers = {
                            'train' : tensorboardX.SummaryWriter(train_log_path),
                            'valid' : tensorboardX.SummaryWriter(valid_log_path)
                    }

        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 
        'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __len__(self):
        return len(self.writers)

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:# and (self.mode in self.writers):
            #add_data = getattr(self.writer, name, None)
            add_data = getattr(self.writers.get(self.mode, None), name, None)
            
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    #add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs)
                    add_data('{}'.format(tag), data, self.step, *args, **kwargs)
            return wrapper
        else:

            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr
            