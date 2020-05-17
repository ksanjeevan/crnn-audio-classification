
import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from data.data_sets import FolderDataset
from utils.util import list_dir, load_image, load_audio


class FolderDataManager(object):

    def __init__(self, config):

        load_formats = {
                'image':load_image,
                'audio':load_audio
                }

        assert np.sum(list(config['splits'].values())) >= .999, "Splits must add up to 1"
        assert config['format'] in load_formats, "Pass valid data format"

        self.dir_path = config['path']
        self.loader_params = config['loader']

        self.splits = config['splits']

        self.load_func = load_formats[config['format']]
                
        data_dic, self.mappings, self.classes = self._get_dic()
        self.class_counts = self._class_counts(data_dic)
        
        path_splits = os.path.join(self.dir_path, '.splits.json')
        if os.path.isfile(path_splits):
            self.data_splits = torch.load(path_splits)
        else:         
            data_arr = self._get_arr(data_dic)        
            self.data_splits = self._get_splits(data_arr)
            torch.save(self.data_splits, path_splits) 


    def _get_splits(self, arr):
        np.random.seed(0)
        ret = {s:[] for s in self.splits.keys()}
        split_vec = np.concatenate([[s]*round(len(arr)*p) for s,p in self.splits.items()])
        np.random.shuffle(split_vec)

        for i in range(len(arr)):
            ret[split_vec[i]].append( arr[i] )

        return ret

    def _get_dic(self):

        ret = {}

        classes = list_dir(self.dir_path)

        class_to_idx = dict(zip(classes, np.arange(len(classes))))
        idx_to_class = dict(zip(np.arange(len(classes)), classes))
        mappings = {'idx_to_class':idx_to_class, 'class_to_idx':class_to_idx}

        for c in classes:
            c_path = os.path.join(self.dir_path, c)
            ret[c] = []

            for n in list_dir(c_path):
                ret[c].append( os.path.join(c_path, n) )
 
        return ret, mappings, classes
    

    def _get_arr(self, data_dic):
        ret = [];
        for c, paths in data_dic.items():
            for path in paths:
                class_idx = self.mappings['class_to_idx'][c]
                ret.append({'path':path, 'class':c, 'class_idx':class_idx})
        return ret

    def _class_counts(self, data_dic):
        ret = {}
        for k,v in data_dic.items():
            ret[k] = len(v)
        return ret


    def get_loader(self, name, transfs):
        assert name in self.data_splits
        dataset = FolderDataset(self.data_splits[name], load_func=self.load_func, transforms=transfs)

        return data.DataLoader(dataset=dataset, **self.loader_params, collate_fn=self.pad_seq)

    def pad_seq(self, batch):
        # sort_ind should point to length
        sort_ind = 0
        sorted_batch = sorted(batch, key=lambda x: x[0].size(sort_ind), reverse=True)
        seqs, srs, labels = zip(*sorted_batch)
        lengths, srs, labels = map(torch.LongTensor, [[x.size(sort_ind) for x in seqs], srs, labels])
        # seqs_pad -> (batch, time, channel) 
        seqs_pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        #seqs_pad = seqs_pad_t.transpose(0,1)
        return seqs_pad, lengths, srs, labels



class CSVDataManager(object):

    def __init__(self, config):

        load_formats = {
                'image':load_image,
                'audio':load_audio
                }


        assert config['format'] in load_formats, "Pass valid data format"

        self.dir_path = config['path']
        self.loader_params = config['loader']

        self.splits = config['splits']

        self.load_func = load_formats[config['format']]
    
        mfile = os.path.join(self.dir_path, 'metadata/UrbanSound8K.csv') 
        metadata_df = pd.read_csv(mfile).sample(frac=1)
        self.metadata_df = self._remove_too_small(metadata_df, 1)

        self.classes = self._get_classes(self.metadata_df[['class', 'classID']])
        self.data_splits = self._10kfold_split(self.metadata_df)
        

    def _remove_too_small(self, df, min_sec=0.5):
        # Could also make length 1 if it is below 1 and then implicitly we
        # are padding the results
        dur_cond = (df['end'] - df['start'])>=min_sec
        not_train_cond = ~df['fold'].isin(self.splits['train'])

        return df[(dur_cond)|(not_train_cond)]

    def _get_classes(self, df):
        c_col = df.columns[0]
        idx_col = df.columns[1]
        return df.drop_duplicates().sort_values(idx_col)[c_col].unique()

    def _10kfold_split(self, df):
        ret = {}
        for s, inds in self.splits.items():

            df_split = df[df['fold'].isin(inds)]
            ret[s] = []

            for row in df_split[['slice_file_name', 'class', 'classID', 'fold']].values:
                fold_mod = 'audio/fold%s'%row[-1]
                fname = os.path.join(self.dir_path, fold_mod, '%s'%row[0])
                ret[s].append( {'path':fname, 'class':row[1], 'class_idx':row[2]} )
        return ret

    def get_loader(self, name, transfs):
        assert name in self.data_splits
        dataset = FolderDataset(self.data_splits[name], load_func=self.load_func, transforms=transfs)

        return data.DataLoader(dataset=dataset, **self.loader_params, collate_fn=self.pad_seq)

    def pad_seq(self, batch):
        # sort_ind should point to length
        sort_ind = 0
        sorted_batch = sorted(batch, key=lambda x: x[0].size(sort_ind), reverse=True)
        seqs, srs, labels = zip(*sorted_batch)
        
        lengths, srs, labels = map(torch.LongTensor, [[x.size(sort_ind) for x in seqs], srs, labels])

        # seqs_pad -> (batch, time, channel) 
        seqs_pad = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        #seqs_pad = seqs_pad_t.transpose(0,1)
        return seqs_pad, lengths, srs, labels


if __name__ == '__main__':

    pass




