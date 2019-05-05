import os
import torch
from tqdm import tqdm

from net import MelspectrogramStretch
from utils import plot_heatmap, mkdir_p

class ClassificationEvaluator(object):

    def __init__(self, data_loader, model):

        self.data_loader = data_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.mel = MelspectrogramStretch(norm='db').to(self.device)

    def evaluate(self, metrics, debug=False):
        with torch.no_grad():
            total_metrics = torch.zeros(len(metrics))
            for batch_idx, batch in enumerate(tqdm(self.data_loader)):
                batch = [b.to(self.device) for b in batch]
                data, target = batch[:-1], batch[-1]
                
                output = self.model(data)

                self.model.classes
                batch_size = data[0].size(0)
                
                if debug:
                    self._store_batch(data, batch_size, output, target)
                
                for i, metric in enumerate(metrics):
                    total_metrics[i] += metric(output, target) * batch_size

            size = len(self.data_loader.dataset)
            ret = {met.__name__ : "%.3f"%(total_metrics[i].item() / size) for i, met in enumerate(metrics)}
            return ret




    def _store_batch(self, data, batch_size, output, target):

        path = 'eval_batch'
        mkdir_p(path)
        sig, lengths, _ = data

        inds = output.argmax(1)
        confs = torch.exp(output)[torch.arange(batch_size), inds]

        spec, lengths = self.mel(sig.transpose(1,2).float(), lengths)
        
        for i in range(batch_size):
            if inds[i] == target[i]:
                label = self.model.classes[inds[i]]
                pred_txt = "%s (%.1f%%)"%(label, 100*confs[inds[i]])
                out_path = os.path.join(path, '%s.png'%i)      
                plot_heatmap(spec[i][...,:lengths[i]].cpu().numpy(), 
                            out_path, 
                            pred=pred_txt)
