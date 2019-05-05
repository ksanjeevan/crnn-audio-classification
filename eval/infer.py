

# make this import cofnigurable as something you pass to run
import torch
from utils.util import load_image, load_audio
from PIL import ImageDraw, ImageFont

class ImageInference:

    def __init__(self, model, transforms):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        
        self.transforms = transforms

    def infer(self, path):
        image = load_image(path)

        image_t, _ = self.transforms.apply(image, None)
        label, conf = self.model.predict( image_t.to(self.device) )

        return label, conf


    def draw(self, path, label, conf):
        
        image = load_image(path)
        draw = ImageDraw.Draw(image)        
        font = ImageFont.truetype('utils/Verdana.ttf', 15)
        draw.text((0,0), "%s (%.1f%%)"%(label, 100*conf),(255,0,255), font)
        image.save(path.split('.')[0] + '_pred.png')



import os
from net import MelspectrogramStretch
from utils import plot_heatmap

class AudioInference:

    def __init__(self, model, transforms):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.transforms = transforms

        self.mel = MelspectrogramStretch(norm='db')
        self.mel.eval()

    def infer(self, path):
        data = load_audio(path)
        sig_t, sr, _ = self.transforms.apply(data, None)

        length = torch.tensor(sig_t.size(0))
        sr = torch.tensor(sr)
        data = [d.unsqueeze(0).to(self.device) for d in [sig_t, length, sr]]
        label, conf = self.model.predict( data )

        return label, conf


    def draw(self, path, label, conf):
        
        sig, sr = load_audio(path)
        sig = torch.tensor(sig).mean(dim=1).view(1,1,-1).float()
        spec = self.mel(sig)[0]
        out_path = os.path.basename(path).split('.')[0] + '_pred.png'
        pred_txt = "%s (%.1f%%)"%(label, 100*conf)
        plot_heatmap(spec.cpu().numpy(), out_path, pred=pred_txt)
        

        