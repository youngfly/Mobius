import io
import json
import numpy as np
import random
import lmdb
import torch
from functools import partial
from PIL import Image
from PIL import Image, ImageSequence

from torch.utils import data
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms as T


frame_lmdb_file = '/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/jiapeizhang/emoticon_processed/frames.lmdb'
frames_env = lmdb.open(
    frame_lmdb_file, map_size=int(1e11),
    readonly=True, lock=False,
) # map_size: 100G
frames_txn = frames_env.begin(buffers=True)


CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def preprocess_img(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]
    img = img.copy().convert('RGBA')
    bg = Image.new('RGBA', img.size, (255,255,255,255))
    bg.paste(img, mask=img)
    bg = bg.convert(mode)
    return bg

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    for frame in ImageSequence.Iterator(img):
        frame = frame.copy().convert('RGBA')
        bg = Image.new('RGBA', (256, 256), (255,255,255,255))
        bg.paste(frame, (8, 8), mask=frame)
        yield bg.convert(mode)

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def frames_to_tensor(frames, channels = 3, transform = T.ToTensor()):
    tensors = [transform(preprocess_img(f, channels = channels)) for f in frames]
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return torch.cat((t, t[:, :frames-f]), dim=1)
    # return F.pad(t, (0, 0, 0, 0, 0, frames - f))

class Dataset(data.Dataset):
    def __init__(
        self,
        json_file,
        image_size,
        channels = 3,
        num_frames = 8,
        # horizontal_flip = False,
        force_num_frames = True,
    ):
        super().__init__()
        self.data = self.load_data(json_file)
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            # T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
    ])

    def load_data(self, file):
        data = []
        with open(file) as fin:
            for l in fin:
                data.append(json.loads(l))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        caption = item['meanings'] + ', ' + item['llava_caption']
        frames = self.get_frames(item['md5'], item['nframes'])
        tensor = frames_to_tensor(frames, self.channels, transform = self.transform)
        tensor = self.cast_num_frames_fn(tensor)
        return {'caption': caption, 'tensor': tensor}

    def get_frames(self, md5, nframes):
        frames = []
        for idx in range(min(nframes, self.num_frames)):
            key = f'{md5}_{idx}'
            frame = frames_txn.get(key.encode())
            frame = Image.open(io.BytesIO(frame))
            frames.append(frame)
        return frames

