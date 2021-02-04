import os
from os import listdir
from os.path import join
from PIL import Image,ImageEnhance
from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','.bmp'])
def _is_pil_image(img):
    return isinstance(img, Image.Image)
def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class CreateTrainValDataSet(Dataset):
    def __init__(self, train_val_dir, transform=None, use_trans_atmos=True, use_crop=False, random_resize = False):
        self.dir= train_val_dir
        self.transform=transform
        self.use_trans_atmos = use_trans_atmos
        self.use_crop = use_crop
        self.random_resize = random_resize

        if self.use_trans_atmos:
            self.data_files = {'haze':[],'trans':[],'atmos':[],'gt':[]}
            # self.data_files = {'haze':[],'depth':[],'trans':[],'atmos':[],'gt':[]}
        else:
            self.data_files = {'haze':[],'gt':[]}

        for key in self.data_files.keys():
            subdir = join(self.dir, key)
            self.data_files[key] += [join(subdir,x) for x in listdir(subdir) if is_image_file(x) ]
            # self.data_files[key].sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        # for root, dirs, files in os.walk(train_val_dir):
        #     for _dir in dirs:
        #         subdir = join(root, _dir)
        #         self.data_files[_dir] += [join(subdir,x) for x in listdir(subdir) if is_image_file(x) ]

    def __getitem__(self, index):
        haze_name = self.data_files['haze'][index]
        gt_name= self.data_files['gt'][index]
        haze = Image.open(haze_name).convert('RGB')
        gt = Image.open(gt_name).convert('RGB')
        new_size= 320,320

        if self.use_crop:
            width, height = haze.size
            left,top, right, bottom = 10, 10, width-10, height-10
            haze = haze.crop((left, top, right, bottom))
            gt = gt.crop((left, top, right, bottom))
        # for train_v9.py transfer learning on different dataset
        if self.random_resize:
            '''Resize'''
            width, height = haze.size
            if random.random() <= 0.5:
                new_w, new_h = np.random.randint(low=321,high=641,size=2)
                new_size = new_w, new_h
                # scale_factor_w = np.random.uniform(0.25,0.4)
                # scale_factor_h = np.random.uniform(0.3,0.55)
                # new_size= int(width*scale_factor_w), int(height*scale_factor_h)
            else:
                new_size = 320,320
            haze = haze.resize(new_size,Image.BILINEAR )
            gt = gt.resize(new_size,Image.BILINEAR)
            '''End Resize'''

            '''Color Contrast Brightness Sharpness'''
            # enhance_factor = np.random.uniform(1.0,1.5)
            # enhancer = ImageEnhance.Color(haze)
            # haze = enhancer.enhance(enhance_factor)

            # enhance_factor = np.random.uniform(0.7,1.0)
            # enhancer = ImageEnhance.Contrast(haze)
            # haze = enhancer.enhance(enhance_factor)

            # enhance_factor = np.random.uniform(0.7,1.0)
            # enhancer = ImageEnhance.Brightness(haze)
            # haze = enhancer.enhance(enhance_factor)

            # enhance_factor = np.random.uniform(0.7,1.0)
            # enhancer = ImageEnhance.Sharpness(haze)
            # haze = enhancer.enhance(enhance_factor)
            ''' End Enhance '''

        sample = {'haze': haze, 'gt': gt}

        if self.use_trans_atmos:
            # depth_name= self.data_files['depth'][index]
            trans_name= self.data_files['trans'][index]
            atmos_name = self.data_files['atmos'][index]

            # trans = Image.open(trans_name).convert('L')
            trans = Image.open(trans_name)
            atmos = Image.open(atmos_name)
            atmos = np.asarray(atmos, dtype=np.float32)/255.
            (r,g,b) = atmos[0,0,0],atmos[0,0,1],atmos[0,0,2]
            if self.use_crop:
                width, height = trans.size
                left,top, right, bottom = 10, 10, width-10, height-10
                trans = trans.crop((left, top, right, bottom))
                # atmos = atmos.crop((left, top, right, bottom))
            if self.random_resize:
                '''Resize'''
                trans = trans.resize(new_size, Image.BILINEAR)
                # atmos = atmos.resize(new_size, Image.BILINEAR)

            sample = {'haze': haze, 'trans': trans,'atmos': (r,g,b), 'gt': gt}
            # sample = {'haze': haze, 'depth': depth, 'trans': trans,'atmos': atmos, 'gt': gt}

        if self.transform:
            # apply transform to each sample in data_files
            sample = self.transform(sample)

        return sample
    def __len__(self):
        return len(self.data_files['haze'])

class Scale(object):
    """
    Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, output_size, interpolation=Image.BILINEAR, use_trans_atmos=True):
        assert isinstance(output_size, (int, tuple))
        # output_size: (H, W)
        self.output_size = output_size
        self.interpolation = interpolation
        self.use_trans_atmos= use_trans_atmos

    def __call__(self, sample):
        haze, gt = sample['haze'], sample['gt']

        w, h = haze.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        haze = haze.resize((new_w, new_h), self.interpolation)
        gt = gt.resize((new_w, new_h), self.interpolation)

        if self.use_trans_atmos:
            trans, atmos= sample['trans'],sample['atmos']
            trans = trans.resize((new_w, new_h), self.interpolation)
            # atmos = atmos.resize((new_w, new_h), self.interpolation)
            return {'haze': haze, 'trans': trans,'atmos': atmos,'gt': gt}
            # return {'haze': haze, 'depth': depth, 'trans': trans,'atmos': atmos,'gt': gt}

        return {'haze': haze, 'gt': gt}

class CenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, use_trans_atmos=True):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            # size: (H, W)
            assert len(size) == 2
            self.size = size

        self.use_trans_atmos= use_trans_atmos

    def __call__(self, sample):
        haze, gt = sample['haze'], sample['gt']
        w, h = haze.size
        new_h, new_w = self.size
        center_x = int( round((w-new_w)/2.) )
        center_y = int( round((h-new_h)/2.) )

        haze = haze.crop((center_x, center_y, center_x+new_w, center_y+new_h))
        gt = gt.crop((center_x, center_y, center_x+new_w, center_y+new_h))

        if self.use_trans_atmos:
            trans, atmos = sample['trans'], sample['atmos']
            trans = trans.crop((center_x, center_y, center_x+new_w, center_y+new_h))
            # atmos = atmos.crop((center_x, center_y, center_x+new_w, center_y+new_h))
            return {'haze': haze, 'trans': trans, 'atmos': atmos,'gt': gt}

        return {'haze': haze,'gt': gt}

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, output_size, use_trans_atmos=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            # output_size: (H,W)
            self.output_size = output_size

        self.use_trans_atmos= use_trans_atmos

    def __call__(self, sample):
        haze, gt = sample['haze'], sample['gt']
        w, h = haze.size
        new_h, new_w = self.output_size

        if w == new_w and h == new_h:
            return sample

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        haze = haze.crop((left, top, left+new_w, top+new_h))
        gt = gt.crop((left, top, left+new_w, top+new_h))

        if self.use_trans_atmos:
            trans, atmos = sample['trans'],sample['atmos']
            trans = trans.crop((left, top, left+new_w, top+new_h))
            # atmos = atmos.crop((left, top, left+new_w, top+new_h))
            return {'haze': haze, 'trans': trans, 'atmos': atmos,'gt': gt}
            # return {'haze': haze, 'depth': depth, 'trans': trans, 'atmos': atmos,'gt': gt}

        return {'haze': haze,'gt': gt}


class RandomHorizontalFlip(object):
    # input: PIL images
    def __init__(self, use_trans_atmos=True):
        self.use_trans_atmos= use_trans_atmos

    def __call__(self, sample):
        haze, gt = sample['haze'], sample['gt']
        flip = False

        if not _is_pil_image(haze):
            raise TypeError(
                'haze should be PIL Image. Got {}'.format(type(haze)))
        if not _is_pil_image(gt):
            raise TypeError(
                'gt should be PIL Image. Got {}'.format(type(gt)))

        if random.random() < 0.5:
            flip = True
            haze = haze.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        if self.use_trans_atmos:
            trans, atmos = sample['trans'], sample['atmos']
            # depth, trans, atmos = sample['depth'], sample['trans'], sample['atmos']
            # if not _is_pil_image(depth):
            #     raise TypeError(
            #         'depth should be PIL Image. Got {}'.format(type(depth)))
            if not _is_pil_image(trans):
                raise TypeError(
                    'trans should be PIL Image. Got {}'.format(type(trans)))
            # if not _is_pil_image(atmos):
            #     raise TypeError(
            #         'atmos should be PIL Image. Got {}'.format(type(atmos)))

            if flip:
                trans = trans.transpose(Image.FLIP_LEFT_RIGHT)
                # atmos = atmos.transpose(Image.FLIP_LEFT_RIGHT)
            return {'haze': haze, 'trans': trans,'atmos': atmos,'gt': gt}
            # return {'haze': haze, 'depth': depth, 'trans': trans,'atmos': atmos,'gt': gt}

        return {'haze': haze,'gt': gt}


class RandomVerticalFlip(object):
    # input: PIL images
    def __init__(self, use_trans_atmos=True):
        self.use_trans_atmos= use_trans_atmos
    def __call__(self, sample):
        haze, gt = sample['haze'], sample['gt']
        flip = False

        if not _is_pil_image(haze):
            raise TypeError(
                'haze should be PIL Image. Got {}'.format(type(haze)))
        if not _is_pil_image(gt):
            raise TypeError(
                'gt should be PIL Image. Got {}'.format(type(gt)))

        if random.random() < 0.5:
            flip = True
            haze = haze.transpose(Image.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.FLIP_TOP_BOTTOM)

        if self.use_trans_atmos:
            trans, atmos = sample['trans'], sample['atmos']
            # depth, trans, atmos = sample['depth'], sample['trans'], sample['atmos']
            # if not _is_pil_image(depth):
            #     raise TypeError(
            #         'depth should be PIL Image. Got {}'.format(type(depth)))
            if not _is_pil_image(trans):
                raise TypeError(
                    'trans should be PIL Image. Got {}'.format(type(trans)))
            # if not _is_pil_image(atmos):
            #     raise TypeError(
            #         'atmos should be PIL Image. Got {}'.format(type(atmos)))

            if flip:
                trans = trans.transpose(Image.FLIP_TOP_BOTTOM)
                # atmos = atmos.transpose(Image.FLIP_TOP_BOTTOM)
            return {'haze': haze, 'trans': trans,'atmos': atmos,'gt': gt}
            # return {'haze': haze, 'depth': depth, 'trans': trans,'atmos': atmos,'gt': gt}

        return {'haze': haze,'gt': gt}

class RandomRotation(object):
    # input: PIL images
    def __init__(self, use_trans_atmos=True):
        self.use_trans_atmos= use_trans_atmos
    def __call__(self, sample):
        haze, gt = sample['haze'], sample['gt']
        rotation = False

        if not _is_pil_image(haze):
            raise TypeError(
                'haze should be PIL Image. Got {}'.format(type(haze)))
        if not _is_pil_image(gt):
            raise TypeError(
                'gt should be PIL Image. Got {}'.format(type(gt)))

        if random.random() < 0.5:
            rotation = True
            angle = random.choice([90,180,270])
            haze = haze.rotate(angle)
            gt = gt.rotate(angle)

        if self.use_trans_atmos:
            trans, atmos = sample['trans'], sample['atmos']
            # depth, trans, atmos = sample['depth'], sample['trans'], sample['atmos']
            # if not _is_pil_image(depth):
            #     raise TypeError(
            #         'depth should be PIL Image. Got {}'.format(type(depth)))
            if not _is_pil_image(trans):
                raise TypeError(
                    'trans should be PIL Image. Got {}'.format(type(trans)))
            # if not _is_pil_image(atmos):
            #     raise TypeError(
            #         'atmos should be PIL Image. Got {}'.format(type(atmos)))

            if rotation:
                trans = trans.rotate(angle)
                # atmos = atmos.rotate(angle)
            return {'haze': haze, 'trans': trans,'atmos': atmos,'gt': gt}
            # return {'haze': haze, 'depth': depth, 'trans': trans,'atmos': atmos,'gt': gt}

        return {'haze': haze,'gt': gt}

class ToTensor(object):

    """
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    haze, gt: ByteTensor RGB
    depth: ByteTensor L
    """
    def __init__(self,is_test=False, use_trans_atmos=True):
        self.is_test = is_test
        self.use_trans_atmos = use_trans_atmos

    def __call__(self, sample):
        haze,  gt = sample['haze'], sample['gt']
        haze = self.to_tensor(haze)
        gt = self.to_tensor(gt)

        if self.use_trans_atmos:
            trans, atmos = sample['trans'],sample['atmos']
            # depth, trans, atmos = sample['depth'], sample['trans'],sample['atmos']
            # w,h = haze.size
            # depth = F.resize(depth, (h//2, w//2))
            # depth: PIL image
            # depth = depth.resize((w//2, h//2))
            # depth = self.to_tensor(depth)
            trans = self.to_tensor(trans)
            atmos = torch.tensor(atmos)
            return {'haze': haze, 'trans': trans, 'atmos': atmos,'gt':gt}
            # return {'haze': haze, 'depth': depth, 'trans': trans, 'atmos': atmos,'gt':gt}

        return {'haze': haze, 'gt':gt}

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        # (H,W,C)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # (C,H,W)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            # normalize to [0,1]
            return img.float().div(255)
        else:
            return img

class Normalize(object):
    """
    Apply after ToTensor()
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
      channel = (channel - mean) / std
    """
#     def __init__(self, mean, std, max_depth):
    def __init__(self, mean, std, use_trans_atmos=True):
        self.mean = mean
        self.std = std
        self.use_trans_atmos = use_trans_atmos
#         self.max_depth = max_depth

    def __call__(self, sample):
        haze, gt = sample['haze'], sample['gt']

        haze = F.normalize(haze, self.mean, self.std)
#         depth = (depth / 256).float() / self.max_depth

        if self.use_trans_atmos:
            trans, atmos = sample['trans'], sample['atmos']
            # depth, trans, atmos = sample['depth'], sample['trans'], sample['atmos']
            return {'haze': haze, 'trans': trans, 'atmos': atmos,'gt': gt}
            # return {'haze': haze, 'depth': depth, 'trans': trans, 'atmos': atmos,'gt': gt}

        return {'haze': haze,'gt': gt}


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensors):
        """
        Args:
            tensor (Tensor): Tensor images of size (B, C, H, W) to be unnormalized.
        Returns:
            Tensor: UnNormalized image.
        """
        tensor_new = tensors.clone()
        batch_size = tensor_new.size()[0]
        for i  in range(batch_size):
            for t, m, s in zip(tensor_new[i], self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return tensor_new

def getTrainLoader(train_dir,batch_size, image_size, trans_atmos,crop, enhance):
    train_transform = transforms.Compose([
                                          RandomCrop((image_size,image_size), use_trans_atmos = trans_atmos),
                                          RandomHorizontalFlip(use_trans_atmos = trans_atmos),
                                          RandomVerticalFlip(use_trans_atmos = trans_atmos),
                                          RandomRotation(use_trans_atmos = trans_atmos),
                                          ToTensor(use_trans_atmos = trans_atmos),
                                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],use_trans_atmos = trans_atmos)
                                        ])
    train_set = CreateTrainValDataSet(train_dir, train_transform,use_trans_atmos = trans_atmos, use_crop=crop, random_resize=enhance)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, train_set.__len__()


def getValLoader(val_dir,batch_size, image_size, trans_atmos, crop, enhance):
    val_transform = transforms.Compose([
                                          Scale((image_size,image_size), use_trans_atmos=trans_atmos),
                                          ToTensor(use_trans_atmos = trans_atmos),
                                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],use_trans_atmos = trans_atmos)
                                        ])
    val_set = CreateTrainValDataSet(val_dir, val_transform, use_trans_atmos = trans_atmos, use_crop=crop, random_resize=enhance)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return val_loader, val_set.__len__()

