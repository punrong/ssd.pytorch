from typing import List

from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = ['car', 'bus', 'van', 'others']

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        sequence = target.findall('sequence')
        for frame in sequence.findall('frame'):
            target_list = frame.find('target_list')
            target_id = target_list.find('target')
            for id in target_id.findall('id'):
                attribute = id.find('attribute')
                name = attribute.find('vehicle_type').text.lower.strip()
                bbox = id.find('box')

                pts = ['height', 'width', 'top', 'left']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(bbox.find(pt).text) - 1
                    # scale height or width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                    bndbox.append(cur_pt)
                label_idx = self.class_to_ind[name]
                bndbox.append(label_idx)
                res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

            # img_id = target.find('filename').text[:-4]
        print(res)
        try:
            print(np.array(res)[:, 4])
            print(np.array(res)[:, :4])
        except IndexError:
            print("\nINDEX ERROR HERE !\n")
            exit(0)

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=[('MVI_20011', '664'), ('MVI_20012', '936'), ('MVI_20032', '437'), ('MVI_20033', '784'),
                             ('MVI_20034', '800'), ('MVI_20035', '800'), ('MVI_20051', '906'), ('MVI_20052', '964'),
                             ('MVI_20061', '800'), ('MVI_20062', '800'), ('MVI_20063', '800'), ('MVI_20064', '800'),
                             ('MVI_20065', '1200'), ('MVI_39761', '1660'), ('MVI_39771', '570'), ('MVI_39781', '1865'),
                             ('MVI_39801', '885'), ('MVI_39811', '1070'), ('MVI_39821', '880'), ('MVI_39851', '1420'),
                             ('MVI_39861', '745'), ('MVI_39931', '1270'), ('MVI_40131', '1645'), ('MVI_40141', '1600'),
                             ('MVI_40152', '1750'), ('MVI_40161', '1490'), ('MVI_40162', '1765'), ('MVI_40171', '1150'),
                             ('MVI_40172', '2635'), ('MVI_40181', '1700'), ('MVI_40191', '2495'), ('MVI_40192', '2195'),
                             ('MVI_40201', '925'), ('MVI_40204', '1225'), ('MVI_40211', '1950'), ('MVI_40212', '1690'),
                             ('MVI_40213', '1790'), ('MVI_40241', '2320'), ('MVI_40243', '1265'), ('MVI_40244', '1345'),
                             ('MVI_40732', '2120'), ('MVI_40751', '1145'), ('MVI_40752', '2025'), ('MVI_40871', '1720'),
                             ('MVI_40962', '1875'), ('MVI_40963', '1820'), ('MVI_40981', '1995'), ('MVI_40991', '1820'),
                             ('MVI_40992', '2160'), ('MVI_41063', '1505'), ('MVI_41073', '1825'), ('MVI_63521', '2055'),
                             ('MVI_63525', '985'), ('MVI_63544', '1160'), ('MVI_63552', '1150'), ('MVI_63553', '1405'),
                             ('MVI_63554', '1445'), ('MVI_63561', '1285'), ('MVI_63562', '1185'), ('MVI_63563', '1390')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.transform = transform
        self.image_set = image_sets
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s', '%s.jpg')
        self.ids_for_annotation = list()
        self.ids = list()
        for (name, length) in image_sets:
            rootpath = osp.join(self.root, 'VEHICLE')
            for x in range(int(length)):
                if x < 10:
                    image = 'img0000' + str((x+1))
                elif x < 100:
                    image = 'img000' + str((x+1))
                elif x < 1000:
                    image = 'img00' + str((x+1))
                elif x < 10000:
                    image = 'img0' + str((x+1))
                self.ids.append((rootpath, name, image))
                self.ids_for_annotation.append((rootpath, name))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        img_annotation_id = self.ids_for_annotation[index]
        # img_path = self._imgpath[index]

        target = ET.parse(self._annopath % img_annotation_id).getroot()
        # img = cv2.imread(img_path % img_id)
        img = cv2.imread(self._imgpath % img_id)
        print(self._annopath % img_annotation_id)
        print(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            print('img'+ img + 'boxes' + boxes + 'labels' + labels)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        img_annotation_id = self.ids_for_annotation[index]
        anno = ET.parse(self._annopath % img_annotation_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)