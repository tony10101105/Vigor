import numpy as np
import json
from torch.utils.data import Dataset
from functools import partial

from .utils import dataset_to_dataloader, max_io_workers
from .utils import check_segmented_object_order, sample_scan_object, pad_samples
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from .utils import ScannetDatasetConfig


class ListeningDataset(Dataset):
    def __init__(self, scans, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 multilabel_pretraining=False, lang_multilabel=False,
                 cascading=False, order_len=4):

        self.scans = scans
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1 # to account for the target.
        self.class_to_idx = class_to_idx
        self.object_transformation = object_transformation
        if not check_segmented_object_order(scans):
            raise ValueError
        with open('data/butd_pcnet_cls_results.json') as fid:
            self.cls_results = json.load(fid) # the scannet object classification results provided by butd-detr

        self.scannetconfig_nr3d = ScannetDatasetConfig('nr3d')
        self.scannetconfig_butd = ScannetDatasetConfig('butd')
        self.lang_multilabel = lang_multilabel
        self.multilabel_pretraining = multilabel_pretraining
        self.cascading = cascading
        self.order_len = order_len

    def __len__(self):
        return len(self.scans)

    def find_closest(self, context, previous_pos, previous_center, ord):
        min_dis = 100000000
        current_pos = -1
        current_center = None
        for j, o in enumerate(context):
            dis = np.linalg.norm(previous_center - o.get_bbox().center())
            if dis < min_dis and j != previous_pos and o.instance_label == ord:
                min_dis = dis
                current_pos = j
                current_center = o.get_bbox().center()
        return current_pos, current_center

    def find_farthest(self, context, previous_pos, previous_center, ord):
        max_dis = -100000000
        current_pos = -1
        current_center = None
        for j, o in enumerate(context):
            dis = np.linalg.norm(previous_center - o.get_bbox().center())
            if dis > max_dis and j != previous_pos and o.instance_label == ord:
                max_dis = dis
                current_pos = j
                current_center = o.get_bbox().center()
        return current_pos, current_center

    def __getitem__(self, index):
        # close-to-far query data synthesis
        res = dict()
        scan_id = list(self.scans.keys())[index]
        scan = self.scans[scan_id]
        distractors = [o for o in scan.three_d_objects if o.instance_label in self.scannetconfig_nr3d.type2class.keys()]
        context = distractors[:self.max_distractors]
        np.random.shuffle(context)
    
        all_cls = [o.instance_label for o in context]
        modes = 'farthest'

        # deal with exception scenes
        if scan_id in ['scene0013_00', 'scene0269_00', 'scene0423_01', 'scene0423_02']:
            anchor_pos_tmp = all_cls.index('coffee table')
        elif scan_id == 'scene0466_00':
            anchor_pos_tmp = all_cls.index('floor')
        elif scan_id == 'scene0484_01':
            anchor_pos_tmp = all_cls.index('couch cushions')
        elif scan_id in ['scene0638_00', 'scene0013_01', 'scene0269_01']:
            anchor_pos_tmp = all_cls.index('table')
        elif scan_id == 'scene0423_00':
            anchor_pos_tmp = all_cls.index('end table')
        else:
            anchor_pos_tmp = 0 # assign an initial anchor object
        anchor = context[anchor_pos_tmp]
        del context[anchor_pos_tmp]

        context = [o for o in context if o.instance_label != anchor.instance_label]
        if len(context) == 0:
            raise Exception('empty context!')
        anchor_pos = np.random.randint(len(context))
        context.insert(anchor_pos, anchor)
        
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])

        if self.object_transformation is not None:
            samples = self.object_transformation(samples)
        res['context_size'] = len(samples)
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)

        res['scan_id'] = scan_id
        box_info = np.zeros((self.max_context_size, 4))
        box_info[:len(context),0] = [o.get_bbox().cx for o in context]
        box_info[:len(context),1] = [o.get_bbox().cy for o in context]
        box_info[:len(context),2] = [o.get_bbox().cz for o in context]
        box_info[:len(context),3] = [o.get_bbox().volume() for o in context]
        box_info_center = np.zeros((self.max_context_size, 3))
        box_info_center[:len(context)] = [o.get_bbox().center() for o in context]
        box_corners = np.zeros((self.max_context_size, 8, 3))
        box_corners[:len(context)] = [o.get_bbox().corners for o in context]
        res['objects'] = pad_samples(samples, self.max_context_size)

        res['center_coors'] = box_info_center
        res['corner_coors'] = box_corners
        
        # get object mask
        obj_existance = np.zeros((self.max_context_size, 1))
        obj_existance[:len(context),0] = 1
        res['obj_mask'] = obj_existance

        # fake utterance
        all_obj_classes = [o.instance_label for o in context]
        all_classes = list(set(all_obj_classes))
        all_classes.remove(anchor.instance_label)
        np.random.shuffle(all_classes)
        order = [anchor.instance_label] + all_classes[:self.order_len-1]
        if len(order) < self.order_len:
            if all_obj_classes.count(order[-1]) >= (self.order_len-len(order)+1):
                while len(order) < self.order_len:
                    order.append(order[-1])
            else:
                # rearrange the order
                all_cls_num = [all_classes.count(i) for i in all_classes]
                max_num_cls = all_classes[all_cls_num.index(max(all_cls_num))]
                all_classes.remove(max_num_cls)
                order = [anchor.instance_label] + all_classes + [max_num_cls]
                while len(order) < self.order_len:
                    order.append(order[-1])

        res['order_labels'] = np.array([self.class_to_idx[i] for i in order])

        if self.multilabel_pretraining:# Each order has only one object as answer
            ordered_multilabel_gt = []
            rel_coors = [] # vectors that take an object as origin
            previous_center, previous_pos = None, None
            res['flag'] = False
            for i, obj in enumerate(order):
                mask = np.zeros(self.max_context_size, dtype=np.bool)
                if i == 0:
                    mask[anchor_pos] = True
                    previous_center = box_info_center[anchor_pos]
                    previous_pos = anchor_pos
                    rel_coor =  box_info_center - previous_center
                    rel_coor[len(context):] = 0
                    assert context[anchor_pos].instance_label == anchor.instance_label
                    if len(order) == 1:
                        res['target_pos'] = anchor_pos
                else:
                    if modes == 'nearest':
                        current_pos, current_center = self.find_closest(context, previous_pos, previous_center, order[i]) 
                    else:
                        current_pos, current_center = self.find_farthest(context, previous_pos, previous_center, order[i])
                    if current_pos == -1:
                        res['flag'], mask[previous_pos] = True, True
                        if i == len(order)-1:
                            res['target_pos'] = previous_pos
                        rel_coor[:] = 0
                        continue

                    rel_coor =  box_info_center - current_center
                    rel_coor[len(context):] = 0
                    mask[current_pos] = True
                    previous_center = current_center
                    previous_pos = current_pos
                    if i == len(order)-1:
                        res['target_pos'] = current_pos
                    
                rel_coors.append(rel_coor)
                    
                ordered_multilabel_gt.append(mask)
            rel_coors = np.stack(rel_coors, axis=0)
            ordered_multilabel_gt = np.stack(ordered_multilabel_gt, axis=0).astype(int)
            res['rel_coors'] = rel_coors
            res['ordered_multilabel_gt'] = ordered_multilabel_gt

        if len(order) == 6:
            synthesized_utterance = "There is a {} in the room, find the {} farthest to it, and then find the {} farthest to that {}, and then find the {} farthest to that {}, and then find the {} farthest to that {}, and finally you can see the {} farthest to that {}.".format(
                order[0], order[1], order[2], order[1], order[3], order[2], order[4], order[3], order[5], order[4]
            )
        elif len(order) == 5:
            synthesized_utterance = "There is a {} in the room, find the {} farthest to it, and then find the {} farthest to that {}, and then find the {} farthest to that {}, and finally you can see the {} farthest to that {}.".format(
                order[0], order[1], order[2], order[1], order[3], order[2], order[4], order[3]
            )
        elif len(order) == 4:
            synthesized_utterance = "There is a {} in the room, find the {} farthest to it, and then find the {} farthest to that {}, and finally you can see the {} farthest to that {}.".format(
                order[0], order[1], order[2], order[1], order[3], order[2]
            )
        elif len(order) == 3:
            synthesized_utterance = "There is a {} in the room, find the {} farthest to it, and finally you can see the {} farthest to that {}.".format(
                order[0], order[1], order[2], order[1]
            )
        elif len(order) == 2:
            synthesized_utterance = "There is a {} in the room, find the {} farthest to it.".format(
                order[0], order[1]
            )
        elif len(order) == 1:
            synthesized_utterance = "Find the {} in the room.".format(
                order[0]
            )

        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [order[-1] == o.instance_label for o in context]
        cascaded_order = []
        if self.cascading:
            for i in range(len(order)):
                cat_names = ''
                sub_order = list(dict.fromkeys(order[i:]))
                for name in sub_order:
                    cat_names += name
                    cat_names += ', '
                cat_names = cat_names[:-2]
                cascaded_order.append(cat_names)
            res['referential_order'] = cascaded_order
        else:
            res['referential_order'] = order

        # for masking trasnformer during the training and inference
        pred_class_mask = []
        for i in range(len(order)):
            if not self.cascading:
                obj = [order[i]]
            else:
                obj = order[i:]
            mask = np.zeros(self.max_context_size, dtype=np.bool)
            tmp = []
            for o in context:
                pred_id = self.cls_results[scan_id][o.object_id]
                if pred_id == -1:
                    pred_id = 325
                pred_classname = self.scannetconfig_butd.class2type[pred_id]
                tmp.append(True) if pred_classname in obj else tmp.append(False)
            mask[:len(context)] = tmp  
            pred_class_mask.append(mask)
        pred_class_mask = np.stack(pred_class_mask, axis=0)

        res['target_class'] = self.class_to_idx[order[-1]]
        res['target_class_mask'] = target_class_mask
        res['is_nr3d'] = False
        res['box_info'] = box_info
    
        res['target_object'] = order[-1]
        res['anchor_objects'] = 'trivial' # not used
        res['pred_class_mask'] = pred_class_mask
        res['tokens'] = synthesized_utterance

        return res


def make_data_loaders(args, scans_split, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)

    for split in splits:
        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1

        new_scans = {key : scans[key] for key in scans.keys() if key in scans_split[split]}
        
        # delete some exception scenes
        if split == 'train' and args.order_len >= 5:
            del new_scans['scene0587_00']
            del new_scans['scene0587_01']
            del new_scans['scene0587_02']
            del new_scans['scene0587_03']
            del new_scans['scene0148_00']
            del new_scans['scene0219_00']
            del new_scans['scene0543_00']
            del new_scans['scene0543_01']
            del new_scans['scene0543_02']
            del new_scans['scene0484_00']
            del new_scans['scene0484_01']
            del new_scans['scene0269_00']
            del new_scans['scene0269_01']
            del new_scans['scene0269_02']
            del new_scans['scene0013_00']
            del new_scans['scene0013_01']
            del new_scans['scene0013_02']
            del new_scans['scene0622_00']
            del new_scans['scene0622_01']
            del new_scans['scene0218_00']
            del new_scans['scene0218_01']
            del new_scans['scene0337_00']
            del new_scans['scene0337_01']
            del new_scans['scene0337_02']
            del new_scans['scene0125_00']
            del new_scans['scene0526_00']
            del new_scans['scene0526_01']
            del new_scans['scene0437_00']
            del new_scans['scene0437_01']
            del new_scans['scene0594_00']
            del new_scans['scene0292_00']
            del new_scans['scene0292_01']
            del new_scans['scene0173_00']
            del new_scans['scene0173_01']
            del new_scans['scene0173_02']
        if split == 'test' and args.order_len == 5:
            del new_scans['scene0609_00']
            del new_scans['scene0609_01']
            del new_scans['scene0609_02']
            del new_scans['scene0609_03']
            del new_scans['scene0432_00']
            del new_scans['scene0432_01']
            del new_scans['scene0660_00']

        if split == 'train' and args.order_len >= 6:
            del new_scans['scene0248_00']
            del new_scans['scene0248_01']
            del new_scans['scene0248_02']
            del new_scans['scene0082_00']
            del new_scans['scene0444_00']
            del new_scans['scene0444_01']
            del new_scans['scene0513_00']
            del new_scans['scene0627_00']
            del new_scans['scene0627_01']
            del new_scans['scene0318_00']
            del new_scans['scene0071_00']
            del new_scans['scene0192_00']
            del new_scans['scene0192_01']
            del new_scans['scene0192_02']
            del new_scans['scene0113_00']
            del new_scans['scene0113_01']
            del new_scans['scene0638_00']
            del new_scans['scene0442_00']
            del new_scans['scene0099_00']
            del new_scans['scene0099_01']
            del new_scans['scene0037_00']
            del new_scans['scene0123_00']
            del new_scans['scene0123_01']
            del new_scans['scene0123_02']
            del new_scans['scene0258_00']
            del new_scans['scene0290_00']
            del new_scans['scene0228_00']
        if split == 'test' and args.order_len >= 6:
            del new_scans['scene0609_00']
            del new_scans['scene0609_01']
            del new_scans['scene0609_02']
            del new_scans['scene0609_03']
            del new_scans['scene0701_00']
            del new_scans['scene0701_01']
            del new_scans['scene0701_02']
            del new_scans['scene0432_00']
            del new_scans['scene0432_01']
            del new_scans['scene0660_00']
            del new_scans['scene0355_00']
            del new_scans['scene0355_01']

        dataset = ListeningDataset(scans=new_scans,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   lang_multilabel=args.lang_multilabel,
                                   multilabel_pretraining=args.multilabel_pretraining,
                                   cascading=args.cascading,
                                   order_len=args.order_len)

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders
