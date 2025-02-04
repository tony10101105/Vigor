import numpy as np
import json
import ast
from random import sample 
from torch.utils.data import Dataset
from functools import partial

from .utils import dataset_to_dataloader, max_io_workers
from .utils import check_segmented_object_order, sample_scan_object, pad_samples
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from .utils import ScannetDatasetConfig
from ...data_generation.nr3d import decode_stimulus_string


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False, multilabel_pretraining=False, lang_multilabel=False, 
                 cascading=False, order_len=4):
        self.references = references
        self.scans = scans
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1 # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        if not check_segmented_object_order(scans):
            raise ValueError

        with open('data/butd_pcnet_cls_results.json') as fid:
            self.cls_results = json.load(fid) # the scannet object classification results provided by butd-detr

        self.scannetconfig_butd = ScannetDatasetConfig('butd')
        self.multilabel_pretraining = multilabel_pretraining
        self.lang_multilabel = lang_multilabel
        self.cascading = cascading
        self.order_len = order_len

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan_id = ref['scan_id']
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        ori_tokens = ref['tokens']
        tokens = " ".join(ori_tokens)
        is_nr3d = ref['dataset'] == 'nr3d'

        LLM_info = dict()
        LLM_info['target_object'] = ref['target_object']
        LLM_info['anchor_objects'] = ast.literal_eval(ref['anchor_objects'])
        LLM_info['referential_order'] = ast.literal_eval(ref['referential_order'])
        LLM_info['referential_order'] = [word.strip('*') for word in LLM_info['referential_order']]

        return scan, target, tokens, is_nr3d, scan_id, LLM_info

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors
    
    def prepare_distractors_ours(self, scan, target, scan_id):
        target_label = target.instance_label
        already_included = {target_label}
        distractors = []
        pred = []
        for i, o in enumerate(scan.three_d_objects):
            if o.instance_label == target_label and (o != target):
                distractors.append(o)
                pred.append(self.cls_results[scan_id][i])
        clutter = []
        pred2 = []
        for i, o in enumerate(scan.three_d_objects):
            if o.instance_label not in already_included:
                clutter.append(o)
                pred2.append(self.cls_results[scan_id][i])
        
        temp = list(zip(clutter, pred2))
        np.random.shuffle(temp)
        clutter, pred2 = zip(*temp)
        clutter, pred2 = list(clutter), list(pred2)

        distractors.extend(clutter)
        pred.extend(pred2)

        distractors = distractors[:self.max_distractors]
        pred = pred[:self.max_distractors]

        temp = list(zip(distractors, pred))
        np.random.shuffle(temp)
        distractors, pred = zip(*temp)
        distractors, pred = list(distractors), list(pred)   

        target_pred = self.cls_results[scan_id][target.object_id]     

        return distractors, pred, target_pred

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, is_nr3d, scan_id, LLM_info = self.get_reference_data(index)

        # Make a context of distractors
        context, pred_box, target_pred = self.prepare_distractors_ours(scan, target, scan_id)
        if target_pred == -1:
            target_pred = 325 # 325 for butd id

        pred_box = [i if i != -1 else 325 for i in pred_box]

        # Add target object into list of context
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)
        pred_box.insert(target_pos, target_pred)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])
        # mark their classes
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)
        res['scan_id'] = scan_id
        box_info = np.zeros((self.max_context_size, 4))
        box_info[:len(context),0] = [o.get_bbox().cx for o in context]
        box_info[:len(context),1] = [o.get_bbox().cy for o in context]
        box_info[:len(context),2] = [o.get_bbox().cz for o in context]
        box_info[:len(context),3] = [o.get_bbox().volume() for o in context]
        box_corners = np.zeros((self.max_context_size, 8, 3))
        box_corners[:len(context)] = [o.get_bbox().corners for o in context]
        box_info_center = np.zeros((self.max_context_size, 3))
        box_info_center[:len(context)] = [o.get_bbox().center() for o in context]
        res['objects'] = pad_samples(samples, self.max_context_size)
        res['center_coors'] = box_info_center
        res['corner_coors'] = box_corners
        if self.object_transformation is not None:
            samples = self.object_transformation(samples)
        # get object mask
        obj_existance = np.zeros((self.max_context_size, 1))
        obj_existance[:len(context),0] = 1
        res['obj_mask'] = obj_existance
        res['context_size'] = len(samples)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        pred_class_labels = [self.scannetconfig_butd.class2type[pred_box[i]] for i in range(len(context))]

        order = LLM_info['referential_order']
        res['ori_order_len'] = len(order)
        
        # pad order
        if order == []:
            tmp = list(set(self.scannetconfig_butd.type2class.keys()).difference(set([o.instance_label for o in context])))
            order = sample(tmp, 1) # this will lead to a all-zero mask
        while len(order) > self.order_len:
            del order[0]
        if self.order_len == 5:
            if len(order) == 1:
                order *= self.order_len
            elif len(order) == 2:
                order = [order[0]]*2 + [order[1]]*3
            elif len(order) == 3:
                order = [order[0]]*1 + [order[1]]*1 + [order[2]]*3
            elif len(order) == 4:
                order.append(order[-1])
        if self.order_len == 6:
            if len(order) == 1:
                order *= self.order_len
            elif len(order) == 2:
                order = [order[0]]*3 + [order[1]]*3
            elif len(order) == 3:
                order = [order[0]]*2 + [order[1]]*2 + [order[2]]*2
            elif len(order) == 4:
                order = [order[0]]*1 + [order[1]]*1 + [order[2]]*1 + [order[3]]*3
            elif len(order) == 5:
                order.append(order[-1])
        if self.order_len == 4:
            if len(order) == 1:
                order *= self.order_len
            elif len(order) == 2:
                order = [order[0]]*2 + [order[1]]*2
            elif len(order) == 3:
                order.append(order[-1])
        elif self.order_len == 3:
            if len(order) == 1:
                order *= self.order_len
            elif len(order) == 2:
                order = [order[0]]*1 + [order[1]]*2
        elif self.order_len == 2:
            if len(order) == 1:
                order *= self.order_len
        elif self.order_len == 1:
            pass

        res['order_labels'] = np.array([self.scannetconfig_butd.type2class[i] for i in order])

        if self.multilabel_pretraining:
            ordered_multilabel_gt = []
            for i, obj in enumerate(order):
                mask = np.zeros(self.max_context_size, dtype=np.bool)
                if not self.cascading:
                    mask[:len(context)] = [obj == o.instance_label for o in context]
                else:
                    mask[:len(context)] = [o.instance_label in order[i:] for o in context]
                ordered_multilabel_gt.append(mask)
            ordered_multilabel_gt = np.stack(ordered_multilabel_gt, axis=0).astype(int)
            res['ordered_multilabel_gt'] = ordered_multilabel_gt

        pred_class_mask = []
        for i, obj in enumerate(order):
            mask = np.zeros(self.max_context_size, dtype=np.bool)
            if not self.cascading:
                all_obj = set([obj])
            else:
                all_obj = set(order[i:])
            mask[:len(context)] = [pred_class_labels[k] in all_obj for k in range(len(context))]

            pred_class_mask.append(mask)
        pred_class_mask = np.stack(pred_class_mask, axis=0)
 
        if self.lang_multilabel:
            anchor_ind = np.zeros(485) # ignore the pad class (525)
            anchor_order = set(order)
            anchor_order.discard(order[-1])
            for i in anchor_order:
                anchor_ind[self.scannetconfig_butd.type2class[i]] = 1
            res['anchor_ind'] = anchor_ind

        res['target_object'] = LLM_info['target_object']
        res['pred_class_mask'] = pred_class_mask

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

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d
        res['box_info'] = box_info
        res['box_corners'] = box_corners

        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)

    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   lang_multilabel=args.lang_multilabel,
                                   multilabel_pretraining=args.multilabel_pretraining,
                                   cascading=args.cascading,
                                   order_len=args.order_len)

        seed = None
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, seed=seed)

    return data_loaders
