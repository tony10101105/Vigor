#!/usr/bin/env python
# coding: utf-8

import torch
import tqdm
import time
import warnings
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored

from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented_pre import load_scan_related_data
from referit3d.in_out.neural_net_oriented_pre import compute_auxiliary_data
from referit3d.in_out.pt_datasets.listening_dataset_pre import make_data_loaders
from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.models.referit3d_net_pre import ReferIt3DNet_transformer
from referit3d.models.referit3d_net_utils_pre import single_epoch_train, evaluate_on_dataset
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions
from transformers import BertTokenizer

def log_train_test_information():
        """Helper logging function.
        Note uses "global" variables defined below.
        """
        logger.info('Epoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            if phase == 'train':
                meters = train_meters
            else:
                meters = test_meters

            info = '{}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(phase,
                                                                        meters[phase + '_total_loss'],
                                                                        meters[phase + '_referential_acc'])

            if args.obj_cls_alpha > 0:
                info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])

            if args.lang_cls_alpha > 0:
                info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_txt_cls_acc'])

            logger.info(info)
            logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
        logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))


if __name__ == '__main__':
    
    # Parse arguments
    args = parse_arguments()
    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    class_to_idx = {'air hockey table': 0, 'airplane': 1, 'alarm': 2, 'alarm clock': 3, 'armchair': 4, 'baby mobile': 5, 'backpack': 6, 'bag': 7, 'bag of coffee beans': 8, 'ball': 9, 'banana holder': 10, 'bananas': 11, 'banister': 12, 'banner': 13, 'bar': 14, 'barricade': 15, 'basket': 16, 'bath products': 17, 'bath walls': 18, 'bathrobe': 19, 'bathroom cabinet': 20, 'bathroom counter': 21, 'bathroom stall': 22, 'bathroom stall door': 23, 'bathroom vanity': 24, 'bathtub': 25, 'battery disposal jar': 26, 'beachball': 27, 'beanbag chair': 28, 'bear': 29, 'bed': 30, 'beer bottles': 31, 'bench': 32, 'bicycle': 33, 'bike lock': 34, 'bike pump': 35, 'bin': 36, 'blackboard': 37, 'blanket': 38, 'blinds': 39, 'block': 40, 'board': 41, 'boards': 42, 'boat': 43, 'boiler': 44, 'book': 45, 'book rack': 46, 'books': 47, 'bookshelf': 48, 'bookshelves': 49, 'boots': 50, 'bottle': 51, 'bowl': 52, 'box': 53, 'boxes': 54, 'boxes of paper': 55, 'breakfast bar': 56, 'briefcase': 57, 'broom': 58, 'bucket': 59, 'bulletin board': 60, 'bunk bed': 61, 'cabinet': 62, 'cabinet door': 63, 'cabinet doors': 64, 'cabinets': 65, 'cable': 66, 'calendar': 67, 'camera': 68, 'can': 69, 'candle': 70, 'canopy': 71, 'car': 72, 'card': 73, 'cardboard': 74, 'carpet': 75, 'carseat': 76, 'cart': 77, 'carton': 78, 'case': 79, 'case of water bottles': 80, 'cat litter box': 81, 'cd case': 82, 'ceiling': 83, 'ceiling fan': 84, 'ceiling light': 85, 'chain': 86, 'chair': 87, 'chandelier': 88, 'changing station': 89, 'chest': 90, 'clock': 91, 'closet': 92, 'closet ceiling': 93, 'closet door': 94, 'closet doorframe': 95, 'closet doors': 96, 'closet floor': 97, 'closet rod': 98, 'closet shelf': 99, 'closet wall': 100, 'closet walls': 101, 'cloth': 102, 'clothes': 103, 'clothes dryer': 104, 'clothes dryers': 105, 'clothes hanger': 106, 'clothes hangers': 107, 'clothing': 108, 'clothing rack': 109, 'coat': 110, 'coat rack': 111, 'coatrack': 112, 'coffee box': 113, 'coffee kettle': 114, 'coffee maker': 115, 'coffee table': 116, 'column': 117, 'compost bin': 118, 'computer tower': 119, 'conditioner bottle': 120, 'container': 121, 'controller': 122, 'cooking pan': 123, 'cooking pot': 124, 'copier': 125, 'costume': 126, 'couch': 127, 'couch cushions': 128, 'counter': 129, 'covered box': 130, 'crate': 131, 'crib': 132, 'cup': 133, 'cups': 134, 'curtain': 135, 'curtains': 136, 'cushion': 137, 'cutting board': 138, 'dart board': 139, 'decoration': 140, 'desk': 141, 'desk lamp': 142, 'diaper bin': 143, 'dining table': 144, 'dish rack': 145, 'dishwasher': 146, 'dishwashing soap bottle': 147, 'dispenser': 148, 'display': 149, 'display case': 150, 'display rack': 151, 'divider': 152, 'doll': 153, 'dollhouse': 154, 'dolly': 155, 'door': 156, 'doorframe': 157, 'doors': 158, 'drawer': 159, 'dress rack': 160, 'dresser': 161, 'drum set': 162, 'dryer sheets': 163, 'drying rack': 164, 'duffel bag': 165, 'dumbbell': 166, 'dustpan': 167, 'easel': 168, 'electric panel': 169, 'elevator': 170, 'elevator button': 171, 'elliptical machine': 172, 'end table': 173, 'envelope': 174, 'exercise bike': 175, 'exercise machine': 176, 'exit sign': 177, 'fan': 178, 'faucet': 179, 'file cabinet': 180, 'fire alarm': 181, 'fire extinguisher': 182, 'fireplace': 183, 'flag': 184, 'flip flops': 185, 'floor': 186, 'flower stand': 187, 'flowerpot': 188, 'folded chair': 189, 'folded chairs': 190, 'folded ladder': 191, 'folded table': 192, 'folder': 193, 'food bag': 194, 'food container': 195, 'food display': 196, 'foosball table': 197, 'footrest': 198, 'footstool': 199, 'frame': 200, 'frying pan': 201, 'furnace': 202, 'furniture': 203, 'fuse box': 204, 'futon': 205, 'garage door': 206, 'garbage bag': 207, 'glass doors': 208, 'globe': 209, 'golf bag': 210, 'grab bar': 211, 'grocery bag': 212, 'guitar': 213, 'guitar case': 214, 'hair brush': 215, 'hair dryer': 216, 'hamper': 217, 'hand dryer': 218, 'hand rail': 219, 'hand sanitzer dispenser': 220, 'hand towel': 221, 'handicap bar': 222, 'handrail': 223, 'hanging': 224, 'hat': 225, 'hatrack': 226, 'headboard': 227, 'headphones': 228, 'heater': 229, 'helmet': 230, 'hose': 231, 'hoverboard': 232, 'humidifier': 233, 'ikea bag': 234, 'instrument case': 235, 'ipad': 236, 'iron': 237, 'ironing board': 238, 'jacket': 239, 'jar': 240, 'kettle': 241, 'keyboard': 242, 'keyboard piano': 243, 'kitchen apron': 244, 'kitchen cabinet': 245, 'kitchen cabinets': 246, 'kitchen counter': 247, 'kitchen island': 248, 'kitchenaid mixer': 249, 'knife block': 250, 'ladder': 251, 'lamp': 252, 'lamp base': 253, 'laptop': 254, 'laundry bag': 255, 'laundry basket': 256, 'laundry detergent': 257, 'laundry hamper': 258, 'ledge': 259, 'legs': 260, 'light': 261, 'light switch': 262, 'loft bed': 263, 'loofa': 264, 'luggage': 265, 'luggage rack': 266, 'luggage stand': 267, 'lunch box': 268, 'machine': 269, 'magazine': 270, 'magazine rack': 271, 'mail': 272, 'mail tray': 273, 'mailbox': 274, 'mailboxes': 275, 'map': 276, 'massage chair': 277, 'mat': 278, 'mattress': 279, 'medal': 280, 'messenger bag': 281, 'metronome': 282, 'microwave': 283, 'mini fridge': 284, 'mirror': 285, 'mirror doors': 286, 'monitor': 287, 'mouse': 288, 'mouthwash bottle': 289, 'mug': 290, 'music book': 291, 'music stand': 292, 'nerf gun': 293, 'night lamp': 294, 'nightstand': 295, 'notepad': 296, 'object': 297, 'office chair': 298, 'open kitchen cabinet': 299, 'organizer': 300, 'organizer shelf': 301, 'ottoman': 302, 'oven': 303, 'oven mitt': 304, 'painting': 305, 'pantry shelf': 306, 'pantry wall': 307, 'pantry walls': 308, 'pants': 309, 'paper': 310, 'paper bag': 311, 'paper cutter': 312, 'paper organizer': 313, 'paper towel': 314, 'paper towel dispenser': 315, 'paper towel roll': 316, 'paper tray': 317, 'papers': 318, 'person': 319, 'photo': 320, 'piano': 321, 'piano bench': 322, 'picture': 323, 'pictures': 324, 'pillar': 325, 'pillow': 326, 'pillows': 327, 'ping pong table': 328, 'pipe': 329, 'pipes': 330, 'pitcher': 331, 'pizza boxes': 332, 'plant': 333, 'plastic bin': 334, 'plastic container': 335, 'plastic containers': 336, 'plastic storage bin': 337, 'plate': 338, 'plates': 339, 'plunger': 340, 'podium': 341, 'pool table': 342, 'poster': 343, 'poster cutter': 344, 'poster printer': 345, 'poster tube': 346, 'pot': 347, 'potted plant': 348, 'power outlet': 349, 'power strip': 350, 'printer': 351, 'projector': 352, 'projector screen': 353, 'purse': 354, 'quadcopter': 355, 'rack': 356, 'rack stand': 357, 'radiator': 358, 'rail': 359, 'railing': 360, 'range hood': 361, 'recliner chair': 362, 'recycling bin': 363, 'refrigerator': 364, 'remote': 365, 'rice cooker': 366, 'rod': 367, 'rolled poster': 368, 'roomba': 369, 'rope': 370, 'round table': 371, 'rug': 372, 'salt': 373, 'santa': 374, 'scale': 375, 'scanner': 376, 'screen': 377, 'seat': 378, 'seating': 379, 'sewing machine': 380, 'shampoo': 381, 'shampoo bottle': 382, 'shelf': 383, 'shirt': 384, 'shoe': 385, 'shoe rack': 386, 'shoes': 387, 'shopping bag': 388, 'shorts': 389, 'shower': 390, 'shower control valve': 391, 'shower curtain': 392, 'shower curtain rod': 393, 'shower door': 394, 'shower doors': 395, 'shower floor': 396, 'shower head': 397, 'shower wall': 398, 'shower walls': 399, 'shredder': 400, 'sign': 401, 'sink': 402, 'sliding wood door': 403, 'slippers': 404, 'smoke detector': 405, 'soap': 406, 'soap bottle': 407, 'soap dish': 408, 'soap dispenser': 409, 'sock': 410, 'soda stream': 411, 'sofa bed': 412, 'sofa chair': 413, 'speaker': 414, 'sponge': 415, 'spray bottle': 416, 'stack of chairs': 417, 'stack of cups': 418, 'stack of folded chairs': 419, 'stair': 420, 'stair rail': 421, 'staircase': 422, 'stairs': 423, 'stand': 424, 'stapler': 425, 'starbucks cup': 426, 'statue': 427, 'step': 428, 'step stool': 429, 'sticker': 430, 'stool': 431, 'storage bin': 432, 'storage box': 433, 'storage container': 434, 'storage organizer': 435, 'storage shelf': 436, 'stove': 437, 'structure': 438, 'studio light': 439, 'stuffed animal': 440, 'suitcase': 441, 'suitcases': 442, 'sweater': 443, 'swiffer': 444, 'switch': 445, 'table': 446, 'tank': 447, 'tap': 448, 'tape': 449, 'tea kettle': 450, 'teapot': 451, 'teddy bear': 452, 'telephone': 453, 'telescope': 454, 'thermostat': 455, 'tire': 456, 'tissue box': 457, 'toaster': 458, 'toaster oven': 459, 'toilet': 460, 'toilet brush': 461, 'toilet flush button': 462, 'toilet paper': 463, 'toilet paper dispenser': 464, 'toilet paper holder': 465, 'toilet paper package': 466, 'toilet paper rolls': 467, 'toilet seat cover dispenser': 468, 'toiletry': 469, 'toolbox': 470, 'toothbrush': 471, 'toothpaste': 472, 'towel': 473, 'towel rack': 474, 'towels': 475, 'toy dinosaur': 476, 'toy piano': 477, 'traffic cone': 478, 'trash bag': 479, 'trash bin': 480, 'trash cabinet': 481, 'trash can': 482, 'tray': 483, 'tray rack': 484, 'treadmill': 485, 'tripod': 486, 'trolley': 487, 'trunk': 488, 'tube': 489, 'tupperware': 490, 'tv': 491, 'tv stand': 492, 'umbrella': 493, 'urinal': 494, 'vacuum cleaner': 495, 'vase': 496, 'vending machine': 497, 'vent': 498, 'wall': 499, 'wall hanging': 500, 'wall lamp': 501, 'wall mounted coat rack': 502, 'wardrobe': 503, 'wardrobe cabinet': 504, 'wardrobe closet': 505, 'washcloth': 506, 'washing machine': 507, 'washing machines': 508, 'water bottle': 509, 'water cooler': 510, 'water fountain': 511, 'water heater': 512, 'water pitcher': 513, 'wet floor sign': 514, 'wheel': 515, 'whiteboard': 516, 'whiteboard eraser': 517, 'window': 518, 'windowsill': 519, 'wood': 520, 'wood beam': 521, 'workbench': 522, 'yoga mat': 523, 'pad': 524}
    mean_rgb = compute_auxiliary_data(scans_split, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, scans_split, class_to_idx, all_scans_in_dict, mean_rgb)
    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0"

    device = torch.device('cuda')
    seed_training_code(args.random_seed)

    # Losses:
    criteria = dict()
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']
    # Object-type classification
    class_name_list = []
    for cate in class_to_idx:
        class_name_list.append(cate)

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    class_name_tokens = tokenizer(class_name_list, return_tensors='pt', padding=True)
    for name in class_name_tokens.data:
        class_name_tokens.data[name] = class_name_tokens.data[name].cuda()

    gpu_num = len(args.gpu.strip(',').split(','))

    if args.model == 'referIt3DNet_transformer':
        model = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)
    else:
        assert False

    if gpu_num > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    # <1>
    if gpu_num > 1:
        param_list=[
            {'params':model.module.language_encoder.parameters(),'lr':args.init_lr*0.1},
            {'params':model.module.object_encoder.parameters(), 'lr':args.init_lr},
            {'params':model.module.obj_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.module.box_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.module.language_clf.parameters(), 'lr': args.init_lr},
            {'params':model.module.object_language_clf.parameters(), 'lr': args.init_lr},
            {'params':model.module.refer_encoder.parameters(), 'lr':args.init_lr*0.1},
        ]
        if args.multilabel_pretraining:
            param_list.append({'params':model.module.feat_to_multilabel_clf.parameters(), 'lr': args.init_lr})
            param_list.append({'params':model.module.feat_to_coor_reg.parameters(), 'lr': args.init_lr})
        if not args.label_lang_sup:
            param_list.append({'params':model.module.obj_clf.parameters(), 'lr': args.init_lr})
    else:
        param_list=[
            {'params':model.language_encoder.parameters(),'lr':args.init_lr*0.1},
            {'params':model.object_encoder.parameters(), 'lr':args.init_lr},
            {'params':model.obj_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.box_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.language_clf.parameters(), 'lr': args.init_lr},
            {'params':model.object_language_clf.parameters(), 'lr': args.init_lr},
            {'params':model.refer_encoder.parameters(), 'lr':args.init_lr*0.1},
        ]
        if args.multilabel_pretraining:
            param_list.append({'params':model.feat_to_multilabel_clf.parameters(), 'lr': args.init_lr})
            param_list.append({'params':model.feat_to_coor_reg.parameters(), 'lr': args.init_lr})
        if not args.label_lang_sup:
            param_list.append({'params':model.obj_clf.parameters(), 'lr': args.init_lr})

    optimizer = optim.Adam(param_list,lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[100, 130, 160, 190, 220, 250], gamma=0.65)
    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    last_test_acc = -1
    last_test_epoch = -1

    if args.resume_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=model)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if not args.fine_tune:
            print('Loaded a model that we do NOT plan to fine-tune.')
            load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            start_training_epoch = loaded_epoch + 1
            start_training_epoch = 0
            best_test_epoch = loaded_epoch
            best_test_acc = 0
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                best_test_acc))
            torch.save({
            'object_encoder': model.object_encoder.state_dict(),
            'obj_feature_mapping': model.obj_feature_mapping.state_dict()
            }, '/work/b08901133/tony/MVT-3DVG/pc.pth')
            print('done')
            exit(0)
        else:
            print('Parameters that do not allow gradients to be back-propped:')
            ft_everything = True
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print('None, all wil be fine-tuned')
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = args.max_train_epochs + 1 - start_training_epoch
            print('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))

    # Training
    if args.mode == 'train':
        logger = create_logger(args.log_dir)
        logger.info('Starting the training. Good luck!')

        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc='epochs') as bar:
            timings = dict()
            for epoch in bar:
                print("cnt_lr", lr_scheduler.get_last_lr())
                # Train:
                tic = time.time()
                train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                  device, pad_idx, args=args, tokenizer=tokenizer,epoch=epoch)
                toc = time.time()
                timings['train'] = (toc - tic) / 60

                # Evaluate:
                tic = time.time()
                test_meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args, tokenizer=tokenizer)
                toc = time.time()
                timings['test'] = (toc - tic) / 60

                eval_acc = test_meters['test_referential_acc']

                last_test_acc = eval_acc
                last_test_epoch = epoch

                lr_scheduler.step()

                save_state_dicts(osp.join(args.checkpoint_dir, 'last_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

                if best_test_acc < eval_acc:
                    logger.info(colored('Test accuracy, improved @epoch {}'.format(epoch), 'green'))
                    best_test_acc = eval_acc
                    best_test_epoch = epoch

                    save_state_dicts(osp.join(args.checkpoint_dir, 'best_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
                else:
                    logger.info(colored('Test accuracy, did not improve @epoch {}'.format(epoch), 'red'))
                
                if epoch % 100 == 0:
                    save_state_dicts(osp.join(args.checkpoint_dir, '{}_model.pth'.format(epoch)),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)                    

                log_train_test_information()
                train_meters.update(test_meters)

                bar.refresh()

        with open(osp.join(args.checkpoint_dir, 'final_result.txt'), 'w') as f_out:
            f_out.write(('Best accuracy: {:.4f} (@epoch {})'.format(best_test_acc, best_test_epoch)))
            f_out.write(('Last accuracy: {:.4f} (@epoch {})'.format(last_test_acc, last_test_epoch)))

        logger.info('Finished training successfully.')

    elif args.mode == 'evaluate':

        meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, args=args, tokenizer=tokenizer)
        print('Reference-Accuracy: {:.4f}'.format(meters['test_referential_acc']))
        print('Object-Clf-Accuracy: {:.4f}'.format(meters['test_object_cls_acc']))
        print('Text-Clf-Accuracy {:.4f}:'.format(meters['test_txt_cls_acc']))
        
        out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
        res = analyze_predictions(model, data_loaders['test'].dataset, class_to_idx, pad_idx, device,
                                  args, out_file=out_file,tokenizer=tokenizer)
        print(res)