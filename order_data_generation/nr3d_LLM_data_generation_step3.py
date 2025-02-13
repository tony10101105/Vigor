'''
This script conducts basic rule-based data cleaning and 
projects predicted class names to a predefined class name pool
'''


from tqdm import tqdm
from itertools import groupby
import numpy as np
import pandas as pd
import torch
import math
import clip
from copy import deepcopy
pd.options.mode.chained_assignment = None


### parameters
in_csv = 'nr3d_train_LLM_step2_485.csv'
out_csv = 'nr3d_train_LLM_step3_485.csv'
class_pool = 'butd'
###


def embed_matrix(samples, concepts):
    dot_prods_c = _clip_dot_prods(samples, concepts)
    return dot_prods_c

def _clip_dot_prods(list1, list2, device = 'cuda', batch_size = 500):
    'Returns: numpy array with dot products'
    text1 = clip.tokenize(list1).to(device)
    text2 = clip.tokenize(list2).to(device)
    features1 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text1)/batch_size)):
            features1.append(clip_model.encode_text(text1[batch_size*i:batch_size*(i+1)]))
        features1 = torch.cat(features1, dim=0)
        features1 /= features1.norm(dim=1, keepdim=True)

    features2 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text2)/batch_size)):
            features2.append(clip_model.encode_text(text2[batch_size*i:batch_size*(i+1)]))
        features2 = torch.cat(features2, dim=0)
        features2 /= features2.norm(dim=1, keepdim=True)
        
    dot_prods = features1 @ features2.T
    return dot_prods.cpu().numpy()

def mod_sample(g_target, p_target, utterance, p_anchor, order):
    if g_target != p_target:
        print('gt_target: ', g_target)
        print('pred_target: ', p_target)

    filtered_pred_anchor = []
    if 'none' in p_anchor or 'nan' in p_anchor or 'mention' in p_anchor or p_anchor == 'nan' or 'n\\a' in p_anchor:
        print('none pred_anchor: ', p_anchor) 
    else:
        p_anchor = p_anchor.split(',')
        for obj in p_anchor:
            obj = obj.strip(' ')
            if obj not in ['left', 'right', 'corner','room', 'corner of the room', 'another', 'other']:
                sim = embed_matrix([obj], all_classes)
                max_id = np.argmax(sim)
                obj = all_classes[max_id]
                filtered_pred_anchor.append(obj)
        filtered_pred_anchor = [i[0] for i in groupby(filtered_pred_anchor)]
        while p_target in filtered_pred_anchor:
            filtered_pred_anchor.remove(p_target)

    filtered_order = []
    if str(order) == 'nan' or 'none' in order or 'nan' in order or 'mention' in order:
        print('none order: ', order)
    else:
        order = order.split('->')
        for obj in order:
            obj = obj.strip(' ')
            if obj not in ['left', 'right', 'corner', 'room', 'corner of the room', 'another', 'other']:
                sim = embed_matrix([obj], all_classes)
                max_id = np.argmax(sim)
                obj = all_classes[max_id]
                filtered_order.append(obj)
        filtered_order = [i[0] for i in groupby(filtered_order)]

    if filtered_order == []:
        if len(filtered_pred_anchor) > 0:
            filtered_order = deepcopy(filtered_pred_anchor)
        else:
            filtered_order.append(p_target)

    if p_target != filtered_order[-1]:
        while p_target in filtered_order:
            filtered_order.remove(p_target)
        filtered_order.append(p_target)

    if filtered_pred_anchor == [] and len(filtered_order) > 1:
        filtered_pred_anchor = filtered_order[:-1]

    for j in filtered_pred_anchor:
        if j not in filtered_order:
            filtered_order.insert(0, j)

    try:
        assert p_target == filtered_order[-1]
        assert set(filtered_pred_anchor).issubset(set(filtered_order[:-1]))
    except:
        print('NOTICE!')
        print('utterance: ', utterance)
        print('pred_target: ', p_target)
        print('filtered_pred_anchor: ', filtered_pred_anchor)
        print('filtered_order: ', filtered_order)
        raise Exception('stop')
    
    return p_target, filtered_pred_anchor, filtered_order


clip_model, _ = clip.load('ViT-B/16', device='cuda')

if class_pool == 'nr3d':
    all_classes = {'air hockey table': 0, 'airplane': 1, 'alarm': 2, 'alarm clock': 3, 'armchair': 4, 'baby mobile': 5, 'backpack': 6, 'bag': 7, 'bag of coffee beans': 8, 'ball': 9, 'banana holder': 10, 'bananas': 11, 'banister': 12, 'banner': 13, 'bar': 14, 'barricade': 15, 'basket': 16, 'bath products': 17, 'bath walls': 18, 'bathrobe': 19, 'bathroom cabinet': 20, 'bathroom counter': 21, 'bathroom stall': 22, 'bathroom stall door': 23, 'bathroom vanity': 24, 'bathtub': 25, 'battery disposal jar': 26, 'beachball': 27, 'beanbag chair': 28, 'bear': 29, 'bed': 30, 'beer bottles': 31, 'bench': 32, 'bicycle': 33, 'bike lock': 34, 'bike pump': 35, 'bin': 36, 'blackboard': 37, 'blanket': 38, 'blinds': 39, 'block': 40, 'board': 41, 'boards': 42, 'boat': 43, 'boiler': 44, 'book': 45, 'book rack': 46, 'books': 47, 'bookshelf': 48, 'bookshelves': 49, 'boots': 50, 'bottle': 51, 'bowl': 52, 'box': 53, 'boxes': 54, 'boxes of paper': 55, 'breakfast bar': 56, 'briefcase': 57, 'broom': 58, 'bucket': 59, 'bulletin board': 60, 'bunk bed': 61, 'cabinet': 62, 'cabinet door': 63, 'cabinet doors': 64, 'cabinets': 65, 'cable': 66, 'calendar': 67, 'camera': 68, 'can': 69, 'candle': 70, 'canopy': 71, 'car': 72, 'card': 73, 'cardboard': 74, 'carpet': 75, 'carseat': 76, 'cart': 77, 'carton': 78, 'case': 79, 'case of water bottles': 80, 'cat litter box': 81, 'cd case': 82, 'ceiling': 83, 'ceiling fan': 84, 'ceiling light': 85, 'chain': 86, 'chair': 87, 'chandelier': 88, 'changing station': 89, 'chest': 90, 'clock': 91, 'closet': 92, 'closet ceiling': 93, 'closet door': 94, 'closet doorframe': 95, 'closet doors': 96, 'closet floor': 97, 'closet rod': 98, 'closet shelf': 99, 'closet wall': 100, 'closet walls': 101, 'cloth': 102, 'clothes': 103, 'clothes dryer': 104, 'clothes dryers': 105, 'clothes hanger': 106, 'clothes hangers': 107, 'clothing': 108, 'clothing rack': 109, 'coat': 110, 'coat rack': 111, 'coatrack': 112, 'coffee box': 113, 'coffee kettle': 114, 'coffee maker': 115, 'coffee table': 116, 'column': 117, 'compost bin': 118, 'computer tower': 119, 'conditioner bottle': 120, 'container': 121, 'controller': 122, 'cooking pan': 123, 'cooking pot': 124, 'copier': 125, 'costume': 126, 'couch': 127, 'couch cushions': 128, 'counter': 129, 'covered box': 130, 'crate': 131, 'crib': 132, 'cup': 133, 'cups': 134, 'curtain': 135, 'curtains': 136, 'cushion': 137, 'cutting board': 138, 'dart board': 139, 'decoration': 140, 'desk': 141, 'desk lamp': 142, 'diaper bin': 143, 'dining table': 144, 'dish rack': 145, 'dishwasher': 146, 'dishwashing soap bottle': 147, 'dispenser': 148, 'display': 149, 'display case': 150, 'display rack': 151, 'divider': 152, 'doll': 153, 'dollhouse': 154, 'dolly': 155, 'door': 156, 'doorframe': 157, 'doors': 158, 'drawer': 159, 'dress rack': 160, 'dresser': 161, 'drum set': 162, 'dryer sheets': 163, 'drying rack': 164, 'duffel bag': 165, 'dumbbell': 166, 'dustpan': 167, 'easel': 168, 'electric panel': 169, 'elevator': 170, 'elevator button': 171, 'elliptical machine': 172, 'end table': 173, 'envelope': 174, 'exercise bike': 175, 'exercise machine': 176, 'exit sign': 177, 'fan': 178, 'faucet': 179, 'file cabinet': 180, 'fire alarm': 181, 'fire extinguisher': 182, 'fireplace': 183, 'flag': 184, 'flip flops': 185, 'floor': 186, 'flower stand': 187, 'flowerpot': 188, 'folded chair': 189, 'folded chairs': 190, 'folded ladder': 191, 'folded table': 192, 'folder': 193, 'food bag': 194, 'food container': 195, 'food display': 196, 'foosball table': 197, 'footrest': 198, 'footstool': 199, 'frame': 200, 'frying pan': 201, 'furnace': 202, 'furniture': 203, 'fuse box': 204, 'futon': 205, 'garage door': 206, 'garbage bag': 207, 'glass doors': 208, 'globe': 209, 'golf bag': 210, 'grab bar': 211, 'grocery bag': 212, 'guitar': 213, 'guitar case': 214, 'hair brush': 215, 'hair dryer': 216, 'hamper': 217, 'hand dryer': 218, 'hand rail': 219, 'hand sanitzer dispenser': 220, 'hand towel': 221, 'handicap bar': 222, 'handrail': 223, 'hanging': 224, 'hat': 225, 'hatrack': 226, 'headboard': 227, 'headphones': 228, 'heater': 229, 'helmet': 230, 'hose': 231, 'hoverboard': 232, 'humidifier': 233, 'ikea bag': 234, 'instrument case': 235, 'ipad': 236, 'iron': 237, 'ironing board': 238, 'jacket': 239, 'jar': 240, 'kettle': 241, 'keyboard': 242, 'keyboard piano': 243, 'kitchen apron': 244, 'kitchen cabinet': 245, 'kitchen cabinets': 246, 'kitchen counter': 247, 'kitchen island': 248, 'kitchenaid mixer': 249, 'knife block': 250, 'ladder': 251, 'lamp': 252, 'lamp base': 253, 'laptop': 254, 'laundry bag': 255, 'laundry basket': 256, 'laundry detergent': 257, 'laundry hamper': 258, 'ledge': 259, 'legs': 260, 'light': 261, 'light switch': 262, 'loft bed': 263, 'loofa': 264, 'luggage': 265, 'luggage rack': 266, 'luggage stand': 267, 'lunch box': 268, 'machine': 269, 'magazine': 270, 'magazine rack': 271, 'mail': 272, 'mail tray': 273, 'mailbox': 274, 'mailboxes': 275, 'map': 276, 'massage chair': 277, 'mat': 278, 'mattress': 279, 'medal': 280, 'messenger bag': 281, 'metronome': 282, 'microwave': 283, 'mini fridge': 284, 'mirror': 285, 'mirror doors': 286, 'monitor': 287, 'mouse': 288, 'mouthwash bottle': 289, 'mug': 290, 'music book': 291, 'music stand': 292, 'nerf gun': 293, 'night lamp': 294, 'nightstand': 295, 'notepad': 296, 'object': 297, 'office chair': 298, 'open kitchen cabinet': 299, 'organizer': 300, 'organizer shelf': 301, 'ottoman': 302, 'oven': 303, 'oven mitt': 304, 'painting': 305, 'pantry shelf': 306, 'pantry wall': 307, 'pantry walls': 308, 'pants': 309, 'paper': 310, 'paper bag': 311, 'paper cutter': 312, 'paper organizer': 313, 'paper towel': 314, 'paper towel dispenser': 315, 'paper towel roll': 316, 'paper tray': 317, 'papers': 318, 'person': 319, 'photo': 320, 'piano': 321, 'piano bench': 322, 'picture': 323, 'pictures': 324, 'pillar': 325, 'pillow': 326, 'pillows': 327, 'ping pong table': 328, 'pipe': 329, 'pipes': 330, 'pitcher': 331, 'pizza boxes': 332, 'plant': 333, 'plastic bin': 334, 'plastic container': 335, 'plastic containers': 336, 'plastic storage bin': 337, 'plate': 338, 'plates': 339, 'plunger': 340, 'podium': 341, 'pool table': 342, 'poster': 343, 'poster cutter': 344, 'poster printer': 345, 'poster tube': 346, 'pot': 347, 'potted plant': 348, 'power outlet': 349, 'power strip': 350, 'printer': 351, 'projector': 352, 'projector screen': 353, 'purse': 354, 'quadcopter': 355, 'rack': 356, 'rack stand': 357, 'radiator': 358, 'rail': 359, 'railing': 360, 'range hood': 361, 'recliner chair': 362, 'recycling bin': 363, 'refrigerator': 364, 'remote': 365, 'rice cooker': 366, 'rod': 367, 'rolled poster': 368, 'roomba': 369, 'rope': 370, 'round table': 371, 'rug': 372, 'salt': 373, 'santa': 374, 'scale': 375, 'scanner': 376, 'screen': 377, 'seat': 378, 'seating': 379, 'sewing machine': 380, 'shampoo': 381, 'shampoo bottle': 382, 'shelf': 383, 'shirt': 384, 'shoe': 385, 'shoe rack': 386, 'shoes': 387, 'shopping bag': 388, 'shorts': 389, 'shower': 390, 'shower control valve': 391, 'shower curtain': 392, 'shower curtain rod': 393, 'shower door': 394, 'shower doors': 395, 'shower floor': 396, 'shower head': 397, 'shower wall': 398, 'shower walls': 399, 'shredder': 400, 'sign': 401, 'sink': 402, 'sliding wood door': 403, 'slippers': 404, 'smoke detector': 405, 'soap': 406, 'soap bottle': 407, 'soap dish': 408, 'soap dispenser': 409, 'sock': 410, 'soda stream': 411, 'sofa bed': 412, 'sofa chair': 413, 'speaker': 414, 'sponge': 415, 'spray bottle': 416, 'stack of chairs': 417, 'stack of cups': 418, 'stack of folded chairs': 419, 'stair': 420, 'stair rail': 421, 'staircase': 422, 'stairs': 423, 'stand': 424, 'stapler': 425, 'starbucks cup': 426, 'statue': 427, 'step': 428, 'step stool': 429, 'sticker': 430, 'stool': 431, 'storage bin': 432, 'storage box': 433, 'storage container': 434, 'storage organizer': 435, 'storage shelf': 436, 'stove': 437, 'structure': 438, 'studio light': 439, 'stuffed animal': 440, 'suitcase': 441, 'suitcases': 442, 'sweater': 443, 'swiffer': 444, 'switch': 445, 'table': 446, 'tank': 447, 'tap': 448, 'tape': 449, 'tea kettle': 450, 'teapot': 451, 'teddy bear': 452, 'telephone': 453, 'telescope': 454, 'thermostat': 455, 'tire': 456, 'tissue box': 457, 'toaster': 458, 'toaster oven': 459, 'toilet': 460, 'toilet brush': 461, 'toilet flush button': 462, 'toilet paper': 463, 'toilet paper dispenser': 464, 'toilet paper holder': 465, 'toilet paper package': 466, 'toilet paper rolls': 467, 'toilet seat cover dispenser': 468, 'toiletry': 469, 'toolbox': 470, 'toothbrush': 471, 'toothpaste': 472, 'towel': 473, 'towel rack': 474, 'towels': 475, 'toy dinosaur': 476, 'toy piano': 477, 'traffic cone': 478, 'trash bag': 479, 'trash bin': 480, 'trash cabinet': 481, 'trash can': 482, 'tray': 483, 'tray rack': 484, 'treadmill': 485, 'tripod': 486, 'trolley': 487, 'trunk': 488, 'tube': 489, 'tupperware': 490, 'tv': 491, 'tv stand': 492, 'umbrella': 493, 'urinal': 494, 'vacuum cleaner': 495, 'vase': 496, 'vending machine': 497, 'vent': 498, 'wall': 499, 'wall hanging': 500, 'wall lamp': 501, 'wall mounted coat rack': 502, 'wardrobe': 503, 'wardrobe cabinet': 504, 'wardrobe closet': 505, 'washcloth': 506, 'washing machine': 507, 'washing machines': 508, 'water bottle': 509, 'water cooler': 510, 'water fountain': 511, 'water heater': 512, 'water pitcher': 513, 'wet floor sign': 514, 'wheel': 515, 'whiteboard': 516, 'whiteboard eraser': 517, 'window': 518, 'windowsill': 519, 'wood': 520, 'wood beam': 521, 'workbench': 522, 'yoga mat': 523}
    all_classes = np.array(list(all_classes))
elif class_pool == 'sr3d':
    raise Exception('sr3d not supported yet')
elif class_pool == 'butd':
    all_classes = {'wall': 0, 'chair': 1, 'floor': 2, 'table': 3, 'door': 4, 'couch': 5, 'cabinet': 6, 'shelf': 7, 'desk': 8, 'office chair': 9, 'bed': 10, 'pillow': 11, 'sink': 12, 'picture': 13, 'window': 14, 'toilet': 15, 'bookshelf': 16, 'monitor': 17, 'curtain': 18, 'book': 19, 'armchair': 20, 'coffee table': 21, 'drawer': 22, 'box': 23, 'refrigerator': 24, 'lamp': 25, 'kitchen cabinet': 26, 'towel': 27, 'clothes': 28, 'tv': 29, 'nightstand': 30, 'counter': 31, 'dresser': 32, 'stool': 33, 'couch cushions': 34, 'plant': 35, 'ceiling': 36, 'bathtub': 37, 'end table': 38, 'dining table': 39, 'keyboard': 40, 'bag': 41, 'backpack': 42, 'toilet paper': 43, 'printer': 44, 'tv stand': 45, 'whiteboard': 46, 'carpet': 47, 'blanket': 48, 'shower curtain': 49, 'trash can': 50, 'closet': 51, 'staircase': 52, 'microwave': 53, 'rug': 54, 'stove': 55, 'shoe': 56, 'computer tower': 57, 'bottle': 58, 'bin': 59, 'ottoman': 60, 'bench': 61, 'board': 62, 'washing machine': 63, 'mirror': 64, 'copier': 65, 'basket': 66, 'sofa chair': 67, 'file cabinet': 68, 'fan': 69, 'laptop': 70, 'shower': 71, 'paper': 72, 'person': 73, 'headboard': 74, 'paper towel dispenser': 75, 'faucet': 76, 'oven': 77, 'footstool': 78, 'blinds': 79, 'rack': 80, 'plate': 81, 'blackboard': 82, 'piano': 83, 'heater': 84, 'soap': 85, 'suitcase': 86, 'rail': 87, 'radiator': 88, 'recycling bin': 89, 'container': 90, 'wardrobe closet': 91, 'soap dispenser': 92, 'telephone': 93, 'bucket': 94, 'clock': 95, 'stand': 96, 'light': 97, 'laundry basket': 98, 'pipe': 99, 'round table': 100, 'clothes dryer': 101, 'coat': 102, 'guitar': 103, 'toilet paper holder': 104, 'seat': 105, 'step': 106, 'speaker': 107, 'vending machine': 108, 'column': 109, 'bicycle': 110, 'ladder': 111, 'cover': 112, 'bathroom stall': 113, 'foosball table': 114, 'shower wall': 115, 'chest': 116, 'cup': 117, 'jacket': 118, 'storage bin': 119, 'screen': 120, 'coffee maker': 121, 'hamper': 122, 'dishwasher': 123, 'paper towel roll': 124, 'machine': 125, 'mat': 126, 'windowsill': 127, 'tap': 128, 'pool table': 129, 'hand dryer': 130, 'bar': 131, 'frame': 132, 'toaster': 133, 'handrail': 134, 'bulletin board': 135, 'ironing board': 136, 'fireplace': 137, 'soap dish': 138, 'kitchen counter': 139, 'glass': 140, 'doorframe': 141, 'toilet paper dispenser': 142, 'mini fridge': 143, 'fire extinguisher': 144, 'shampoo bottle': 145, 'ball': 146, 'hat': 147, 'shower curtain rod': 148, 'toiletry': 149, 'water cooler': 150, 'desk lamp': 151, 'paper cutter': 152, 'switch': 153, 'tray': 154, 'shower door': 155, 'shirt': 156, 'pillar': 157, 'ledge': 158, 'vase': 159, 'toaster oven': 160, 'mouse': 161, 'nerf gun': 162, 'toilet seat cover dispenser': 163, 'can': 164, 'furniture': 165, 'cart': 166, 'step stool': 167, 'dispenser': 168, 'storage container': 169, 'side table': 170, 'lotion': 171, 'cooking pot': 172, 'toilet brush': 173, 'scale': 174, 'tissue box': 175, 'remote': 176, 'light switch': 177, 'crate': 178, 'ping pong table': 179, 'platform': 180, 'slipper': 181, 'power outlet': 182, 'cutting board': 183, 'controller': 184, 'decoration': 185, 'trolley': 186, 'sign': 187, 'projector': 188, 'sweater': 189, 'globe': 190, 'closet door': 191, 'plastic container': 192, 'statue': 193, 'vacuum cleaner': 194, 'wet floor sign': 195, 'candle': 196, 'easel': 197, 'wall hanging': 198, 'dumbell': 199, 'ping pong paddle': 200, 'plunger': 201, 'soap bar': 202, 'stuffed animal': 203, 'water fountain': 204, 'footrest': 205, 'headphones': 206, 'plastic bin': 207, 'coatrack': 208, 'dish rack': 209, 'broom': 210, 'guitar case': 211, 'mop': 212, 'magazine': 213, 'range hood': 214, 'scanner': 215, 'bathrobe': 216, 'futon': 217, 'dustpan': 218, 'hand towel': 219, 'organizer': 220, 'map': 221, 'helmet': 222, 'hair dryer': 223, 'exercise ball': 224, 'iron': 225, 'studio light': 226, 'cabinet door': 227, 'exercise machine': 228, 'workbench': 229, 'water bottle': 230, 'handicap bar': 231, 'tank': 232, 'purse': 233, 'vent': 234, 'piano bench': 235, 'bunk bed': 236, 'shoe rack': 237, 'shower floor': 238, 'case': 239, 'swiffer': 240, 'stapler': 241, 'cable': 242, 'garbage bag': 243, 'banister': 244, 'trunk': 245, 'tire': 246, 'folder': 247, 'car': 248, 'flower stand': 249, 'water pitcher': 250, 'loft bed': 251, 'shopping bag': 252, 'curtain rod': 253, 'alarm': 254, 'washcloth': 255, 'toolbox': 256, 'sewing machine': 257, 'mailbox': 258, 'toothpaste': 259, 'rope': 260, 'electric panel': 261, 'bowl': 262, 'boiler': 263, 'paper bag': 264, 'alarm clock': 265, 'music stand': 266, 'instrument case': 267, 'paper tray': 268, 'paper shredder': 269, 'projector screen': 270, 'boots': 271, 'kettle': 272, 'mail tray': 273, 'cat litter box': 274, 'covered box': 275, 'ceiling fan': 276, 'cardboard': 277, 'binder': 278, 'beachball': 279, 'envelope': 280, 'thermos': 281, 'breakfast bar': 282, 'dress rack': 283, 'frying pan': 284, 'divider': 285, 'rod': 286, 'magazine rack': 287, 'laundry detergent': 288, 'sofa bed': 289, 'storage shelf': 290, 'loofa': 291, 'bycicle': 292, 'file organizer': 293, 'fire hose': 294, 'media center': 295, 'umbrella': 296, 'barrier': 297, 'subwoofer': 298, 'stepladder': 299, 'shorts': 300, 'rocking chair': 301, 'elliptical machine': 302, 'coffee mug': 303, 'jar': 304, 'door wall': 305, 'traffic cone': 306, 'pants': 307, 'garage door': 308, 'teapot': 309, 'barricade': 310, 'exit sign': 311, 'canopy': 312, 'kinect': 313, 'kitchen island': 314, 'messenger bag': 315, 'buddha': 316, 'block': 317, 'stepstool': 318, 'tripod': 319, 'chandelier': 320, 'smoke detector': 321, 'baseball cap': 322, 'toothbrush': 323, 'bathroom counter': 324, 'object': 325, 'bathroom vanity': 326, 'closet wall': 327, 'laundry hamper': 328, 'bathroom stall door': 329, 'ceiling light': 330, 'trash bin': 331, 'dumbbell': 332, 'stair rail': 333, 'tube': 334, 'bathroom cabinet': 335, 'cd case': 336, 'closet rod': 337, 'coffee kettle': 338, 'wardrobe cabinet': 339, 'structure': 340, 'shower head': 341, 'keyboard piano': 342, 'case of water bottles': 343, 'coat rack': 344, 'storage organizer': 345, 'folded chair': 346, 'fire alarm': 347, 'power strip': 348, 'calendar': 349, 'poster': 350, 'potted plant': 351, 'luggage': 352, 'mattress': 353, 'hand rail': 354, 'folded table': 355, 'poster tube': 356, 'thermostat': 357, 'flip flops': 358, 'cloth': 359, 'banner': 360, 'clothes hanger': 361, 'whiteboard eraser': 362, 'shower control valve': 363, 'compost bin': 364, 'teddy bear': 365, 'pantry wall': 366, 'tupperware': 367, 'beer bottles': 368, 'salt': 369, 'mirror doors': 370, 'folded ladder': 371, 'carton': 372, 'soda stream': 373, 'metronome': 374, 'music book': 375, 'rice cooker': 376, 'dart board': 377, 'grab bar': 378, 'flowerpot': 379, 'painting': 380, 'railing': 381, 'stair': 382, 'quadcopter': 383, 'pitcher': 384, 'hanging': 385, 'mail': 386, 'closet ceiling': 387, 'hoverboard': 388, 'beanbag chair': 389, 'spray bottle': 390, 'soap bottle': 391, 'ikea bag': 392, 'duffel bag': 393, 'oven mitt': 394, 'pot': 395, 'hair brush': 396, 'tennis racket': 397, 'display case': 398, 'bananas': 399, 'carseat': 400, 'coffee box': 401, 'clothing rack': 402, 'bath walls': 403, 'podium': 404, 'storage box': 405, 'dolly': 406, 'shampoo': 407, 'changing station': 408, 'crutches': 409, 'grocery bag': 410, 'pizza box': 411, 'shaving cream': 412, 'luggage rack': 413, 'urinal': 414, 'hose': 415, 'bike pump': 416, 'bear': 417, 'humidifier': 418, 'mouthwash bottle': 419, 'golf bag': 420, 'food container': 421, 'card': 422, 'mug': 423, 'boxes of paper': 424, 'flag': 425, 'rolled poster': 426, 'wheel': 427, 'blackboard eraser': 428, 'doll': 429, 'laundry bag': 430, 'sponge': 431, 'lotion bottle': 432, 'lunch box': 433, 'sliding wood door': 434, 'briefcase': 435, 'bath products': 436, 'star': 437, 'coffee bean bag': 438, 'ipad': 439, 'display rack': 440, 'massage chair': 441, 'paper organizer': 442, 'cap': 443, 'dumbbell plates': 444, 'elevator': 445, 'cooking pan': 446, 'trash bag': 447, 'santa': 448, 'jewelry box': 449, 'boat': 450, 'sock': 451, 'plastic storage bin': 452, 'dishwashing soap bottle': 453, 'xbox controller': 454, 'airplane': 455, 'conditioner bottle': 456, 'tea kettle': 457, 'wall mounted coat rack': 458, 'film light': 459, 'sofa': 460, 'pantry shelf': 461, 'fish': 462, 'toy dinosaur': 463, 'cone': 464, 'fire sprinkler': 465, 'contact lens solution bottle': 466, 'hand sanitzer dispenser': 467, 'pen holder': 468, 'wig': 469, 'night light': 470, 'notepad': 471, 'drum set': 472, 'closet shelf': 473, 'exercise bike': 474, 'soda can': 475, 'stovetop': 476, 'telescope': 477, 'battery disposal jar': 478, 'closet floor': 479, 'clip': 480, 'display': 481, 'postcard': 482, 'paper towel': 483, 'food bag': 484}
    all_classes = np.array(list(all_classes))
else:
    raise Exception('not supported yet')

in_data = pd.read_csv(in_csv)
bs = 500
for i in tqdm(range(0, len(in_data), bs), desc = 'Processing'):
    print('{} / {}'.format(i, len(in_data)))
    sample = in_data.iloc[i:i+bs, :].reset_index()
    gt_target = sample['instance_type']
    pred_target = sample['target_object']
    sim_score = embed_matrix(pred_target.tolist(), all_classes)
    max_id = np.argmax(sim_score, axis=1)
    pred_target = all_classes[max_id]
    sim_score = embed_matrix(gt_target.tolist(), all_classes)
    max_id = np.argmax(sim_score, axis=1)
    gt_target = all_classes[max_id]

    pred_anchor = sample['anchor_objects']
    od = sample['referential_order']

    for b in range(len(sample)):
        p_t, filter_p_anchor, filter_order = mod_sample(gt_target[b], pred_target[b], sample['utterance'][b], str(pred_anchor[b]), od[b])
        sample['target_object'][b] = p_t
        sample['anchor_objects'][b] = filter_p_anchor
        sample['referential_order'][b] = filter_order

    sample.to_csv(out_csv, index = False, header = i==0, mode = 'a')