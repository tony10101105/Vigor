'''
This script conducts referential order generation of the SR3D dataset
LLMs are not required since anchor object entities are provided and their orders are also known
'''


import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import json
import torch
import clip
import math
import inflect
p = inflect.engine()
pd.options.mode.chained_assignment = None
clip_model, _ = clip.load('ViT-B/16', device='cuda')


### parameters
in_csv = 'sr3d_train.csv' # this file is from referit3d challenge: https://referit3d.github.io/benchmarks.html
out_csv = 'sr3d_train_LLM_step4_485.csv'
###


def embed_matrix(samples, concepts):
    dot_prods_c = _clip_dot_prods(samples, concepts)
    return dot_prods_c

def _clip_dot_prods(list1, list2, device='cuda', batch_size=1):
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


f = open('concept_relation.json')
concept = json.load(f)

in_data = pd.read_csv(in_csv)

# 485 cls
all_classes = {'wall': 0, 'chair': 1, 'floor': 2, 'table': 3, 'door': 4, 'couch': 5, 'cabinet': 6, 'shelf': 7, 'desk': 8, 'office chair': 9, 'bed': 10, 'pillow': 11, 'sink': 12, 'picture': 13, 'window': 14, 'toilet': 15, 'bookshelf': 16, 'monitor': 17, 'curtain': 18, 'book': 19, 'armchair': 20, 'coffee table': 21, 'drawer': 22, 'box': 23, 'refrigerator': 24, 'lamp': 25, 'kitchen cabinet': 26, 'towel': 27, 'clothes': 28, 'tv': 29, 'nightstand': 30, 'counter': 31, 'dresser': 32, 'stool': 33, 'couch cushions': 34, 'plant': 35, 'ceiling': 36, 'bathtub': 37, 'end table': 38, 'dining table': 39, 'keyboard': 40, 'bag': 41, 'backpack': 42, 'toilet paper': 43, 'printer': 44, 'tv stand': 45, 'whiteboard': 46, 'carpet': 47, 'blanket': 48, 'shower curtain': 49, 'trash can': 50, 'closet': 51, 'staircase': 52, 'microwave': 53, 'rug': 54, 'stove': 55, 'shoe': 56, 'computer tower': 57, 'bottle': 58, 'bin': 59, 'ottoman': 60, 'bench': 61, 'board': 62, 'washing machine': 63, 'mirror': 64, 'copier': 65, 'basket': 66, 'sofa chair': 67, 'file cabinet': 68, 'fan': 69, 'laptop': 70, 'shower': 71, 'paper': 72, 'person': 73, 'headboard': 74, 'paper towel dispenser': 75, 'faucet': 76, 'oven': 77, 'footstool': 78, 'blinds': 79, 'rack': 80, 'plate': 81, 'blackboard': 82, 'piano': 83, 'heater': 84, 'soap': 85, 'suitcase': 86, 'rail': 87, 'radiator': 88, 'recycling bin': 89, 'container': 90, 'wardrobe closet': 91, 'soap dispenser': 92, 'telephone': 93, 'bucket': 94, 'clock': 95, 'stand': 96, 'light': 97, 'laundry basket': 98, 'pipe': 99, 'round table': 100, 'clothes dryer': 101, 'coat': 102, 'guitar': 103, 'toilet paper holder': 104, 'seat': 105, 'step': 106, 'speaker': 107, 'vending machine': 108, 'column': 109, 'bicycle': 110, 'ladder': 111, 'cover': 112, 'bathroom stall': 113, 'foosball table': 114, 'shower wall': 115, 'chest': 116, 'cup': 117, 'jacket': 118, 'storage bin': 119, 'screen': 120, 'coffee maker': 121, 'hamper': 122, 'dishwasher': 123, 'paper towel roll': 124, 'machine': 125, 'mat': 126, 'windowsill': 127, 'tap': 128, 'pool table': 129, 'hand dryer': 130, 'bar': 131, 'frame': 132, 'toaster': 133, 'handrail': 134, 'bulletin board': 135, 'ironing board': 136, 'fireplace': 137, 'soap dish': 138, 'kitchen counter': 139, 'glass': 140, 'doorframe': 141, 'toilet paper dispenser': 142, 'mini fridge': 143, 'fire extinguisher': 144, 'shampoo bottle': 145, 'ball': 146, 'hat': 147, 'shower curtain rod': 148, 'toiletry': 149, 'water cooler': 150, 'desk lamp': 151, 'paper cutter': 152, 'switch': 153, 'tray': 154, 'shower door': 155, 'shirt': 156, 'pillar': 157, 'ledge': 158, 'vase': 159, 'toaster oven': 160, 'mouse': 161, 'nerf gun': 162, 'toilet seat cover dispenser': 163, 'can': 164, 'furniture': 165, 'cart': 166, 'step stool': 167, 'dispenser': 168, 'storage container': 169, 'side table': 170, 'lotion': 171, 'cooking pot': 172, 'toilet brush': 173, 'scale': 174, 'tissue box': 175, 'remote': 176, 'light switch': 177, 'crate': 178, 'ping pong table': 179, 'platform': 180, 'slipper': 181, 'power outlet': 182, 'cutting board': 183, 'controller': 184, 'decoration': 185, 'trolley': 186, 'sign': 187, 'projector': 188, 'sweater': 189, 'globe': 190, 'closet door': 191, 'plastic container': 192, 'statue': 193, 'vacuum cleaner': 194, 'wet floor sign': 195, 'candle': 196, 'easel': 197, 'wall hanging': 198, 'dumbell': 199, 'ping pong paddle': 200, 'plunger': 201, 'soap bar': 202, 'stuffed animal': 203, 'water fountain': 204, 'footrest': 205, 'headphones': 206, 'plastic bin': 207, 'coatrack': 208, 'dish rack': 209, 'broom': 210, 'guitar case': 211, 'mop': 212, 'magazine': 213, 'range hood': 214, 'scanner': 215, 'bathrobe': 216, 'futon': 217, 'dustpan': 218, 'hand towel': 219, 'organizer': 220, 'map': 221, 'helmet': 222, 'hair dryer': 223, 'exercise ball': 224, 'iron': 225, 'studio light': 226, 'cabinet door': 227, 'exercise machine': 228, 'workbench': 229, 'water bottle': 230, 'handicap bar': 231, 'tank': 232, 'purse': 233, 'vent': 234, 'piano bench': 235, 'bunk bed': 236, 'shoe rack': 237, 'shower floor': 238, 'case': 239, 'swiffer': 240, 'stapler': 241, 'cable': 242, 'garbage bag': 243, 'banister': 244, 'trunk': 245, 'tire': 246, 'folder': 247, 'car': 248, 'flower stand': 249, 'water pitcher': 250, 'loft bed': 251, 'shopping bag': 252, 'curtain rod': 253, 'alarm': 254, 'washcloth': 255, 'toolbox': 256, 'sewing machine': 257, 'mailbox': 258, 'toothpaste': 259, 'rope': 260, 'electric panel': 261, 'bowl': 262, 'boiler': 263, 'paper bag': 264, 'alarm clock': 265, 'music stand': 266, 'instrument case': 267, 'paper tray': 268, 'paper shredder': 269, 'projector screen': 270, 'boots': 271, 'kettle': 272, 'mail tray': 273, 'cat litter box': 274, 'covered box': 275, 'ceiling fan': 276, 'cardboard': 277, 'binder': 278, 'beachball': 279, 'envelope': 280, 'thermos': 281, 'breakfast bar': 282, 'dress rack': 283, 'frying pan': 284, 'divider': 285, 'rod': 286, 'magazine rack': 287, 'laundry detergent': 288, 'sofa bed': 289, 'storage shelf': 290, 'loofa': 291, 'bycicle': 292, 'file organizer': 293, 'fire hose': 294, 'media center': 295, 'umbrella': 296, 'barrier': 297, 'subwoofer': 298, 'stepladder': 299, 'shorts': 300, 'rocking chair': 301, 'elliptical machine': 302, 'coffee mug': 303, 'jar': 304, 'door wall': 305, 'traffic cone': 306, 'pants': 307, 'garage door': 308, 'teapot': 309, 'barricade': 310, 'exit sign': 311, 'canopy': 312, 'kinect': 313, 'kitchen island': 314, 'messenger bag': 315, 'buddha': 316, 'block': 317, 'stepstool': 318, 'tripod': 319, 'chandelier': 320, 'smoke detector': 321, 'baseball cap': 322, 'toothbrush': 323, 'bathroom counter': 324, 'object': 325, 'bathroom vanity': 326, 'closet wall': 327, 'laundry hamper': 328, 'bathroom stall door': 329, 'ceiling light': 330, 'trash bin': 331, 'dumbbell': 332, 'stair rail': 333, 'tube': 334, 'bathroom cabinet': 335, 'cd case': 336, 'closet rod': 337, 'coffee kettle': 338, 'wardrobe cabinet': 339, 'structure': 340, 'shower head': 341, 'keyboard piano': 342, 'case of water bottles': 343, 'coat rack': 344, 'storage organizer': 345, 'folded chair': 346, 'fire alarm': 347, 'power strip': 348, 'calendar': 349, 'poster': 350, 'potted plant': 351, 'luggage': 352, 'mattress': 353, 'hand rail': 354, 'folded table': 355, 'poster tube': 356, 'thermostat': 357, 'flip flops': 358, 'cloth': 359, 'banner': 360, 'clothes hanger': 361, 'whiteboard eraser': 362, 'shower control valve': 363, 'compost bin': 364, 'teddy bear': 365, 'pantry wall': 366, 'tupperware': 367, 'beer bottles': 368, 'salt': 369, 'mirror doors': 370, 'folded ladder': 371, 'carton': 372, 'soda stream': 373, 'metronome': 374, 'music book': 375, 'rice cooker': 376, 'dart board': 377, 'grab bar': 378, 'flowerpot': 379, 'painting': 380, 'railing': 381, 'stair': 382, 'quadcopter': 383, 'pitcher': 384, 'hanging': 385, 'mail': 386, 'closet ceiling': 387, 'hoverboard': 388, 'beanbag chair': 389, 'spray bottle': 390, 'soap bottle': 391, 'ikea bag': 392, 'duffel bag': 393, 'oven mitt': 394, 'pot': 395, 'hair brush': 396, 'tennis racket': 397, 'display case': 398, 'bananas': 399, 'carseat': 400, 'coffee box': 401, 'clothing rack': 402, 'bath walls': 403, 'podium': 404, 'storage box': 405, 'dolly': 406, 'shampoo': 407, 'changing station': 408, 'crutches': 409, 'grocery bag': 410, 'pizza box': 411, 'shaving cream': 412, 'luggage rack': 413, 'urinal': 414, 'hose': 415, 'bike pump': 416, 'bear': 417, 'humidifier': 418, 'mouthwash bottle': 419, 'golf bag': 420, 'food container': 421, 'card': 422, 'mug': 423, 'boxes of paper': 424, 'flag': 425, 'rolled poster': 426, 'wheel': 427, 'blackboard eraser': 428, 'doll': 429, 'laundry bag': 430, 'sponge': 431, 'lotion bottle': 432, 'lunch box': 433, 'sliding wood door': 434, 'briefcase': 435, 'bath products': 436, 'star': 437, 'coffee bean bag': 438, 'ipad': 439, 'display rack': 440, 'massage chair': 441, 'paper organizer': 442, 'cap': 443, 'dumbbell plates': 444, 'elevator': 445, 'cooking pan': 446, 'trash bag': 447, 'santa': 448, 'jewelry box': 449, 'boat': 450, 'sock': 451, 'plastic storage bin': 452, 'dishwashing soap bottle': 453, 'xbox controller': 454, 'airplane': 455, 'conditioner bottle': 456, 'tea kettle': 457, 'wall mounted coat rack': 458, 'film light': 459, 'sofa': 460, 'pantry shelf': 461, 'fish': 462, 'toy dinosaur': 463, 'cone': 464, 'fire sprinkler': 465, 'contact lens solution bottle': 466, 'hand sanitzer dispenser': 467, 'pen holder': 468, 'wig': 469, 'night light': 470, 'notepad': 471, 'drum set': 472, 'closet shelf': 473, 'exercise bike': 474, 'soda can': 475, 'stovetop': 476, 'telescope': 477, 'battery disposal jar': 478, 'closet floor': 479, 'clip': 480, 'display': 481, 'postcard': 482, 'paper towel': 483, 'food bag': 484}
all_classes = np.array(list(all_classes.keys()))
all_target_object, all_anchor_object, all_order, all_ref_rel = [], [], [], []

in_data['summarized_utterance'] = in_data['utterance']

for i in tqdm(range(len(in_data)), desc='Processing'):
    sample = in_data.iloc[i, :]
    anchors = ast.literal_eval(sample['anchors_types'])
    target = sample['instance_type']
    if target not in all_classes:
        sin = p.singular_noun(target)
        if sin in all_classes:
            target = sin
        else:
            print('target {} not in!'.format(target))
            sim = embed_matrix([target], all_classes)
            max_id = np.argmax(sim)
            target = all_classes[max_id]
            print('reassigned target: '+target)
            
    all_target_object.append(target)
    target_rel = concept[target]

    anchor_objects, anchor_rel = [], []
    for anc in anchors:
        if anc not in all_classes:
            sin = p.singular_noun(anc)
            if sin in all_classes:
                anc = sin
            else:
                print('anchor {} not in!'.format(anc))
                sim = embed_matrix([anc], all_classes)
                max_id = np.argmax(sim)
                anc = all_classes[max_id]
                print('reassigned anchor: ' + anc)
        anchor_objects.append(anc)
        anchor_rel.append(concept[anc])

    all_anchor_object.append(anchor_objects)
    
    anchor_objects.append(target)
    all_order.append(anchor_objects)

    anchor_rel.append(target_rel)
    all_ref_rel.append(anchor_rel)

in_data['target_object'] = all_target_object
in_data['anchor_objects'] = all_anchor_object
in_data['referential_order'] = all_order
in_data['ref_objects_rel'] = all_ref_rel
in_data.to_csv(out_csv, index = False, header = True)