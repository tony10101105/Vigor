'''
This script generates the csv file with parts or all of the order shuffled
'''


import random
import ast
import pandas as pd
pd.options.mode.chained_assignment = None
random.seed(2023)


### parameters
options = 'wo_anchor_order' # wo_all_order, wo_anchor_order
in_csv = 'nr3d_train_LLM_step4_485.csv'
out_csv = 'nr3d_train_LLM_step4_485_{}.csv'.format(options)
###


cnt = 0
in_data = pd.read_csv(in_csv)
for i in range(len(in_data)):
    sample = in_data.iloc[i, :]
    order = ast.literal_eval(sample['referential_order'])
    rel = ast.literal_eval(sample['ref_objects_rel'])
    assert len(order) == len(rel)
    if options == 'wo_anchor_order':
        if len(order) <= 2:
            pass
        else:
            temp = list(zip(order[:-1], rel[:-1]))
            random.shuffle(temp)
            res1, res2 = zip(*temp)
            res1, res2 = list(res1), list(res2)
            order[:-1] = res1
            rel[:-1] = res2
            cnt += 1
    elif options == 'wo_all_order':
        if len(order) <= 1:
            pass
        else:
            temp = list(zip(order, rel))
            random.shuffle(temp)
            res1, res2 = zip(*temp)
            res1, res2 = list(res1), list(res2)
            order = res1
            rel = res2
            cnt += 1
    else:
        raise Exception('not implemented yet')
    
    sample['referential_order'] = order
    sample['ref_objects_rel'] = rel
    sample = sample.to_frame().T
    sample.to_csv(out_csv, index = False, header = i==0, mode = 'a')

print('number of changed samples: ')
print('cnt / total: {} / {}'.format(cnt, len(in_data)))