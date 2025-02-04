'''
This script extracts the anchor objects and constructs initial referential order
'''


import pandas as pd
import openai
import time
pd.options.mode.chained_assignment = None


### parameters
in_csv = 'nr3d_train_LLM_step1.csv'
out_csv = 'nr3d_train_LLM_step2.csv'
key = 'YOUR_KEY' # OpenAI key
organization = 'YOUR_ORG'
###


openai.api_key = key
openai.organization = organization

with open('prompts_anchor_and_order.txt', 'r') as f: 
	prompts = f.read()

in_data = pd.read_csv(in_csv)

for i in range(len(in_data)):
	flag = False
	sample = in_data.iloc[i, :]
	while (flag == False):
		try:
			response = openai.ChatCompletion.create(
				model='gpt-3.5-turbo',
				messages=[
						{'role': 'user', 'content': prompts},
						{'role': 'user', 'content': 'Now for the sentence: ' + sample['summarized_utterance']},
						{'role': 'user', 'content': 'It has the target object: ' + sample['target_object']},
						{'role': 'user', 'content': 'Give me the anchor objects and the referential order. Your answer must be in the form "referential order: \n anchor objects: "'},
					],
				temperature=0.2
			)
			flag = True
			answer = response['choices'][0]['message']['content'].split('\n')
			referential_order = sample['target_object']
			anchor_objects = []
			for ans in answer:
				ans = ans.lower()
				if 'referential' in ans:
					referential_order = ans.split(': ')[-1]
				elif 'anchor' in ans:
					anchor_objects = ans.split(': ')[-1]

			sample['anchor_objects'] = anchor_objects
			sample['referential_order'] = referential_order
			sample = sample.to_frame().T
			sample.to_csv(out_csv, index = False, header = i==0, mode = 'a')
		except:
			print('error. retry')
		time.sleep(3)