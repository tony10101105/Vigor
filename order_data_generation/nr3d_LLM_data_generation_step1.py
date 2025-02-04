'''
This script conducts target object entity extraction and description summarization
'''


import pandas as pd
import openai
import time
pd.options.mode.chained_assignment = None


### parameters
in_csv = 'nr3d_train.csv' # this file is from referit3d challenge: https://referit3d.github.io/benchmarks.html
out_csv = 'nr3d_train_LLM_step1.csv'
key = 'YOUR_KEY' # OpenAI key
organization = 'YOUR_ORG'
###


openai.api_key = key
openai.organization = organization

with open('prompts_target_and_summarization.txt', 'r') as f: 
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
						{'role': 'user', 'content': 'Now for the sentence: ' + sample['utterance']},
						{'role': 'user', 'content': 'Give me the summarized sentence and the target object. Your answer must be in the form "summarized sentence: \n target object: "'},
					],
				temperature=0.2
			)
			flag = True
			answer = response['choices'][0]['message']['content'].split('\n')
			summarized_sentence = sample['utterance']
			target_object = sample['instance_type']

			for ans in answer:
				ans = ans.lower()
				if 'summarized' in ans:
					summarized_sentence = ans.split(': ')[-1]
				elif 'target' in ans:
					target_object = ans.split(': ')[-1]

			sample['target_object'] = target_object
			sample['summarized_utterance'] = summarized_sentence
			sample = sample.to_frame().T
			sample.to_csv(out_csv, index = False, header = i==0, mode = 'a')
		except:
			print('error. retry')
		time.sleep(3)