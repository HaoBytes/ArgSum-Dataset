# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is used to generated summary by GPT-4, GPT-3.5.

run:
python summary_generation.py \
  --output_dir ./ \\
  --model_name="gpt-3.5-turbo" \\
  --openai_api_key= "*******" \\
  --top_p=0.9\\
  --max_output_length=2048\\
  --input_file gpt_instructions/gpt_baseline_best_instructions.csv\
  
"""

import json, logging
import argparse
from functools import partial
import os, math
import sys
import nltk
import numpy as np
import pandas as pd
import random as rn
from tqdm.auto import tqdm
import openai


def load_dataset(input_file) -> list:
    # 

    # Load the CSV file into a pandas dataframe
    input_file = pd.read_csv(input_file)
    data_dict = input_file['Instructions'].to_list()

    return data_dict

def generate_debate_script(dataframe):
    '''
    building user instruction for GPT
    '''
    instructions = pd.DataFrame(columns=["Instructions"])  # Create an empty dataframe to store instructions
    evidence_columns = ['evidence']
    claim_variable = "claim"

    grouped_data = dataframe.groupby(['topic', 'stance'])

    for (topic, stance), group in grouped_data:
        #script = "You are an expert debater. Follow the argumentative structure of the debate script to summarize the given debate material presented below. The debate material consists of a topic; a stance; a series of arguments and evidence to support that argument. We already provide the best evidence that you need to make the debate scripts well-argued and well-persuaded. You may add background information as appropriate. The material for the debate as follow:\n\n"
        script = f"Given the topic: '{topic}', "
        if np.float64(stance) == 1.0:
            script += "Its positive supporters argue that"
        elif np.float64(stance) == -1.0:
            script += "Its opponents argue that"
        script += ":\n"

        claim_number = 1  # Start claim numbering from 1

        for i, row in group.iterrows():
            arguments = [row[col] for col in dataframe.columns if col.startswith('argument')]
            evidence = [row[col] for col in dataframe.columns if col in evidence_columns]

            for j, arg in enumerate(arguments):
                claim_variable_name = f"{claim_variable}_{claim_number}"
                script += f"- {claim_variable_name}: {arg}.\n"
                script += f"  Evidence supporting {claim_variable_name}:\n"
                for k, evi in enumerate(evidence):
                    script += f"    - Evidence {k+1}: {evi}.\n"

                claim_number += 1  # Increment the claim number

        instructions = instructions.append({"Instructions": script}, ignore_index=True)

    # Replace claim_* and evidence_* with actual values
    num_rows = len(dataframe)
    for i in range(1, num_rows+1):
        claim_variable_name = f"{claim_variable}_{i}"
        instructions["Instructions"] = instructions["Instructions"].str.replace(claim_variable_name, f"claim_{i}")
        instructions["Instructions"] = instructions["Instructions"].str.replace(f"evidence_{i}", f"evidence_{i+1}")

    return instructions



def generate_label(instructions, best_evidence=False, model_name="gpt-3.5-turbo",top_p=0.9,openai_api_key='Here is a key', max_output_length=2048) -> pd.DataFrame:
    """  
    Args:
        dataframe: A pandas dataframe containing the data to be labeled.
        instructions: A list of instructions to be given to the model.
        best_evidence: A boolean indicating whether to perform evidence classification.
        all_top2_evidence: A boolean indicating whether to perform pairwise evidence convinceness.
        save_file: A string indicating the path to save the result to.
        save_interval: An integer indicating how often (in seconds) to save the result to disk.
    """
    openai.api_key=openai_api_key

    model_responses = []
    model_outputs = [] # Define variable here
    model_engine = model_name

    # Define the prompt for generating text
    if best_evidence:
        chat_instruction = [{"role":"system","content":"You are an expert debater. Follow the argumentative structure of the debate script to summarize the given debate material presented below. The debate material consists of a topic; a stance; a series of arguments and evidence to support that argument. We already provide the best evidence that you need to make the debate scripts well-argued and well-persuaded. You may add background information as appropriate. The material for the debate as follow:\n\n"}]
    else :
        chat_instruction = [{"role":"system","content":"You are an expert debater. Follow the argumentative structure of the debate script to summarize the given debate material presented below. The debate material consists of a topic; a stance; a series of arguments and evidence to support that argument. You do not need to use all of the evidence, just select the most appropriate one or more that are well-argued and well-persuaded. You may add background information as appropriate. The material for the debate as follow:\n\n"}]


    if model_name == "text-davinci-003":
        #for gpt3
        for instruction in tqdm(instructions):
            input_text = "\n".join([msg["content"] for msg in chat_instruction]) + instruction
            model_response = openai.Completion.create(
                engine=model_engine,
                prompt=input_text,
                max_tokens=max_output_length,
                top_p=top_p,
                temperature=1
            )
            model_responses.append(model_response)
            model_outputs.append(model_response.choices[0].text)
    
    else:
        
        #for gpt3.5,gpt-4
        for instruction in tqdm(instructions):
            messages = chat_instruction + [{"role":"user","content":instruction}]
            model_response = openai.ChatCompletion.create(
                                model=model_engine, messages=messages, max_tokens=max_output_length, top_p=top_p, stop=['"'],
                                temperature=1)
            model_responses.append(model_response)

            model_outputs.append(model_response["choices"][0]["message"]["content"]) # Append output to variable


    # Create a new dataframe with the generated summaries
    output_df = pd.DataFrame({"Generated_summary": model_outputs})

    return output_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory to which the generated data is saved")
    
    # Text generation parameters
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo",
                        help="The pretrained model to use for dataset generation. Currently, only variants of GPT3.5,GPT4 are supported.")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="p value for top-p sampling (set to 0 to perform no top-p sampling)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="k value for top-k sampling (set to 0 to perform no top-k sampling)")
    parser.add_argument("--best_evidence", action="store_true",
                        help="Decide whether the input evidence contains only the best evidence, which can lead to different INSTRUCTIONs")
    parser.add_argument("--max_output_length", type=int, default=2048,
                        help="The maximum output length for each generated text.")

    # Data parameters
    parser.add_argument("--input_file", type=str, required=True, help="Path to a csv file into model")

    args = parser.parse_args()
    output_dir = args.output_dir

    data_dict = load_dataset(input_file = args.input_file)
    output_df = generate_label(data_dict, best_evidence = args.best_evidence, 
                               model_name = args.model_name, top_p = args.top_p, openai_api_key = args.openai_api_key, max_output_length = args.max_output_length)
    
    output_df.to_csv(output_dir ,index=False)