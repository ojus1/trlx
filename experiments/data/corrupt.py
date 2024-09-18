import random
import json
import os
from data.get_data import load_all_good
import openai
from typing import List
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

# Set your OpenAI API key
client = openai.Client(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1/"
)

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ROOT = "processed/processing/llama3.1-8b/"

template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're an helpful assistant. Your responses are concise. You are not verbose; containing no "notes" or explanations or assumptions.\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_prefill}'''

def clean_text(thoughts):
    if isinstance(thoughts, list):
        return [line.replace("<reflection>", "").replace("</reflection>", "").replace("<reflection/>", "") for line in thoughts]
    return thoughts.replace("<reflection>", "").replace("</reflection>", "").replace("<reflection/>", "")

def delete_random_thoughts(thoughts: List[str], delete_probability=0.3, delim='\n'):
    if len(thoughts) > 1:
        line_to_delete = random.randint(0, len(thoughts) - 1)
        thoughts.pop(line_to_delete)
        
        kept_lines = [line for i, line in enumerate(thoughts) if random.random() > delete_probability]
        
        if not kept_lines:
            kept_lines = [random.choice(thoughts)]
    else:
        kept_lines = thoughts

    return delim.join(kept_lines)

def double_corrupt(output):
    lines = output.splitlines()
    delim = '\n'
    if len(lines) == 1:
        lines = [' '.join(output.split()[i::4]) for i in range(4)]
        delim = ''
    lines = delete_random_thoughts(lines, 0.3, delim=delim)
    return lines

def change_reasoning_chains(instruction, original_chains):
    prompt = f"""Given the following instruction and original reasoning chains, create a new set of reasoning chains that are different and incorrect, such that the output no longer corresponds to the instruction:

Instruction:
{instruction}

Original Reasoning Chains:
{original_chains}

"""

    response = client.completions.create(
        model=MODEL,
        prompt=template.format(prompt=prompt, assistant_prefill="Incorrect reasoning chain:\n"),
        max_tokens=512
    )

    return response.choices[0].text.strip()

def change_instruction(instruction, original_chains):
    prompt = f"""Given the following instruction, write a new instruction of similar subject such that the output no longer corresponds to the instruction:

Instruction:
{instruction}

Original Output:
{original_chains}

"""

    response = client.completions.create(
        model=MODEL,
        prompt=template.format(prompt=prompt, assistant_prefill="New instruction:\n"),
        max_tokens=512
    )

    return response.choices[0].text.strip()

def generate_corrupted_output(instruction, corrupted_reasoning):
    prompt = f"""Given the following instruction and reasoning, generate an output that follows the reasoning but does not correctly answer the original instruction:

Instruction:
{instruction}

Reasoning:
{corrupted_reasoning}

"""

    response = client.completions.create(
        model=MODEL,
        prompt=template.format(prompt=prompt, assistant_prefill="New output:\n"),
        max_tokens=512
    )

    return response.choices[0].text.strip()

def apply_corruptions(example):
    # Create filename for this example
    filename = f"{example['dataset']}-{example['id']}.json"
    filepath = os.path.join(ROOT, filename)

    # Check if file already exists
    if os.path.exists(filepath):
        return None  # Skip processing if file exists

    corruption_type = random.choice(['delete', 'change'])
    
    original_chains = clean_text(json.loads(example['reasoning_chains']))
    original_reasoning = clean_text(example["reasoning"])

    if corruption_type == 'delete':
        corrupted_reasoning = delete_random_thoughts(original_chains)
    else:  # change
        corrupted_reasoning = change_reasoning_chains(example['instruction'], original_reasoning)

    corrupted_output = generate_corrupted_output(example['instruction'], corrupted_reasoning)

    metadata = {
        'corruption_type': corruption_type,
        'original_id': int(example["id"]),
        'original_dataset': example["dataset"]
    }

    result = {
        'instruction': example['instruction'],
        'corrupted_reasoning': corrupted_reasoning,
        'corrupted_output': corrupted_output,
        'metadata': metadata
    }

    # Save the result to a JSON file
    with open(filepath, 'w') as f:
        json.dump(result, f)

    return result

def main():
    # Load the dataset from Hugging Face
    dataset = load_all_good()

    # Ensure the 'processed' directory exists
    os.makedirs('processed', exist_ok=True)

    # Apply corruptions to the dataset
    corrupted_dataset = Parallel(n_jobs=128, prefer='threads')(delayed(apply_corruptions)(row) for _, row in tqdm(dataset.iterrows(), total=len(dataset)))

    # Filter out None results (skipped examples)
    corrupted_dataset = []
    for item in os.listdir(ROOT):
        j = json.load(open(os.path.join(ROOT, item), 'r'))
        corrupted_dataset.append(j)
    
    corrupted_dataset = pd.DataFrame(corrupted_dataset)
    print(f"Processed {len(corrupted_dataset)} examples.")

if __name__ == "__main__":
    main()