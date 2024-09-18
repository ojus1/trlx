import json
from datasets import load_dataset
import pandas as pd
import re
import os

good_path = "processed/good.parquet"

def clean_text(thoughts):
    if isinstance(thoughts, list):
        return [line.replace("<reflection>", "").replace("</reflection>", "").replace("<reflection/>", "") for line in thoughts]
    return thoughts.replace("<reflection>", "").replace("</reflection>", "").replace("<reflection/>", "")

def extract_assistant_message(assistant_text):
    # Regex to capture the sections
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", assistant_text, re.DOTALL)
    reflections_match = re.findall(r"<reflection>(.*?)</reflection>", assistant_text, re.DOTALL)
    output_match = re.search(r"<output>(.*?)</output>", assistant_text, re.DOTALL)

    # Extract the matched sections or assign empty if not found
    thinking = thinking_match.group(1).strip() if thinking_match else ''
    reflections = [reflection.strip() for reflection in reflections_match]
    output = output_match.group(1).strip() if output_match else ''
    
    # Return the result in dictionary form
    return {
        "thinking": thinking,
        "reflections": reflections,
        "output": output
    }

def load_skunkworks():
    ds = load_dataset("SkunkworksAI/reasoning-0.01")
    ds = ds["train"]
    rows = []
    for i in range(len(ds)):
        data = ds[i]
        data["id"] = i
        data["dataset"] = "skunkworks"
        data["reasoning_chains"] = json.dumps([item["thought"] for item in data["reasoning_chains"]])
        rows.append(data)
    return pd.DataFrame(rows)

def load_reflective_magllama():
    ds = load_dataset("thesven/Reflective-MAGLLAMA-v0.1")
    ds = ds["train"]
    rows = []
    skipped = 0
    for i in range(len(ds)):
        data_raw = ds[i]
        try:
            parsed = extract_assistant_message(data_raw["text"][1]["content"])
            data = {
                "instruction": data_raw["instruction"],
                "reasoning": clean_text("\n".join([parsed["thinking"]] + parsed["reflections"])),
                "output": parsed["output"],
                "reasoning_chains": json.dumps(clean_text([parsed["thinking"]] + parsed["reflections"]))
            } 
            data["id"] = i
            data["dataset"] = "reflective_magllama"
            rows.append(data)
        except Exception as e:
            skipped += 1
            print(e)
    print(f"Skipped {skipped}/{len(ds)}.")
    return pd.DataFrame(rows)

def load_all_good():
    if os.path.exists(good_path):
        return pd.read_parquet(good_path)

    df = pd.concat([
        load_reflective_magllama(),
        load_skunkworks()
        ],
        axis=0
    )
    df.to_parquet(good_path, compression='snappy', index=False)
    return df

def extract_text(element):
    """
    Extracts and cleans text from an XML element, including its tail text.
    """
    texts = []
    if element.text:
        texts.append(element.text)
    for child in element:
        if child.tail:
            texts.append(child.tail)
    return ' '.join(texts).strip()


if __name__ == "__main__":

    df = load_all_good()


    import tiktoken, numpy as np
    enc = tiktoken.get_encoding('o200k_base')

    examples = df["instruction"] + df["reasoning"] + df["output"]
    token_counts = examples.apply(lambda text: len(enc.encode(text)))

    # Calculate statistics
    min_tokens = token_counts.min()
    max_tokens = token_counts.max()
    avg_tokens = token_counts.mean()
    p90_tokens = np.percentile(token_counts, 90)
    p99_tokens = np.percentile(token_counts, 99)

    # Display results
    result = {
        "Min tokens": min_tokens,
        "Max tokens": max_tokens,
        "Average tokens": avg_tokens,
        "P90 tokens": p90_tokens,
        "P99 tokens": p99_tokens
    }

    print(df)
    print(result)
