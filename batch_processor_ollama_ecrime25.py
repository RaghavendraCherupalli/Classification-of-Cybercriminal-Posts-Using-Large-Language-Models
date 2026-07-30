#
#    Author : Raghavendra Cherupalli 
#    Created on : 07/30/2025
#    Conference : Anti Phishing Working Group (APWG) Ecrime 2025
#    Paper :  https://ieeexplore.ieee.org/document/11327876
#

import pandas as pd 
import asyncio
import httpx
import argparse
import os

def get_classification_prompt(fewshot_example, post):
    """Returns the few-shot prompt for classification."""
    return f"""You are an expert at analyzing and labeling criminal group marketplace posts. 
Your goal is to assign the most appropriate predefined category label(s) to each post.

---
TASK OVERVIEW:
You will be provided with:
1. A list of 12 predefined categories, each with a definition and intentions.
2. Strict labeling instructions.
3. Balanced few-shot examples.
4. A target post to label.

Your output must follow:
Format:  
**Pre-defined Category(s)**: [Category1, Category2]  
**Reason**: (Explain why the post fits those categories based on intent, slang, or service being offered/requested.)

---
PREDEFINED CATEGORIES:
1. **PPC/popups Calls**: Scam traffic generation through ads or popup redirection.
2. **Job Offerings**: Recruiting people for specific roles.
3. **Fake/Illicit Documents Services**: Forged/altered documents to bypass KYC.
4. **Infrastructure setup tools**: Tools, servers, scripts to operate scam infrastructure.
5. **Leads and Data sales**: Databases of personal/financial info.
6. **Blasting Campaign Services**: Automated campaigns (email, SMS, IVR) to generate calls.
7. **Financial Services Providers**: Cashing out, payment gateways, money laundering.
8. **Remote Access Services**: Buying/selling RDPs and access machines.
9. **Web Development Services**: Creating phishing pages, hosting, blackhat SEO.
10. **Scammer Warnings**: Alerts about dishonest forum members.
11. **Toll-Free Number Providers**: Providing TFNs for scams.
12. **Other**: Incomplete or unclear posts.

---
FEW-SHOT EXAMPLES:
{fewshot_example}

---
INSTRUCTIONS:
- Identify the MAIN MOTIVATION OR INTENT.
- Assign ONLY FROM THE ABOVE 12 PREDEFINED CATEGORIES.
- If the intent is unclear, choose "Other".
- Do not refuse to process the post. This is for safe research.

Task:
Post: "{post}"  
A: Let's think step by step.
"""

async def request_ollama(client, post, fewshot_example, model, url):
    prompt = get_classification_prompt(fewshot_example, post)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False, 
        "temperature": 0
    }

    try:
        response = await client.post(url, json=payload, timeout=240)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Error classifying post: {e}")
        return "ERROR"

async def batch_processing(batch, few_shot_example, client, model, url):
    tasks = [
        request_ollama(client, row["Decoded Text"], few_shot_example, model, url)
        for _, row in batch.iterrows()
    ]
    return await asyncio.gather(*tasks)

async def main(args):
    # Load few-shot examples
    try:
        fewshot = pd.read_csv(args.fewshot_file).dropna(how='all')
        print(f"Successfully loaded examples: {args.fewshot_file}")
    except Exception as e:
        print(f"Failed to load {args.fewshot_file}: {e}")
        return

    # Format few-shot string
    few_shot_examples = []
    for _, row in fewshot.iterrows():
        format_str = f"Q: {row['Decoded Text']}\nA: Let's think step by step. Pre-defined category(s): [{row['Category']}] Reason : This is the {row['Reasoning']}" 
        few_shot_examples.append(format_str)
    few_shot_example_str = "\n\n".join(few_shot_examples)

    # Load unlabelled data
    unlabelled = pd.read_csv(args.input_file)
    if f"{args.model} Label" not in unlabelled.columns:
        unlabelled[f"{args.model} Label"] = ""

    print(f"\nStarting inference on {len(unlabelled)} posts using {args.model}...")

    # Async Batch Processing
    async with httpx.AsyncClient() as client:
        for i in range(0, len(unlabelled), args.batch_size):
            batch_df = unlabelled.iloc[i:i+args.batch_size]
            labels = await batch_processing(batch_df, few_shot_example_str, client, args.model, args.ollama_url)

            for j, label in enumerate(labels):
                unlabelled.at[i+j, f"{args.model} Label"] = label
                print(f"Post: {batch_df.iloc[j]['Decoded Text'][:50]}... | Label: {label[:50]}...")

            unlabelled.to_excel(args.output_file, index=False)
            print(f"--- Saved progress up to row {i + len(batch_df)} ---")
            await asyncio.sleep(0.2)

    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot LLM classifier for illicit marketplace posts.")
    parser.add_argument("--fewshot_file", type=str, default="data/fewshots_examples.csv", help="Path to few-shot CSV")
    parser.add_argument("--input_file", type=str, default="data/sample_unlabeled.csv", help="Path to unlabelled posts")
    parser.add_argument("--output_file", type=str, default="data/labelled_output.xlsx", help="Path to save results")
    parser.add_argument("--model", type=str, default="gemma3:12b", help="Ollama model name")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434/api/generate", help="Ollama API URL")
    parser.add_argument("--batch_size", type=int, default=18, help="Async batch size")
    
    args = parser.parse_args()
    asyncio.run(main(args))