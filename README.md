[**Contextual Classification of Cybercriminal Posts Using Large Language Models: A Comprehensive Study on Tech Support Scam Marketplaces**](https://ieeexplore.ieee.org/document/11327876)

An asynchronous, few-shot NLP pipeline designed to classify posts from cybercriminal and illicit marketplaces into 12 distinct threat intelligence categories using Large Language Models (LLMs).

This project was developed as part of my PhD research to automate the analysis of dark web and illicit forum communications.

##  Overview

Analyzing informal marketplace data is difficult due to niche slang, evolving evasion tactics, and sheer volume. This project leverages **Gemma-3 (12B)** combined with strict few-shot prompting and Chain-of-Thought (CoT) reasoning to categorize raw criminal posts into actionable intelligence categories, such as:

* `Criminal IT infrastructure Operations`
* `Victim Data Sales`
* `Blasting Campaign Services`
* `Fake/Illicit Documents Services`
* `Job Offerings`
* `Money Launderers`
* `PPC/Popup Calls`
* `Remote Access Services`
* `Toll Free Number Provider`
* `Web Development Services`
* `Scammer Warnings`
* `Other`

Classifying these posts will help us develop a more targeted, data-driven understanding of the tech support scam ecosystem and inform more effective disruption strategies.

##  Technical Highlights

* **Asynchronous LLM Inference:** Utilizes `asyncio` and `httpx` to send concurrent API requests, significantly speeding up the labeling of large datasets.
* **Local Processing:** Integrated with [Ollama](https://ollama.com/) to run inference locally, ensuring sensitive threat intelligence data never leaves the local environment.
* **Complex Prompt Engineering:** Implements strict intent-to-definition matching to prevent model hallucination and enforce rigid taxonomy adherence.

## Quickstart

### 1. Prerequisites
Ensure you have Python 3.9+ installed and Ollama running locally with your model of choice:
```bash
ollama run gemma3:12b
