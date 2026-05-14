#!/usr/bin/env python3
"""
Batch generate prompt optimization pairs using Ollama Cloud API.
Generates data in small batches per category to handle rate limits.
"""

import json
import random
import re
import subprocess
import time
import sys
from pathlib import Path
from typing import Optional

CATEGORIES = {
    "writing": {
        "count": 300,
        "topics": [
            "artificial intelligence", "climate change", "space exploration",
            "remote work", "mental health", "sustainable living", "digital privacy",
            "creativity", "education reform", "ocean conservation", "music therapy",
            "urban gardening", "cryptocurrency", "social media impact",
            "cultural diversity", "renewable energy", "vintage fashion",
            "street food", "minimalism", "public libraries", "board games",
            "podcasting", "amateur astronomy", "fermentation", "urban cycling",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt is about {category} topics.\nThe vague prompt should be 3-15 words, typical of what a lazy user would type.\nThe optimized prompt should be 2-5x longer with: clear task, format, audience, tone, length, or constraints.\nDo NOT over-engineer. Add 2-4 dimensions per prompt, not all 6.\nTopics to vary: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
    "coding": {
        "count": 300,
        "topics": [
            "sorting algorithm", "file upload handler", "API authentication",
            "database connection pool", "email validator", "caching system",
            "rate limiter", "data parser", "web scraper", "queue processor",
            "log rotation", "config loader", "retry mechanism", "pagination helper",
            "input sanitizer", "CSV export", "image resizer", "search function",
            "notification system", "backup script", "health check endpoint",
            "task scheduler", "data migration", "encryption utility",
            "dependency injection",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt is about {category} (programming) tasks.\nThe vague prompt should be 3-15 words, like what a developer asks ChatGPT.\nThe optimized prompt should specify: language, exact requirements, edge cases, output format, error handling.\nAdd 2-4 dimensions, not all possible ones.\nProgramming topics to vary: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
    "analysis": {
        "count": 250,
        "topics": [
            "quarterly sales data", "user engagement metrics", "market trends",
            "competitor strategies", "supply chain efficiency", "customer feedback",
            "website performance", "team productivity", "risk factors",
            "investment opportunities", "technology adoption", "energy consumption",
            "traffic patterns", "budget allocation", "employee satisfaction",
            "product features", "pricing strategy", "conversion rates",
            "security vulnerabilities", "resource utilization",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt asks for analysis of {category} topics.\nThe vague prompt should be 3-15 words, underspecified.\nThe optimized prompt should add: framework, metrics, scope, depth, audience, format, or focus areas.\nAnalysis topics to vary: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
    "translation": {
        "count": 200,
        "topics": [
            "Finnish", "English", "Swedish", "German", "French",
            "Spanish", "Japanese", "formal Finnish", "casual English",
            "academic English", "technical language", "business Finnish",
            "legal text", "medical text", "marketing copy",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt is about translation tasks.\nThe vague prompt should be 3-10 words, like 'translate this to Finnish'.\nThe optimized prompt should specify: target language variant, register (formal/casual), domain, audience, what to preserve, formatting.\nTranslation scenarios: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
    "q_and_a": {
        "count": 250,
        "topics": [
            "Kubernetes", "blockchain", "machine learning", "DNS",
            "OAuth 2.0", "containerization", "REST APIs", "Git branching",
            "microservices", "load balancing", "SQL indexes", "CI/CD pipelines",
            "HTTP/3", "WebAssembly", "GraphQL", "event sourcing",
            "serverless computing", "edge computing", "zero trust",
            "observability", "feature flags", "canary deployments",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt asks a question about {category} topics.\nThe vague prompt is a short question a beginner would ask.\nThe optimized prompt should add: depth level, audience, examples wanted, analogies, format, prerequisites.\nTech topics: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
    "roleplay": {
        "count": 150,
        "topics": [
            "senior developer reviewing code", "career counselor",
            "fitness coach", "travel planner for Europe",
            "interviewer for a tech role", "nutritionist",
            "financial advisor", "study group leader",
            "project manager", "UX researcher",
            "security auditor", "data analyst",
            "language tutor", "startup mentor",
            "devops engineer",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt asks to roleplay as someone.\nThe vague prompt is short like 'act as a career counselor'.\nThe optimized prompt should specify: persona depth, experience level, scenario, interaction style, specific tasks to perform, constraints.\nRoleplay personas: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
    "summarization": {
        "count": 200,
        "topics": [
            "this article", "the meeting notes", "the research paper",
            "the book chapter", "the podcast episode", "the quarterly report",
            "the user feedback", "the documentation", "the incident report",
            "the product roadmap", "the team retrospective", "the conference talk",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt asks to summarize something.\nThe vague prompt is short like 'summarize this' or 'tldr the meeting'.\nThe optimized prompt should specify: length, format, audience, focus areas, style, what to extract.\nSummarization contexts: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
    "brainstorming": {
        "count": 150,
        "topics": [
            "a mobile app", "a birthday party", "team building",
            "content marketing", "cost reduction", "employee engagement",
            "sustainability initiatives", "product features", "learning projects",
            "side hustles", "community events", "wellness programs",
            "gift ideas", "portfolio projects", "startup ideas",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt asks for ideas or brainstorming.\nThe vague prompt is short like 'give me ideas' or 'brainstorm X'.\nThe optimized prompt should specify: number of ideas, constraints, format, domain, feasibility level, evaluation criteria.\nBrainstorming topics: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
    "instruction": {
        "count": 100,
        "topics": [
            "setting up a VPN", "creating a Docker container", "writing unit tests",
            "deploying to AWS", "building a REST API", "configuring a firewall",
            "implementing CI/CD", "writing documentation", "code review process",
            "database migration", "setting up monitoring", "creating a GitHub Action",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt asks how to do something.\nThe vague prompt is short like 'how to set up docker'.\nThe optimized prompt should specify: detail level, audience, prerequisites, format (step-by-step), troubleshooting, verification steps.\nInstruction topics: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
    "editing": {
        "count": 100,
        "topics": [
            "email", "cover letter", "blog post", "presentation",
            "product description", "social media post", "report",
            "proposal", "README", "documentation", "job posting",
            "FAQ section", "landing page copy", "newsletter",
        ],
        "prompt_templates": [
            "Generate {n} pairs where the vague prompt asks to improve or rewrite text.\nThe vague prompt is short like 'improve this email' or 'make this better'.\nThe optimized prompt should specify: tone, audience, format, style guide, what to preserve, specific improvements wanted.\nEditing contexts: {topics}\n\nReturn ONLY a JSON array of objects with 'vague' and 'optimized' keys.",
        ],
    },
}


def call_ollama(prompt: str, model: str = "qwen3-coder:480b-cloud", temperature: float = 0.8) -> Optional[str]:
    """Call Ollama Cloud API."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 4096}
    })

    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/generate", "-d", payload],
            capture_output=True, text=True, timeout=180
        )
        if result.returncode == 0:
            resp = json.loads(result.stdout)
            return resp.get("response", "")
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  API error: {e}")
    return None


def parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from model response."""
    # Try to find JSON array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Try to find individual JSON objects
    pairs = []
    for match in re.finditer(r'\{[^{}]*"vague"[^{}]*"optimized"[^{}]*\}', text):
        try:
            obj = json.loads(match.group())
            pairs.append(obj)
        except json.JSONDecodeError:
            continue
    return pairs


import re
from typing import Optional


def main():
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load seed data to know how many we already have
    seed_path = data_dir / "seed_data.jsonl"
    seed_count = len(seed_path.read_text().strip().split("\n")) if seed_path.exists() else 0

    all_generated = []
    model = "qwen3-coder:480b-cloud"

    # Try fallback models if primary fails
    fallback_models = ["deepseek-v4-pro:cloud", "qlm-5.1:cloud", "deepseek-v3.1:671b-cloud"]

    for category, config in CATEGORIES.items():
        # How many seed examples we have for this category
        seed_cats = ["writing"] * 4 + ["coding"] * 4 + ["analysis"] * 3 + ["translation"] * 2 + \
                    ["q_and_a"] * 2 + ["roleplay"] * 2 + ["summarization"] * 2 + \
                    ["brainstorming"] * 2 + ["instruction"] * 2 + ["editing"] * 2
        existing = seed_cats.count(category) if category in seed_cats else 0
        target = config["count"]
        needed = target - existing

        print(f"\n{'='*50}")
        print(f"Category: {category}")
        print(f"Seed: {existing}, Target: {target}, Need: {needed}")

        if needed <= 0:
            print("  Already at target, skipping")
            continue

        topics_str = ", ".join(random.sample(config["topics"], min(8, len(config["topics"]))))
        batch_size = 10
        batches_needed = (needed + batch_size - 1) // batch_size
        category_items = []

        for batch_num in range(batches_needed):
            n = min(batch_size, needed - len(category_items))
            if n <= 0:
                break

            print(f"  Batch {batch_num + 1}/{batches_needed} ({n} pairs)...")

            prompt = config["prompt_templates"][0].format(
                n=n,
                category=category,
                topics=topics_str,
            )

            response = call_ollama(prompt, model=model)
            if response is None:
                # Try fallback models
                for fb_model in fallback_models:
                    print(f"    Primary model failed, trying {fb_model}...")
                    response = call_ollama(prompt, model=fb_model)
                    if response:
                        break

            if response:
                pairs = parse_json_response(response)
                valid = 0
                for p in pairs:
                    if "vague" in p and "optimized" in p:
                        p["category"] = category
                        # Validate: optimized must be meaningfully longer
                        if len(p["optimized"]) > len(p["vague"]) * 1.5:
                            category_items.append(p)
                            valid += 1
                print(f"    Got {valid} valid pairs")
            else:
                print(f"    All models failed for this batch")

            time.sleep(3)  # Rate limiting

        all_generated.extend(category_items)
        print(f"  Total for {category}: {len(category_items)}")

    # Save generated data
    gen_path = data_dir / "generated_data.jsonl"
    with open(gen_path, "w") as f:
        for item in all_generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Generated {len(all_generated)} total items")
    print(f"Saved to {gen_path}")

    # Run cleaning
    print("\nRunning cleaning pipeline...")
    from clean_data import clean_dataset, load_jsonl

    input_paths = [seed_path, gen_path]
    output_path = Path(__file__).parent.parent / "data" / "cleaned_data.jsonl"

    all_items = []
    for path in input_paths:
        if path.exists():
            all_items.extend(load_jsonl(path))

    # Quick stats
    from collections import Counter
    cats = Counter(i.get("category", "unknown") for i in all_items)
    print(f"\nRaw total: {len(all_items)}")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()