#!/usr/bin/env python3
"""
Generate v2 data - simplified, fast, with checkpointing.
"""

import json
import random
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "diverse_v2.jsonl"

MODEL = "qwen3-coder:480b-cloud"
API_URL = "http://localhost:11434/api/generate"
NUM_WORKERS = 4
REQUEST_TIMEOUT = 30
CHECKPOINT_EVERY = 50

random.seed(456)

PROMPTS_PER_CAT = {
    "coding": 123, "analysis": 89, "brainstorming": 111,
    "instruction": 112, "editing": 105, "q_and_a": 135,
    "roleplay": 106, "translation": 61,
}

STRATEGIES = [
    "Provide a detailed, structured plan with specific constraints, target audience, and deliverables",
    "Create a specification with numbered requirements, success criteria, and scope boundaries",
    "Design with a focus on practical implementation: concrete steps, tools, and verification methods",
    "Write as if briefing a senior professional: include context, constraints, and expected outcomes",
    "Break down into phases or components, each with clear objectives and measurable results",
    "Frame as a time-boxed task with specific format, approximate length, and key sections to include",
    "Approach from the user's perspective: what they need, why, and how the result should help them",
    "Specify the output format explicitly: structure, sections, approximate length, and tone",
]

VAGUE_INPUTS = {
    "coding": [
        "help me add authentication", "make my API faster", "build a dashboard",
        "set up error handling", "create a data pipeline", "implement rate limiting",
        "write a batch processor", "add logging to my service", "build a retry mechanism",
        "create a config system", "implement data validation", "set up health checks",
        "build a notification service", "add search functionality", "create API versioning",
        "implement data encryption", "build a job scheduler", "add request throttling",
        "create a plugin architecture", "implement audit trail", "build a caching layer",
        "set up database indexing", "create a deployment pipeline", "implement SSO",
        "build a file storage service", "add data migration tooling", "create monitoring dashboards",
        "implement webhook delivery", "build a task queue", "add content negotiation",
        "create a dependency injection system", "implement circuit breaker",
        "build a service registry", "add distributed locking", "create a feature management system",
        "implement data deduplication", "build a message broker integration",
        "set up database connection pooling", "create a job retry system",
        "implement graceful degradation", "build a tenant isolation layer",
        "add observability to my service", "create a deployment rollback system",
        "implement request coalescing", "build a service mesh integration",
        "add data archival strategy", "create a load shedding mechanism",
        "implement idempotency keys", "build a consensus protocol wrapper",
    ],
    "analysis": [
        "review our cloud costs", "evaluate ML framework options", "assess team velocity",
        "compare database solutions", "analyze user retention", "review security audit findings",
        "evaluate build vs buy", "analyze API performance bottlenecks",
        "assess technical debt impact", "compare CI/CD platforms",
        "evaluate microservices readiness", "analyze deployment frequency",
        "review error rates by service", "compare observability tools",
        "assess data pipeline reliability", "evaluate caching strategy",
        "analyze user journey drop-offs", "compare search engine options",
        "evaluate container orchestration", "assess incident response effectiveness",
        "analyze cost per transaction", "compare storage solutions",
        "evaluate authentication approaches", "assess documentation coverage",
        "analyze resource utilization", "compare testing frameworks",
        "evaluate vendor lock-in risk", "assess scalability bottlenecks",
    ],
    "brainstorming": [
        "ways to improve developer experience", "ideas for reducing meeting time",
        "approaches to knowledge management", "features for a project management tool",
        "methods for reducing context switching", "ideas for team onboarding",
        "approaches to incident prevention", "ways to improve documentation",
        "ideas for internal tooling", "approaches to reduce technical debt",
        "methods for better code review", "ideas for remote collaboration",
        "approaches to capacity planning", "ways to improve deployment safety",
        "ideas for cost optimization", "methods for improving code quality",
        "approaches to reducing toil", "ideas for better monitoring",
        "ways to speed up feedback loops", "approaches to knowledge sharing",
        "ideas for improving MTTR", "methods for better estimation",
        "approaches to reducing WIP", "ideas for self-service platforms",
    ],
    "q_and_a": [
        "what is SRE", "explain zero trust security", "how does Kafka work",
        "what is data mesh", "explain observability vs monitoring",
        "how does container networking work", "what is infrastructure as code",
        "explain blue-green deployment", "how does service discovery work",
        "what is event sourcing", "explain CQRS pattern",
        "how does distributed tracing work", "what is chaos engineering",
        "explain canary deployment", "how does OAuth2 differ from SAML",
        "what is the CAP theorem", "explain idempotency in distributed systems",
        "how does content negotiation work", "what is domain-driven design",
        "explain the actor model", "how does raft consensus work",
        "what is backpressure in streaming", "explain eventual consistency",
        "how does TLS handshake work", "what is feature flagging",
        "explain saga pattern", "how does load shedding work",
    ],
    "editing": [
        "improve this README", "tighten this pull request description",
        "make this error message clearer", "rewrite this outage post-mortem",
        "improve this onboarding guide", "clarify this architecture decision record",
        "polish this release notes draft", "simplify this API documentation",
        "tighten this status update", "make this RFC more persuasive",
        "clarify this incident timeline", "improve this code review comment",
        "rewrite this alert notification", "enhance this design doc abstract",
        "simplify this runbook", "refine this stakeholder update",
        "improve this changelog entry", "tighten this retrospective summary",
        "clarify this debugging guide", "polish this project proposal",
        "make this SLA definition more precise", "rewrite this sprint review notes",
    ],
    "instruction": [
        "help me learn Vim", "guide me through Terraform setup",
        "teach me about message queues", "walk me through Docker networking",
        "show me how to debug memory issues", "help me understand observability",
        "teach me about API gateway patterns", "guide me through Kubernetes networking",
        "how to set up disaster recovery", "help me learn about service mesh",
        "walk me through database replication", "show me how to implement SLOs",
        "teach me about distributed consensus", "guide me on data pipeline design",
        "help me understand rate limiting algorithms", "teach me about cache invalidation",
        "walk me through event-driven architecture", "show me how to do canary releases",
    ],
    "roleplay": [
        "act as a principal engineer", "be a reliability expert",
        "roleplay as a platform team lead", "simulate being a DBA",
        "act as a security auditor", "be a product-minded engineer",
        "roleplay as an engineering manager", "pretend to be a solutions architect",
        "simulate being a cloud cost optimizer", "act as a migration specialist",
        "be a performance engineer", "roleplay as a data platform owner",
    ],
    "translation": [
        "translate this error page to German", "localize this signup flow for Japanese",
        "convert this CLI help text to Korean", "translate this onboarding email to Spanish",
        "adapt this privacy policy for French", "localize this pricing page for Brazilian Portuguese",
        "translate this notification template to Italian", "convert this error message catalog to Dutch",
        "adapt this legal disclaimer for Swedish", "translate this changelog to Polish",
        "localize this API status page for Thai", "convert this documentation to Turkish",
    ],
}

def make_prompt(vague, category, strategy):
    return f"""Transform this vague prompt into a specific, well-structured one. Output ONLY the optimized prompt text, nothing else. No prefix, no label.

Vague: "{vague}"
Category: {category}
Strategy: {strategy}

Optimized:"""


def call_api(prompt, retries=2):
    for attempt in range(retries):
        try:
            payload = json.dumps({
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.85, "num_predict": 250, "top_p": 0.9}
            }).encode()
            req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                result = json.loads(resp.read().decode())
                text = result.get("response", "").strip()
                # Remove prefixes the model might add
                for prefix in ["Optimized Prompt:", "Optimized prompt:", "**Optimized:**",
                                "**Optimized Prompt:**", "Optimized:", "Here is the optimized prompt:",
                                "Here's the optimized prompt:", "Of course.", "Sure!"]:
                    if text.startswith(prefix):
                        text = text[len(prefix):].strip()
                if text.startswith("**"):
                    text = text.lstrip("*").lstrip(":").strip()
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].strip()
                return text
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None
        except Exception:
            return None
    return None


def validate(output, vague):
    if not output or len(output) < 40 or len(output) > 600:
        return False
    if any(a in output.lower() for a in ["use none", "-/-/-"]):
        return False
    if output.lower().strip() == vague.lower().strip():
        return False
    if output.lower().startswith(("optimized prompt", "optimized prompt:", "here is")):
        return False
    return True


def main():
    print("Starting v2 generation...")
    print(f"Model: {MODEL}, Workers: {NUM_WORKERS}")

    # Load existing for dedup
    existing = set()
    for fname in ["cleaned_data.jsonl", "diverse_generated.jsonl"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        existing.add(item["optimized"].strip().lower()[:150])

    # Also load any already-generated v2 items
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing.add(item["optimized"].strip().lower()[:150])

    print(f"Existing unique outputs: {len(existing)}")

    # Build tasks
    tasks = []
    for category, count in PROMPTS_PER_CAT.items():
        strategies = STRATEGIES
        inputs = VAGUE_INPUTS[category]
        for i in range(count):
            strategy = strategies[i % len(strategies)]
            # Combine strategy index with input index for variety
            input_idx = (i * 7 + i) % len(inputs)  # pseudo-random but deterministic
            vague = inputs[input_idx]
            tasks.append((vague, category, strategy, i))

    random.shuffle(tasks)
    print(f"Tasks to generate: {len(tasks)}")

    generated = []
    failed = 0
    start = time.time()

    def gen_one(args):
        vague, category, strategy, idx = args
        prompt = make_prompt(vague, category, strategy)
        output = call_api(prompt)
        if output and validate(output, vague):
            return {"vague": vague, "optimized": output, "category": category}, True
        return None, False

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(gen_one, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            item, success = future.result()
            if success and item["optimized"].strip().lower()[:150] not in existing:
                generated.append(item)
                existing.add(item["optimized"].strip().lower()[:150])
            elif not success:
                failed += 1

            # Checkpoint every CHECKPOINT_EVERY items
            if len(generated) > 0 and len(generated) % CHECKPOINT_EVERY == 0:
                with open(OUTPUT_FILE, "w") as f:
                    for g in generated:
                        f.write(json.dumps(g, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                print(f"  {i+1}/{len(tasks)} done, {len(generated)} generated, {failed} failed, {rate:.0f}/min")

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        for item in generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    cat_counts = defaultdict(int)
    for item in generated:
        cat_counts[item["category"]] += 1

    print(f"\n{'='*50}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"Total generated: {len(generated)}")
    print(f"Failed: {failed}")
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat}: {cat_counts[cat]}")


if __name__ == "__main__":
    main()