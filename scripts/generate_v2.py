#!/usr/bin/env python3
"""
Generate v2 diverse data for prompt optimizer.
More strategies per category, longer outputs, no prefixes.
Target: ~800 additional items to reach 1500+ total.
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
NUM_WORKERS = 3
REQUEST_TIMEOUT = 30

random.seed(123)

STRATEGIES = {
    "coding": [
        "Provide a step-by-step implementation plan with file structure, function signatures, and edge case handling",
        "Write a detailed specification for a developer including expected inputs, outputs, error codes, and performance targets",
        "Design a technical solution with architecture diagram description, data flow, and integration points",
        "Create a debugging workflow: reproduction steps, diagnostic commands, common failure modes, and resolution paths",
        "Build a feature specification with acceptance criteria, test scenarios, and rollback plan",
        "Draft an API contract with endpoints, request/response schemas, authentication, and rate limits",
        "Outline a refactoring plan: current issues, proposed changes, migration steps, and risk assessment",
        "Define a performance optimization task with current metrics, target benchmarks, and optimization strategies",
    ],
    "analysis": [
        "Conduct a structured analysis using SWOT framework with quantified metrics and prioritized recommendations",
        "Perform a comparative analysis with feature matrix, scoring rubric, and weighted decision criteria",
        "Analyze trends over time with historical context, current state assessment, and 3-month projections",
        "Evaluate risk factors using probability x impact matrix with mitigation strategies and contingency plans",
        "Assess the current situation against industry benchmarks with gap identification and improvement roadmap",
        "Break down a complex problem into component parts, analyze each independently, then synthesize findings",
        "Map stakeholder interests and conflicts, then propose a resolution framework with trade-off analysis",
        "Identify root causes using 5-why analysis, then propose targeted interventions with expected outcomes",
    ],
    "brainstorming": [
        "Generate solutions organized by effort-impact matrix: quick wins, medium investment, and moonshots",
        "Create ideas for 3 distinct audience segments, with tailored messaging and delivery channels for each",
        "List creative approaches using systematic techniques: analogies from other domains, constraint removal, and inversion",
        "Develop a solution portfolio: 3 conservative, 3 moderate, and 3 radical ideas with feasibility scores",
        "Brainstorm by first identifying failure modes, then inverting each into a positive approach",
        "Generate implementation-ready ideas with resource estimates, timeline, and success metrics for each",
        "Map the solution space across 2 axes (complexity vs impact) and populate all 4 quadrants",
        "Create variations on a theme: take 1 core concept and explore 5 different applications or adaptations",
    ],
    "q_and_a": [
        "Explain using the Feynman technique: simple analogy first, then progressive depth with technical details",
        "Answer with a comparison table covering definition, key differences, use cases, and common misconceptions",
        "Provide a layered explanation: one-sentence answer, paragraph for context, and detailed technical dive",
        "Address the question by first stating common misconceptions, then the correct understanding with evidence",
        "Explain through a chronological narrative: origin, evolution, current state, and future trajectory",
        "Answer with practical examples and counter-examples, emphasizing boundary conditions and edge cases",
        "Structure as Q&A cascade: main question, 3 follow-up questions a curious person would ask, with answers",
        "Provide both a conceptual overview (for non-technical readers) and a technical deep-dive (for practitioners)",
    ],
    "editing": [
        "Restructure for clarity: reorder sections, add transitions, simplify jargon, and cut redundancy by 20%",
        "Rewrite for impact: strengthen the opening, add data-driven evidence, and close with a clear call to action",
        "Adapt for a different audience: change vocabulary level, adjust examples, and modify assumed knowledge",
        "Compress while preserving key information: remove filler words, combine related points, use active voice",
        "Enhance readability: shorten sentences to under 20 words, add headers every 3 paragraphs, use bullet lists",
        "Strengthen argumentation: add supporting evidence for each claim, address counterarguments, and close gaps",
        "Convert from informal to formal register: replace slang, add citations, use passive voice strategically",
        "Polish for publication: fix grammar, improve flow, ensure consistent terminology, and add a clear thesis",
    ],
    "instruction": [
        "Create a numbered tutorial with prerequisites, verification steps, troubleshooting tips, and estimated time",
        "Write a quick-reference guide with command examples, common patterns, and gotchas to avoid",
        "Design a project-based learning path with milestones, deliverables, and self-assessment checkpoints",
        "Build a decision-tree guide: start with a question, branch based on user context, provide targeted instructions",
        "Write a setup guide from zero to working in under 30 minutes, with OS-specific instructions",
        "Create an interactive walkthrough: key concepts, hands-on exercises, and expected results at each stage",
        "Draft an operations runbook: daily tasks, monitoring commands, alert responses, and escalation procedures",
        "Write a migration guide with pre-flight checklist, step-by-step migration, and post-migration verification",
    ],
    "roleplay": [
        "Create a character profile with expertise areas, communication style, decision framework, and knowledge boundaries",
        "Set up a scenario-based consultation: present the problem, role constraints, and expected deliverable format",
        "Design a mentorship persona that adapts explanation depth based on the learner's demonstrated understanding",
        "Act as a strategist: analyze the situation from your expert perspective, identify 3 options, and recommend one",
        "Play a devil's advocate role: challenge assumptions, highlight risks, and propose safer alternatives",
        "Simulate a technical interview: ask probing questions, evaluate responses, and provide constructive feedback",
        "Role-play as a consultant delivering a brief: executive summary, detailed analysis, and clear recommendations",
        "Act as a crisis manager: assess the situation, prioritize actions, delegate tasks, and set communication protocol",
    ],
    "translation": [
        "Translate with localization focus: adapt cultural references, idioms, and measurement units for the target audience",
        "Provide a dual-register translation: formal business version and casual conversational version side by side",
        "Translate with terminology glossary: highlight key terms, explain translation choices, and note ambiguous phrases",
        "Adapt for target locale: adjust date formats, currency, honorifics, and legal/regulatory context",
        "Create a translation with annotations explaining linguistic decisions and cultural adaptations made",
        "Translate preserving rhetorical devices: metaphors, alliteration, and emotional tone must survive the translation",
        "Produce a technical translation maintaining consistent terminology, following industry-standard glossary",
        "Translate and restructure: not just word-for-word, but reorganize for target language natural flow and conventions",
    ],
}

# More vague inputs per category, avoiding overlap with v1
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
    return f"""You are creating training data for a prompt optimization model. Transform this vague prompt into a specific, well-structured one.

IMPORTANT: Output ONLY the optimized prompt. No prefixes, no labels, no "Optimized Prompt:" — just the optimized text directly.

Vague prompt: "{vague}"
Category: {category}
Optimization strategy: {strategy}

Rules:
- Be specific and actionable
- Add concrete details: word counts, audience, format, deliverables
- Preserve the core intent
- Vary sentence structure and formatting
- 60-400 characters ideal
- NO prefix, NO "Optimized Prompt:", NO quotes around the whole thing

Optimized prompt:"""


def call_api(prompt, retries=2):
    for attempt in range(retries):
        try:
            payload = json.dumps({
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.85, "num_predict": 300, "top_p": 0.9}
            }).encode()
            req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                result = json.loads(resp.read().decode())
                text = result.get("response", "").strip()
                # Clean up model prefixes
                for prefix in ["Optimized Prompt:", "Optimized prompt:", "**Optimized:**",
                                "**Optimized Prompt:**", "Optimized:", "Here is the optimized prompt:",
                                "Here's the optimized prompt:", "Of course.", "Sure!"]:
                    if text.startswith(prefix):
                        text = text[len(prefix):].strip()
                # Remove markdown bold at start
                if text.startswith("**"):
                    text = text.lstrip("*").lstrip(":").strip()
                # Remove surrounding quotes
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].strip()
                return text
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return None
        except Exception:
            return None
    return None


def validate(output, vague):
    if not output or len(output) < 40:
        return False
    if len(output) > 600:
        return False
    if any(a in output.lower() for a in ["use none", "-/-/-", "in none"]):
        return False
    if output.lower().strip() == vague.lower().strip():
        return False
    # Reject if starts with "Optimized Prompt:" (model still adding it)
    if output.lower().startswith("optimized prompt"):
        return False
    has_short = any(w in output.lower() for w in ["briefly", "short response", "under 100 words"])
    has_long = any(w in output.lower() for w in ["in detail", "1500+", "comprehensive guide"])
    if has_short and has_long:
        return False
    return True


def generate_item(args):
    """Generate a single item."""
    vague, category, strategy, idx = args
    prompt = make_prompt(vague, category, strategy)
    output = call_api(prompt)
    if output and validate(output, vague):
        return {"vague": vague, "optimized": output, "category": category}, True
    return None, False


def main():
    print("Starting v2 batch generation...")
    print(f"Model: {MODEL}, Workers: {NUM_WORKERS}")

    # Load existing data to avoid duplicates
    existing = set()
    for fname in ["diverse_generated.jsonl", "diverse_v2.jsonl", "train.jsonl", "val.jsonl", "test.jsonl"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Index by vague+optimized for dedup
                        existing.add((item["vague"].strip().lower(), item["optimized"].strip().lower()[:100]))

    print(f"Existing items: {len(existing)}")

    # Count unique items per category (deduplicated)
    seen_for_count = set()
    cat_counts = defaultdict(int)
    for fname in ["diverse_generated.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "cleaned_data.jsonl"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        key = item["optimized"].strip().lower()[:150]
                        if key not in seen_for_count:
                            seen_for_count.add(key)
                            cat_counts[item["category"]] += 1

    # Target per category to reach ~200+ each
    target_per_cat = {
        "coding": 250, "analysis": 200, "brainstorming": 200,
        "q_and_a": 200, "editing": 200, "instruction": 200,
        "roleplay": 150, "translation": 100,
    }

    tasks = []
    for category, target in target_per_cat.items():
        current = cat_counts.get(category, 0)
        needed = max(0, target - current)
        strategies = STRATEGIES[category]
        inputs = VAGUE_INPUTS[category]
        for i in range(needed):
            strategy = strategies[i % len(strategies)]
            vague = inputs[i % len(inputs)]
            tasks.append((vague, category, strategy, i))

    random.shuffle(tasks)
    print(f"Tasks to generate: {len(tasks)}")

    # Generate with thread pool
    generated = []
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(generate_item, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            item, success = future.result()
            if success:
                key = (item["vague"].strip().lower(), item["optimized"].strip().lower()[:100])
                if key not in existing:
                    generated.append(item)
                    existing.add(key)
            elif not success:
                failed += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                print(f"  Progress: {i+1}/{len(tasks)} done, {len(generated)} generated, {failed} failed, {rate:.0f}/min")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        for item in generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    cat_counts_final = defaultdict(int)
    for item in generated:
        cat_counts_final[item["category"]] += 1

    print(f"\n{'='*50}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"Total items generated: {len(generated)}")
    print(f"Failed: {failed}")
    for cat in sorted(cat_counts_final.keys()):
        print(f"  {cat}: {cat_counts_final[cat]}")


if __name__ == "__main__":
    main()