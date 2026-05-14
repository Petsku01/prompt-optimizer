#!/usr/bin/env python3
"""
Fast batch generation using Ollama API with concurrency and retries.
Generates diverse prompt optimization pairs across all categories.
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
OUTPUT_FILE = DATA_DIR / "diverse_generated.jsonl"

MODEL = "qwen3-coder:480b-cloud"
API_URL = "http://localhost:11434/api/generate"
NUM_WORKERS = 3
REQUEST_TIMEOUT = 30

random.seed(42)

STRATEGIES = {
    "coding": [
        "step-by-step implementation guide with prerequisites and error handling",
        "code review request specifying what to check and why",
        "architecture design prompt asking for component relationships and data flow",
        "debugging assistance prompt with error context and reproduction steps",
        "performance optimization request with benchmarking requirements",
        "security audit prompt listing threat model and attack vectors",
        "refactoring request with before/after comparison and design patterns",
        "testing strategy prompt covering unit, integration, and edge cases",
    ],
    "analysis": [
        "SWOT analysis with visual layout and actionable recommendations",
        "risk assessment with probability/impact matrix and mitigation strategies",
        "cost-benefit analysis with quantified metrics and ROI calculation",
        "stakeholder impact assessment with conflicting interests identified",
        "competitive landscape comparison with feature matrix and positioning",
        "trend analysis with historical data, current state, and projection",
        "root cause analysis with fishbone diagram structure and evidence chain",
        "gap analysis between current and desired state with prioritized actions",
    ],
    "brainstorming": [
        "ideation session with divergent thinking techniques and evaluation criteria",
        "reverse brainstorming: identify what would make it fail, then invert",
        "SCAMPER technique applied systematically with examples for each letter",
        "cross-industry inspiration: transfer solutions from unrelated domains",
        "constraint-based creativity: ideate within specific real-world limitations",
        "progressive ideation: quick wins first, then moonshots, then hybrids",
        "audience-specific ideation: tailor solutions for 3 different user personas",
        "time-boxed ideation with phased deliverables and feasibility scoring",
    ],
    "q_and_a": [
        "Socratic dialogue format with leading questions and progressive depth",
        "comparison table with key differences highlighted and explained",
        "FAQ format addressing common misconceptions and follow-up questions",
        "analogy-based explanation connecting familiar domain to the target concept",
        "historical evolution approach: how the concept developed and why it matters now",
        "hands-on exercise with step-by-step verification checkpoints",
        "myth vs reality breakdown with evidence and sources for each claim",
        "layered explanation: 1-sentence version, 1-paragraph version, deep-dive version",
    ],
    "editing": [
        "clarity pass: simplify jargon, shorten sentences, add transition sentences",
        "persuasion pass: strengthen arguments, add evidence, address counterpoints",
        "accessibility pass: add definitions, simplify structure, improve readability score",
        "tone adjustment: convert from casual to formal, technical to conversational",
        "structural reorganization: reorder for impact and logical flow",
        "precision pass: replace vague terms with specific data and concrete examples",
        "audience alignment: reshape content for a specific reader profile",
        "compression pass: cut 30% while preserving all key information",
    ],
    "instruction": [
        "numbered tutorial with prerequisites, steps, verification, and troubleshooting",
        "interactive learning guide with exercises and self-assessment questions",
        "quick-reference guide with command cheat sheet and common patterns",
        "project-based learning path with milestone deliverables and dependencies",
        "video-style walkthrough with timestamps and chapter markers",
        "mentorship-style guide with decision points and trade-off explanations",
        "checklist-driven process with conditional branches and decision trees",
        "sandboxed exploration guide with safe experiments and expected outcomes",
    ],
    "roleplay": [
        "character profile with personality traits, communication style, and boundaries",
        "scenario-based roleplay with context, objectives, and success criteria",
        "interview-style interaction with role-specific questions and expected depth",
        "consultation format with structured discovery, analysis, and recommendations",
        "debate prep with arguments for both sides and strategic concessions",
        "teaching persona that adapts explanations to the student's level",
        "advisory role with decision framework and risk warnings",
        "collaborative partner role with shared goals and contribution expectations",
    ],
    "translation": [
        "localization-focused translation preserving cultural context and idioms",
        "technical translation with terminology consistency and glossary references",
        "creative adaptation that maintains tone, humor, and emotional impact",
        "formal register translation for official documents and certifications",
        "accessible translation with simplified syntax and plain language principles",
        "domain-specific translation with field expertise and jargon handling",
        "dual-format translation providing both literal and natural versions",
        "comparative translation highlighting key differences and linguistic choices",
    ],
}

VAGUE_INPUTS = {
    "coding": [
        "fix my python code", "help with my REST API", "optimize my database queries",
        "build a web scraper", "debug my authentication system", "implement caching",
        "create a rate limiter", "refactor my monolith", "set up CI/CD pipeline",
        "write unit tests for my app", "implement file upload handling", "build a search feature",
        "add pagination to my API", "create a data validation library", "implement websockets",
        "set up logging and monitoring", "build a notification system", "implement OAuth2 flow",
        "create a task queue worker", "add error handling to my service", "migrate database schema",
        "build an event-driven architecture", "implement feature flags", "create API documentation",
        "set up container orchestration", "build a payment integration", "implement real-time sync",
        "create a configuration management system", "set up automated backups", "build a health check system",
        "implement retry logic with exponential backoff", "create a plugin system",
        "add telemetry to my microservice", "build a data pipeline", "implement audit logging",
        "create a deployment automation script", "set up service mesh", "build a content management system",
        "implement idempotent API endpoints", "create a data migration tool",
        "debug memory leak in production", "optimize cold start time", "implement circuit breaker pattern",
        "build a recommendation engine", "add SSO to my application", "create a webhook delivery system",
        "implement data encryption at rest", "build a batch processing system",
    ],
    "analysis": [
        "evaluate our security posture", "analyze market trends for SaaS", "assess team productivity",
        "compare React vs Vue for our project", "evaluate investment opportunities in AI",
        "analyze customer churn patterns", "assess technical debt in our codebase",
        "compare cloud providers for migration", "evaluate vendor proposals",
        "analyze energy consumption patterns", "assess supply chain risks",
        "compare microservices vs monolith for startup", "evaluate open source tools",
        "analyze user engagement metrics", "assess regulatory compliance gaps",
        "compare build vs buy for enterprise software", "evaluate pricing strategies",
        "analyze traffic patterns for capacity planning", "assess disaster recovery readiness",
        "compare database solutions for analytics workload", "evaluate hiring pipeline efficiency",
        "analyze content performance across channels", "assess infrastructure cost optimization",
        "compare agile vs waterfall for our team", "evaluate API design approaches",
        "analyze conversion funnel drop-off points", "assess data quality issues",
        "compare observability platforms", "evaluate training program effectiveness",
    ],
    "brainstorming": [
        "give me app ideas", "how to reduce costs", "team building activities",
        "product features for our launch", "ways to improve customer retention",
        "startup ideas in fintech", "creative marketing campaigns", "content ideas for our blog",
        "solutions for remote work challenges", "growth strategies for B2B",
        "names for my startup", "ways to reduce onboarding time", "social media strategies",
        "features for a productivity app", "community engagement ideas",
        "ways to automate repetitive tasks", "ideas for a hackathon project",
        "approaches to technical documentation", "ideas for employee wellness program",
        "ways to improve code review process", "sustainability initiatives for office",
        "ideas for API monetization", "approaches to reduce technical debt",
        "creative ways to celebrate team wins", "solutions for knowledge sharing",
        "ideas for developer experience improvements", "approaches to incident management",
    ],
    "q_and_a": [
        "what is blockchain", "explain machine learning", "how does DNS work",
        "what is Kubernetes", "explain Docker containers", "how does OAuth work",
        "what is GraphQL", "explain microservices architecture", "how does git branching work",
        "what is CI/CD", "explain message queues", "how does TLS encryption work",
        "what is functional programming", "explain event-driven architecture",
        "how does load balancing work", "what is test-driven development",
        "explain API rate limiting", "how does garbage collection work",
        "what is eventual consistency", "explain the CAP theorem",
        "how does SSL certificate validation work", "what is domain-driven design",
        "explain reverse proxy", "how does DNS propagation work",
        "what is observability vs monitoring", "explain idempotency in APIs",
        "how does content-based routing work", "what is blue-green deployment",
        "explain circuit breaker pattern", "how does connection pooling work",
        "what is the difference between SQL and NoSQL", "explain horizontal vs vertical scaling",
    ],
    "editing": [
        "improve this email draft", "make this report more concise",
        "fix grammar in my essay", "strengthen this argument",
        "rewrite for a technical audience", "make this more persuasive",
        "simplify this documentation", "improve this landing page copy",
        "tighten this abstract", "polish this proposal",
        "restructure this blog post", "enhance this presentation script",
        "make this announcement clearer", "improve this README section",
        "refine this project brief", "clean up these meeting notes",
        "sharpen this executive summary", "clarify this technical specification",
        "rewrite this error message", "improve this changelog entry",
        "make this API docs more readable", "tighten this product description",
        "revise this incident report", "improve this job posting",
    ],
    "instruction": [
        "teach me Docker basics", "how to set up a firewall",
        "guide me through database migration", "walk me through API design",
        "help me learn Git", "teach me about load testing",
        "show me how to write a Dockerfile", "guide me through Kubernetes setup",
        "how to implement logging", "teach me about web security",
        "walk me through setting up monitoring", "help me learn about networking",
        "show me how to create a REST API", "guide me through deploying to cloud",
        "how to set up automated testing", "teach me about data modeling",
        "help me understand microservices", "walk me through setting up HTTPS",
        "show me how to use message queues", "guide me on API versioning",
        "help me learn about caching strategies", "teach me about database indexing",
        "walk me through setting up SSO", "show me how to implement rate limiting",
    ],
    "roleplay": [
        "act as a senior developer", "be a product manager", "roleplay as a CTO",
        "pretend to be a DevOps engineer", "simulate being a security consultant",
        "act as a data scientist", "be a tech lead", "roleplay as a startup founder",
        "pretend to be a UX researcher", "simulate being a solutions architect",
        "act as a project manager", "be a code reviewer", "roleplay as a tech interviewer",
        "pretend to be a mentor developer", "simulate being a reliability engineer",
    ],
    "translation": [
        "translate this to French", "convert this to Spanish",
        "localize this for Japanese market", "translate this documentation to German",
        "adapt this for Brazilian Portuguese", "translate this error message to Korean",
        "localize this UI for Arabic speakers", "convert this legal text to formal Italian",
        "translate this email to Mandarin", "adapt this marketing copy for Hindi speakers",
        "localize this app for Swedish users", "translate this contract to Dutch",
        "convert this recipe to Polish", "adapt this technical guide for Turkish readers",
    ],
}

def make_prompt(vague, category, strategy):
    return f"""You are creating training data for a prompt optimization model. Transform this vague prompt into a specific, well-structured one.

Vague prompt: "{vague}"
Category: {category}
Strategy: {strategy}

Rules:
- Be specific and actionable, not generic
- Add concrete constraints (word count, format, audience, deliverables)
- Preserve the core intent
- Do NOT start with generic verbs like "Explain" or "Write" unless the strategy specifically calls for it
- Vary sentence structure, constraint placement, and formatting directives
- 60-400 characters ideal

Optimized prompt:"""


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
                # Clean up model prefixes/formatting
                text = text.strip('"\'').strip()
                # Remove common prefixes
                for prefix in ["Optimized prompt:", "**Optimized:**", "Optimized:", "**Optimized prompt:**", "**Optimized Prompt:**", "Of course. ", "Of course!", "Sure! ", "Sure, ", "Here is the optimized prompt:\n", "### Optimized Prompt:\n", "### Optimized:\n"]:
                    if text.startswith(prefix):
                        text = text[len(prefix):].strip()
                # Remove markdown bold at start
                if text.startswith("**"):
                    text = text.lstrip("*").lstrip(":").strip()
                return text
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2)
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
    # Check for contradictory length instructions
    has_short = any(w in output.lower() for w in ["briefly", "short response", "under 100 words"])
    has_long = any(w in output.lower() for w in ["in detail", "1500+", "comprehensive guide"])
    if has_short and has_long:
        return False
    return True


def generate_item(args):
    """Generate a single item. Returns (item, success)."""
    vague, category, strategy, idx = args
    prompt = make_prompt(vague, category, strategy)
    output = call_api(prompt)
    if output and validate(output, vague):
        return {"vague": vague, "optimized": output, "category": category}, True
    return None, False


def main():
    print("Starting batch generation...")
    print(f"Model: {MODEL}, Workers: {NUM_WORKERS}")

    # Load existing
    existing = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing.add(item["optimized"].strip().lower())
    print(f"Existing items: {len(existing)}")

    # Build task list
    target_per_cat = {
        "coding": 100, "analysis": 70, "brainstorming": 55,
        "q_and_a": 55, "editing": 50, "instruction": 50,
        "roleplay": 35, "translation": 30,
    }

    # Count what we have per category
    cat_counts = defaultdict(int)
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cat_counts[item["category"]] += 1

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

    random.shuffle(tasks)  # Mix categories for variety
    print(f"Tasks to generate: {len(tasks)}")

    # Generate with thread pool
    generated = []
    failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(generate_item, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            item, success = future.result()
            if success and item["optimized"].strip().lower() not in existing:
                generated.append(item)
                existing.add(item["optimized"].strip().lower())
            elif not success:
                failed += 1

            # Progress
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
                print(f"  Progress: {i+1}/{len(tasks)} done, {len(generated)} generated, {failed} failed, {rate:.0f}/min")

            # Save checkpoint every 100 items
            if len(generated) % 100 == 0 and generated:
                with open(OUTPUT_FILE, "a") as f:
                    for g in generated[-100:]:
                        f.write(json.dumps(g, ensure_ascii=False) + "\n")
                generated_new = []  # Track what's been saved

    # Final save
    # Read existing + add all new
    all_items = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    all_items.append(json.loads(line))
    all_items.extend(generated)

    # Deduplicate by output
    seen = set()
    unique_items = []
    for item in all_items:
        key = item["optimized"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    with open(OUTPUT_FILE, "w") as f:
        for item in unique_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    cat_counts_final = defaultdict(int)
    for item in unique_items:
        cat_counts_final[item["category"]] += 1

    print(f"\n{'='*50}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"Total items: {len(unique_items)}")
    print(f"Failed: {failed}")
    print(f"Rate: {len(tasks)/elapsed*60:.0f} items/min")
    for cat in sorted(cat_counts_final.keys()):
        print(f"  {cat}: {cat_counts_final[cat]}")


if __name__ == "__main__":
    main()