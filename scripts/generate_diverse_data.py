#!/usr/bin/env python3
"""
Generate structurally diverse prompt optimization pairs using Ollama Cloud.
Each output uses a different optimization strategy, not template filling.
"""

import json
import random
import time
import urllib.request
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHECKPOINT_FILE = DATA_DIR / "generation_checkpoint.json"
OUTPUT_FILE = DATA_DIR / "diverse_generated.jsonl"
LOG_FILE = DATA_DIR / "generation_log.json"

random.seed(42)

# Use flash for speed, pro for quality
MODEL = "deepseek-v4-flash:cloud"
API_URL = "http://localhost:11434/api/generate"
REQUEST_DELAY = 0.3  # seconds between requests

# ─── OPTIMIZATION STRATEGIES PER CATEGORY ───

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

# ─── VAGUE INPUT POOLS PER CATEGORY ───

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
        "analyze social media sentiment trends", "assess carbon footprint of operations",
        "compare authentication approaches", "evaluate mobile app performance",
        "analyze competitor feature releases", "assess accessibility compliance",
        "compare messaging queue technologies", "evaluate caching strategies",
        "analyze deployment failure patterns", "assess knowledge management gaps",
        "compare ORM solutions", "evaluate incident response process",
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
        "ways to improve release cadence", "ideas for data-driven decision making",
        "solutions for reducing meeting overload", "ideas for onboarding automation",
        "approaches to testing strategy", "ways to improve documentation culture",
        "ideas for internal tooling", "approaches to dependency management",
        "creative debugging methodologies", "ideas for observability improvements",
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
        "explain the actor model", "how does connection pooling work",
        "what is eventual consistency vs strong consistency", "explain circuit breaker pattern",
        "how does WebRTC work", "what is the difference between SQL and NoSQL",
        "explain horizontal vs vertical scaling", "how does a CDN work",
        "what is zero-trust security", "explain gRPC vs REST",
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
        "clarify this architecture decision record", "polish this code comment",
        "make this tutorial introduction more engaging", "streamline this onboarding guide",
        "rewrite this commit message for clarity", "improve this release notes",
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
        "guide me through container security", "how to set up health checks",
        "teach me about feature flags", "help me learn about infrastructure as code",
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
        "translate this announcement to Thai",
    ],
}

# ─── GENERATION PROMPTS ───

def make_generation_prompt(vague: str, category: str, strategy: str) -> str:
    """Create a prompt for the LLM to generate a diverse optimized prompt."""
    return f"""You are creating training data for a prompt optimization model. Transform this vague prompt into a specific, well-structured one.

Vague prompt: "{vague}"
Category: {category}
Optimization strategy: {strategy}

Rules:
- Be specific and actionable, not generic
- Add concrete constraints (word count, format, audience, deliverables)
- Preserve the core intent of the vague prompt
- Do NOT use the same structure as other {category} optimizations
- Do NOT start with generic verbs like "Explain", "Write", "Create" unless the strategy specifically calls for it
- Vary sentence structure, constraint placement, and formatting directives
- 60-300 characters ideal

Output ONLY the optimized prompt, nothing else."""


def call_ollama(prompt: str, model: str = MODEL, timeout: int = 30) -> str | None:
    """Call Ollama API to generate text."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.85,
            "num_predict": 300,
            "top_p": 0.9,
        }
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "").strip()
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return None


def validate_output(output: str, vague: str, category: str) -> list[str]:
    """Validate a generated output. Returns list of issues (empty = valid)."""
    issues = []
    
    if not output or len(output) < 40:
        issues.append("too_short")
    if len(output) > 500:
        issues.append("too_long")
    
    # Check for placeholder artifacts
    if any(artifact in output.lower() for artifact in ["use none", "-/-/-", "in none", "n/a"]):
        issues.append("placeholder_artifact")
    
    # Check for contradictory instructions
    length_markers = []
    if "briefly" in output.lower() or "short" in output.lower():
        length_markers.append("short")
    if "in detail" in output.lower() or "comprehensive" in output.lower() or "1500+" in output.lower():
        length_markers.append("long")
    if len(length_markers) > 1:
        issues.append("contradictory_length")
    
    # Check for broken grammar
    if "write a as" in output.lower() or "write a as a" in output.lower():
        issues.append("broken_grammar")
    
    # Check if output just repeats the input without adding value
    if output.lower().strip() == vague.lower().strip():
        issues.append("no_optimization")
    
    return issues


def load_checkpoint() -> dict:
    """Load generation checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "total_generated": 0}


def save_checkpoint(checkpoint: dict):
    """Save generation checkpoint."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def main():
    print("=" * 60)
    print("  DIVERSE DATA GENERATION")
    print("=" * 60)
    print(f"  Model: {MODEL}")
    print(f"  Target: ~600-800 diverse items")
    print()
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    existing_outputs = set()
    
    # Load any previously generated data
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                item = json.loads(line)
                existing_outputs.add(item["optimized"].strip().lower())
        print(f"  Loaded {len(existing_outputs)} existing generated items")
    
    # Plan generation
    target_per_category = {
        "coding": 100, "analysis": 70, "brainstorming": 60,
        "q_and_a": 60, "editing": 55, "instruction": 55,
        "roleplay": 35, "translation": 30,
    }
    
    total_target = sum(target_per_category.values())
    print(f"  Target per category: {target_per_category}")
    print(f"  Total target: {total_target}")
    print()
    
    # Generate
    generated = []
    failed = []
    cat_counts = defaultdict(int)
    
    # Load existing generated items
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                item = json.loads(line)
                generated.append(item)
                cat_counts[item["category"]] += 1
    
    for category, target in target_per_category.items():
        strategies = STRATEGIES[category]
        inputs = VAGUE_INPUTS[category]
        current_count = cat_counts.get(category, 0)
        
        if current_count >= target:
            print(f"  [{category}] Already have {current_count}/{target}, skipping")
            continue
        
        print(f"\n  [{category}] Generating {target - current_count} more items (have {current_count})...")
        
        # Cycle through inputs and strategies
        idx = 0
        attempts = 0
        max_attempts = (target - current_count) * 3  # Allow 3x attempts
        
        while cat_counts.get(category, 0) < target and attempts < max_attempts:
            strategy = strategies[idx % len(strategies)]
            vague = inputs[idx % len(inputs)]
            idx += 1
            attempts += 1
            
            # Skip if we already have enough
            if cat_counts.get(category, 0) >= target:
                break
            
            # Create generation prompt
            gen_prompt = make_generation_prompt(vague, category, strategy)
            
            # Call API
            output = call_ollama(gen_prompt)
            time.sleep(REQUEST_DELAY)
            
            if not output:
                failed.append({"category": category, "vague": vague, "strategy": strategy, "error": "api_failed"})
                continue
            
            # Validate
            issues = validate_output(output, vague, category)
            if issues:
                failed.append({"category": category, "vague": vague, "strategy": strategy, "error": f"validation: {issues}"})
                continue
            
            # Check uniqueness
            output_key = output.strip().lower()
            if output_key in existing_outputs:
                continue
            
            # Create item
            item = {
                "vague": vague,
                "optimized": output,
                "category": category,
            }
            generated.append(item)
            existing_outputs.add(output_key)
            cat_counts[category] = cat_counts.get(category, 0) + 1
            
            # Save progress every 50 items
            if len(generated) % 50 == 0:
                with open(OUTPUT_FILE, "w") as f:
                    for g in generated:
                        f.write(json.dumps(g, ensure_ascii=False) + "\n")
                print(f"    [Progress] {len(generated)} total items saved")
            
            if len(generated) % 20 == 0:
                print(f"    Generated {cat_counts[category]}/{target} for {category}, total: {len(generated)}")
    
    # Final save
    with open(OUTPUT_FILE, "w") as f:
        for item in generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Save failed log
    with open(LOG_FILE, "w") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total generated: {len(generated)}")
    print(f"  Failed: {len(failed)}")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")
    target_met = sum(1 for cat, target in target_per_category.items() if cat_counts.get(cat, 0) >= target)
    print(f"\n  Categories meeting target: {target_met}/{len(target_per_category)}")


if __name__ == "__main__":
    main()