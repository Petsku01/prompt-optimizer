#!/usr/bin/env python3
"""
Generate synthetic prompt optimization pairs for fine-tuning.

Creates (vague_prompt, optimized_prompt) pairs across 10 categories.
Uses Ollama Cloud models for generation with local templates as fallback.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

# ============================================================================
# Category definitions and templates
# ============================================================================

CATEGORIES = {
    "writing": {
        "count": 300,
        "vague_templates": [
            "write about {topic}",
            "tell me about {topic}",
            "create a story about {topic}",
            "write something about {topic}",
            "make a poem about {topic}",
            "draft something on {topic}",
            "write a blog post about {topic}",
            "compose an essay on {topic}",
            "write about {topic} briefly",
            "describe {topic} for me",
        ],
        "topics": [
            "artificial intelligence", "climate change", "space exploration",
            "remote work", "mental health", "sustainable living", "digital privacy",
            "creativity", "education reform", "ocean conservation", "music therapy",
            "urban gardening", "cryptocurrency", "social media impact",
            "cultural diversity", "renewable energy", "vintage fashion",
            "street food", "minimalism", "public libraries", "board games",
            "podcasting", "amateur astronomy", "fermentation", "urban cycling",
        ],
        "optimization_dims": ["format", "tone", "audience", "length", "structure", "perspective"],
    },
    "coding": {
        "count": 300,
        "vague_templates": [
            "fix this code: {topic}",
            "write a function for {topic}",
            "help me with {topic}",
            "debug {topic}",
            "create a script for {topic}",
            "optimize {topic}",
            "refactor {topic}",
            "add error handling to {topic}",
            "write tests for {topic}",
            "implement {topic}",
        ],
        "topics": [
            "a sorting algorithm", "file upload handler", "API authentication",
            "database connection pool", "email validator", "caching system",
            "rate limiter", "data parser", "web scraper", "queue processor",
            "log rotation", "config loader", "retry mechanism", "pagination helper",
            "input sanitizer", "CSV export", "image resizer", "search function",
            "notification system", "backup script", "health check endpoint",
            "task scheduler", "data migration", "encryption utility",
            "dependency injection",
        ],
        "optimization_dims": ["language", "constraints", "edge_cases", "style", "performance", "testing"],
    },
    "analysis": {
        "count": 250,
        "vague_templates": [
            "analyze {topic}",
            "compare {topic}",
            "what do you think about {topic}",
            "evaluate {topic}",
            "give me insights on {topic}",
            "break down {topic}",
            "assess {topic}",
            "investigate {topic}",
            "review {topic}",
            "explain the implications of {topic}",
        ],
        "topics": [
            "quarterly sales data", "user engagement metrics", "market trends",
            "competitor strategies", "supply chain efficiency", "customer feedback",
            "website performance", "team productivity", "risk factors",
            "investment opportunities", "technology adoption", "energy consumption",
            "traffic patterns", "budget allocation", "employee satisfaction",
            "product features", "pricing strategy", "conversion rates",
            "security vulnerabilities", "resource utilization",
        ],
        "optimization_dims": ["framework", "metrics", "scope", "depth", "audience", "format"],
    },
    "translation": {
        "count": 200,
        "vague_templates": [
            "translate {topic}",
            "translate this to {topic}",
            "convert {topic}",
            "make {topic} version",
            "say this in {topic}",
        ],
        "topics": [
            "Finnish", "English", "Swedish", "German", "French",
            "Spanish", "Japanese", "formal Finnish", "casual English",
            "academic English", "technical language", "business Finnish",
            "legal text", "medical text", "marketing copy",
        ],
        "optimization_dims": ["register", "domain", "audience", "style", "constraints", "context"],
    },
    "q_and_a": {
        "count": 250,
        "vague_templates": [
            "what is {topic}",
            "how does {topic} work",
            "explain {topic}",
            "tell me about {topic}",
            "why {topic}",
            "when to use {topic}",
            "difference between {topic}",
            "basics of {topic}",
        ],
        "topics": [
            "Kubernetes", "blockchain", "machine learning", "DNS",
            "OAuth 2.0", "containerization", "REST APIs", "Git branching",
            "microservices", "load balancing", "SQL indexes", "CI/CD pipelines",
            "HTTP/3", "WebAssembly", "GraphQL", "event sourcing",
            "serverless computing", "edge computing", "zero trust",
            "observability", "feature flags", "canary deployments",
        ],
        "optimization_dims": ["depth", "audience", "examples", "analogies", "format", "prerequisites"],
    },
    "roleplay": {
        "count": 150,
        "vague_templates": [
            "act as {topic}",
            "pretend you are {topic}",
            "roleplay as {topic}",
            "be a {topic}",
            "simulate {topic}",
        ],
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
        "optimization_dims": ["persona_depth", "scenario", "constraints", "style", "interaction_model"],
    },
    "summarization": {
        "count": 200,
        "vague_templates": [
            "summarize {topic}",
            "make a summary of {topic}",
            "shorten {topic}",
            "condense {topic}",
            "give me the key points of {topic}",
            "tldr {topic}",
            "recap {topic}",
            "extract main ideas from {topic}",
        ],
        "topics": [
            "this article", "the meeting notes", "the research paper",
            "the book chapter", "the podcast episode", "the quarterly report",
            "the user feedback", "the documentation", "the incident report",
            "the product roadmap", "the team retrospective", "the conference talk",
        ],
        "optimization_dims": ["length", "format", "audience", "focus", "style", "level"],
    },
    "brainstorming": {
        "count": 150,
        "vague_templates": [
            "give me ideas for {topic}",
            "brainstorm {topic}",
            "help me think about {topic}",
            "suggest {topic}",
            "what could I do with {topic}",
            "creative ideas for {topic}",
            "how to approach {topic}",
        ],
        "topics": [
            "a mobile app", "a birthday party", "team building",
            "content marketing", "cost reduction", "employee engagement",
            "sustainability initiatives", "product features", "learning projects",
            "side hustles", "community events", "wellness programs",
            "gift ideas", "portfolio projects", "startup ideas",
        ],
        "optimization_dims": ["quantity", "constraints", "format", "diversity", "feasibility", "domain"],
    },
    "instruction": {
        "count": 100,
        "vague_templates": [
            "how to {topic}",
            "teach me {topic}",
            "guide for {topic}",
            "steps for {topic}",
            "tutorial on {topic}",
            "instructions for {topic}",
        ],
        "topics": [
            "setting up a VPN", "creating a Docker container", "writing unit tests",
            "deploying to AWS", "building a REST API", "configuring a firewall",
            "implementing CI/CD", "writing documentation", "code review process",
            "database migration", "setting up monitoring", "creating a GitHub Action",
        ],
        "optimization_dims": ["detail_level", "audience", "format", "prerequisites", "troubleshooting"],
    },
    "editing": {
        "count": 100,
        "vague_templates": [
            "improve this {topic}",
            "rewrite {topic}",
            "make {topic} better",
            "polish {topic}",
            "fix the writing in {topic}",
            "enhance {topic}",
        ],
        "topics": [
            "email", "cover letter", "blog post", "presentation",
            "product description", "social media post", "report",
            "proposal", "README", "documentation", "job posting",
            "FAQ section", "landing page copy", "newsletter",
        ],
        "optimization_dims": ["tone", "audience", "format", "constraints", "style_guide", "focus"],
    },
}

# System prompt for generation
SYSTEM_PROMPT = """You are a prompt engineering expert. Your task is to generate realistic prompt optimization pairs.

RULES:
1. The "vague" prompt must be short (3-15 words), ambiguous, and representative of how real users write
2. The "optimized" prompt must preserve the original intent but add 2-4 optimization dimensions
3. Optimized prompts should be 2-5x longer than vague prompts
4. Do NOT over-engineer - not every prompt needs all dimensions
5. Be diverse in optimization strategies across examples
6. Use clear, specific language in optimized prompts
7. Vary the structure - some optimized prompts use bullet points, some use numbered steps, some are paragraphs

OUTPUT FORMAT - return ONLY a JSON array of objects with "vague" and "optimized" keys:
[{"vague": "...", "optimized": "..."}, ...]
"""


def generate_with_ollama(category: str, cat_config: dict, count: int = 10) -> list[dict]:
    """Generate pairs using Ollama Cloud models."""
    import subprocess

    topics = cat_config["topics"]
    templates = cat_config["vague_templates"]
    dims = cat_config["optimization_dims"]

    # Build generation prompt
    example_vague = random.choice(templates).format(topic=random.choice(topics))
    selected_topics = random.sample(topics, min(5, len(topics)))
    selected_dims = random.sample(dims, min(3, len(dims)))

    user_prompt = f"""Generate {count} prompt optimization pairs for the "{category}" category.

Category info:
- Vague templates: {templates[:5]}
- Topics: {selected_topics}
- Optimization dimensions to vary: {selected_dims}

Requirements:
- Each pair has a vague input and an optimized output
- Vague prompts: 3-15 words, ambiguous, realistic user language
- Optimized prompts: 2-5x longer, adds 2-4 relevant dimensions
- Vary which dimensions you add (don't add all to every prompt)
- Be creative and diverse
- Return ONLY the JSON array"""

    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/generate",
             "-d", json.dumps({
                 "model": "qwen3-coder:480b-cloud",
                 "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                 "stream": False,
                 "options": {"temperature": 0.8, "num_predict": 4096}
             })],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            response = json.loads(result.stdout)
            text = response.get("response", "")
            # Extract JSON array from response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                pairs = json.loads(text[start:end])
                return pairs
    except Exception as e:
        print(f"  Ollama generation failed: {e}")

    return []


def generate_manual_pairs(category: str, cat_config: dict) -> list[dict]:
    """Generate pairs manually from templates - fallback method."""
    pairs = []
    topics = cat_config["topics"]
    templates = cat_config["vague_templates"]
    dims = cat_config["optimization_dims"]

    # Specific optimization patterns per category
    optimization_patterns = {
        "writing": [
            "Write a {length} {format} about {topic}. {audience} {tone} {additional}",
            "Compose a {format} on {topic}. Focus on {focus}. {tone} {length}",
        ],
        "coding": [
            "Write a {language} function that {task}. Include: {requirements}. Handle edge cases: {edges}.",
            "Implement {task} in {language}. Requirements:\n- {req1}\n- {req2}\n- {req3}\n\nInclude error handling and type hints.",
        ],
        "analysis": [
            "Analyze {topic} using {framework}. Focus on: {focus_areas}. Present findings as {format} for {audience}.",
            "Perform a {depth} analysis of {topic}. Consider {aspects}. Format: {format}. Audience: {audience}.",
        ],
        "translation": [
            "Translate the following text to {target_lang}. Maintain a {register} tone appropriate for {context}. Preserve {preserve}.",
        ],
        "q_and_a": [
            "Explain {topic} in detail for a {audience} audience. Include {elements}. Use {analogy_type} if helpful. Format as {format}.",
        ],
        "roleplay": [
            "Act as a {persona} with {years} years of experience. Scenario: {scenario}. Respond in a {style} manner. Focus on {focus}.",
        ],
        "summarization": [
            "Summarize {topic} in {length} for {audience}. Focus on {focus}. Format as {format}. {additional_constraint}.",
        ],
        "brainstorming": [
            "Generate {quantity} creative ideas for {topic}. Constraints: {constraints}. Format: {format}. Prioritize {priority}.",
        ],
        "instruction": [
            "Write a step-by-step guide for {topic}. Target audience: {audience}. Include: {includes}. Format as numbered steps with headers.",
        ],
        "editing": [
            "Improve this {doc_type} by {actions}. Target audience: {audience}. Maintain {preservation}. Style: {style}.",
        ],
    }

    # Fill-in options for patterns
    fill_options = {
        "length": ["300-word", "500-word", "concise", "detailed", "brief", "comprehensive"],
        "format": ["essay", "report", "bullet points", "structured list", "markdown document", "numbered steps"],
        "audience": ["for a general audience", "for professionals", "for beginners", "for executives", "for developers", "for students"],
        "tone": ["Use a professional tone.", "Use a conversational tone.", "Use an academic tone.", "Use a friendly, approachable tone."],
        "language": ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C#"],
        "additional": ["Support your claims with examples.", "Include relevant statistics.", "Address counterarguments."],
    }

    for _ in range(cat_config["count"] // len(templates) + 2):
        topic = random.choice(topics)
        template = random.choice(templates)
        vague_prompt = template.format(topic=topic)

        # Simple approach: use category-specific optimization
        # This is a fallback - the LLM approach produces much better data
        optimized = vague_prompt + "."  # placeholder

        pairs.append({
            "vague": vague_prompt,
            "optimized": optimized,
            "category": category,
        })

    return pairs


# ============================================================================
# Seed data - high quality hand-crafted examples
# ============================================================================

SEED_DATA = [
    # Writing
    {"vague": "write about dogs", "optimized": "Write a 500-word informative essay about working dog breeds, covering their historical roles, modern applications, and training requirements. Use a professional tone and support your claims with specific examples.", "category": "writing"},
    {"vague": "tell me a story", "optimized": "Write a short adventure story (800-1000 words) set in a near-future Arctic research station. Include a morally complex decision the protagonist must face. Use vivid sensory details and end with an ambiguous resolution.", "category": "writing"},
    {"vague": "blog post about productivity", "optimized": "Write a 1200-word blog post titled 'The Myth of Maximum Productivity' for knowledge workers aged 25-40. Use a conversational tone with personal anecdotes. Structure with H2 headers. Include 3 actionable takeaways and cite at least 2 studies.", "category": "writing"},
    {"vague": "poem about rain", "optimized": "Write a free-verse poem about an unexpected summer rain in a city. Use vivid imagery and contrast the mechanical rhythm of urban life with the rain's unpredictable arrival. 15-20 lines, no rhyme scheme.", "category": "writing"},

    # Coding
    {"vague": "fix my python code", "optimized": "Debug the following Python code. Identify all bugs and issues, explain the root cause of each, and provide the corrected version with inline comments explaining each fix. Focus on: off-by-one errors, edge cases, and type safety.", "category": "coding"},
    {"vague": "write a web scraper", "optimized": "Write a Python web scraper using requests and BeautifulSoup that extracts article titles, authors, and publication dates from news websites. Include: retry logic with exponential backoff, rate limiting (1 req/sec), proper error handling, and output as JSON. Respect robots.txt.", "category": "coding"},
    {"vague": "make a REST API", "optimized": "Design and implement a RESTful API in Python using FastAPI for a task management system. Endpoints: CRUD for tasks, user authentication via JWT, filtering by status/priority. Include Pydantic models, input validation, proper HTTP status codes, and OpenAPI documentation.", "category": "coding"},
    {"vague": "database connection pool", "optimized": "Implement a thread-safe database connection pool in Python using psycopg2 for PostgreSQL. Requirements: configurable pool size (default 5), connection timeout (30s), automatic reconnection, context manager support, and graceful shutdown. Include type hints and docstrings.", "category": "coding"},

    # Analysis
    {"vague": "analyze sales data", "optimized": "Analyze the quarterly sales data focusing on: (1) year-over-year growth trends, (2) top 5 performing product categories, (3) seasonal patterns. Present findings as a structured report with charts for the executive team. Flag any anomalies or concerns.", "category": "analysis"},
    {"vague": "compare frameworks", "optimized": "Compare React, Vue, and Svelte for building a medium-complexity SaaS dashboard. Evaluate on: learning curve, bundle size, performance, ecosystem maturity, TypeScript support, and hiring availability. Provide a recommendation matrix and final suggestion for a 3-person startup team.", "category": "analysis"},
    {"vague": "evaluate our security", "optimized": "Conduct a security assessment of our web application covering: OWASP Top 10 vulnerabilities, authentication/authorization gaps, data encryption at rest and in transit, API rate limiting, and dependency vulnerabilities. Prioritize findings by risk level (Critical/High/Medium/Low) and provide remediation steps for each.", "category": "analysis"},

    # Translation
    {"vague": "translate to Finnish", "optimized": "Translate the following English text to Finnish. Maintain a formal register suitable for business communication. Preserve the original meaning exactly—do not localize cultural references. Keep technical terms in English where Finnish equivalents are not established.", "category": "translation"},
    {"vague": "make a Swedish version", "optimized": "Translate this product description to Swedish for the Swedish market (not Finland-Swedish). Use a marketing-friendly tone. Adapt measurements and formatting conventions to Swedish standards. Keep brand names unchanged.", "category": "translation"},

    # Q&A
    {"vague": "what is Kubernetes", "optimized": "Explain Kubernetes to a developer who understands containers but has no orchestration experience. Cover: (1) core concepts (pods, services, deployments), (2) how it differs from Docker Compose, (3) when you'd actually need it. Use analogies from software development. Format as a structured explanation with headings.", "category": "q_and_a"},
    {"vague": "how does OAuth work", "optimized": "Explain OAuth 2.0 authorization flow step by step, including: (1) the roles (client, authorization server, resource server, resource owner), (2) the authorization code grant flow with a sequence diagram description, (3) common pitfalls. Assume the reader is a junior backend developer. Include a real-world example with a food delivery app.", "category": "q_and_a"},

    # Roleplay
    {"vague": "act as a career counselor", "optimized": "Act as a career counselor with 15 years of experience in the tech industry. I'm a mid-level developer considering a move to management. Ask me 3-5 clarifying questions about my goals, then provide guidance on: skills to develop, potential career paths, and how to evaluate if management is right for me. Be direct but supportive.", "category": "roleplay"},
    {"vague": "be a security auditor", "optimized": "You are a senior security auditor conducting a penetration test review. Analyze the provided network architecture diagram and access control policies. Identify: misconfigurations, privilege escalation paths, and compliance gaps against ISO 27001. Format findings as a professional audit report with severity ratings.", "category": "roleplay"},

    # Summarization
    {"vague": "summarize this article", "optimized": "Summarize the following article in 3-5 bullet points for a busy executive. Focus on: key findings, business implications, and actionable recommendations. Maintain factual accuracy—do not add information not present in the original text.", "category": "summarization"},
    {"vague": "tldr the meeting", "optimized": "Summarize this meeting transcript into: (1) a 2-sentence executive summary, (2) key decisions made, (3) action items with owners and deadlines, (4) unresolved questions. Format as a structured document suitable for email distribution to stakeholders who weren't present.", "category": "summarization"},

    # Brainstorming
    {"vague": "give me app ideas", "optimized": "Generate 10 mobile app ideas for the Finnish market that solve real problems in daily life. For each idea, provide: (1) the problem it solves, (2) target demographic, (3) key differentiator from existing solutions, (4) estimated development complexity (Low/Medium/High). Prioritize ideas that can be built by a solo developer in under 3 months.", "category": "brainstorming"},
    {"vague": "ideas for team building", "optimized": "Suggest 8 team-building activities for a remote team of 12 developers who are introverted and dislike forced socializing. Each activity should: take under 60 minutes, require no special equipment, and focus on collaboration rather than competition. Include a brief description and estimated engagement level (1-5).", "category": "brainstorming"},

    # Instruction
    {"vague": "how to set up docker", "optimized": "Write a step-by-step guide for installing Docker Desktop on Ubuntu 22.04 and running a first container. Target audience: developers who have never used containers. Include: prerequisites, verification steps after each command, common errors and fixes, and next steps for learning Docker Compose. Format as numbered steps with code blocks.", "category": "instruction"},
    {"vague": "guide for code review", "optimized": "Create a code review checklist for a Python backend team. Cover: security, performance, error handling, testing, naming conventions, and documentation. For each category, list 3-5 specific checks. Format as a markdown checklist that can be pasted into a PR template.", "category": "instruction"},

    # Editing
    {"vague": "improve this email", "optimized": "Rewrite the following email to be more professional and concise. Reduce word count by 30%, remove passive voice, add a clear call-to-action, and ensure the tone is direct but polite. Preserve all factual content and the original intent.", "category": "editing"},
    {"vague": "make the README better", "optimized": "Improve this GitHub README by: (1) adding a clear project description in the first 3 lines, (2) adding installation steps with code blocks, (3) adding a usage example, (4) adding a contributing section. Keep the existing content but restructure for clarity. Use standard README conventions.", "category": "editing"},
]


def main():
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save seed data first
    seed_path = output_dir / "seed_data.jsonl"
    with open(seed_path, "w") as f:
        for item in SEED_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(SEED_DATA)} seed examples to {seed_path}")

    # Generate category-specific data using Ollama Cloud
    all_generated = []
    for category, config in CATEGORIES.items():
        remaining = config["count"] - len([s for s in SEED_DATA if s["category"] == category])
        print(f"\n--- {category} ---")
        print(f"Seed examples: {len([s for s in SEED_DATA if s['category'] == category])}")
        print(f"Target: {config['count']}, Need to generate: {remaining}")

        if remaining <= 0:
            print("  Already at target, skipping generation")
            continue

        batch_size = 10
        batches = (remaining + batch_size - 1) // batch_size

        for batch_num in range(batches):
            count = min(batch_size, remaining - batch_num * batch_size)
            print(f"  Generating batch {batch_num + 1}/{batches} ({count} pairs)...")

            pairs = generate_with_ollama(category, config, count)

            if pairs:
                for p in pairs:
                    if "vague" in p and "optimized" in p:
                        p["category"] = category
                        all_generated.append(p)
                print(f"    Got {len(pairs)} pairs")
            else:
                print(f"    Ollama failed, using manual fallback")
                manual = generate_manual_pairs(category, config)
                all_generated.extend(manual[:count])
                print(f"    Got {min(len(manual), count)} manual pairs")

            time.sleep(2)  # Rate limiting

    # Save generated data
    gen_path = output_dir / "generated_data.jsonl"
    with open(gen_path, "w") as f:
        for item in all_generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_generated)} generated examples to {gen_path}")
    total = len(SEED_DATA) + len(all_generated)
    print(f"Total dataset: {total} examples")


if __name__ == "__main__":
    main()