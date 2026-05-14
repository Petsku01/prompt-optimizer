#!/usr/bin/env python3
"""
Generate prompt optimization pairs programmatically.
No LLM needed - uses templates, variations, and combinatorial expansion.
Produces ~2000 training pairs across 10 categories.
"""

import json
import random
import hashlib
from pathlib import Path
from itertools import product

random.seed(42)

# ============================================================================
# CATEGORY DATA - each category has vague patterns and optimization templates
# ============================================================================

VAGUE_WRITING_STARTS = [
    "write about", "tell me about", "create a", "compose a", "draft a",
    "make a", "put together a", "come up with a", "give me a", "help me write a",
    "I need a", "can you write", "write me", "generate a", "produce a",
    "craft a", "develop a", "construct a", "formulate a", "build a",
]

OPTIMIZATION_DIMS = {
    "length": [
        "in 300 words", "in 500 words", "in 1000 words", "briefly (150-200 words)",
        "in detail (1500+ words)", "in 2-3 paragraphs", "in exactly 5 paragraphs",
        "as a short response (under 100 words)", "comprehensively",
    ],
    "tone": [
        "in a professional tone", "in a conversational tone", "in an academic tone",
        "in a friendly and approachable tone", "in a formal tone", "in a persuasive tone",
        "in a neutral and objective tone", "in a humorous tone where appropriate",
        "in an authoritative tone", "in a reflective and thoughtful tone",
    ],
    "audience": [
        "for a general audience", "for beginners with no prior knowledge",
        "for industry professionals", "for college students", "for executives",
        "for a technical audience", "for middle school students", "for stakeholders",
        "for developers", "for policy makers", "for a non-English speaking audience",
    ],
    "format": [
        "as a structured essay with headings", "as bullet points", "as numbered steps",
        "as a blog post with H2 headers", "as a formal report", "in markdown format",
        "as an FAQ", "as a comparison table", "in a narrative style", "as a cheat sheet",
    ],
    "specificity": [
        "Focus on {topic_specific}.", "Cover {sub_topics}.",
        "Include real-world examples.", "Support claims with data.",
        "Address counterarguments.", "Provide actionable takeaways.",
        "Include case studies.", "Compare and contrast different approaches.",
    ],
}


def generate_writing_pairs(count: int = 300) -> list[dict]:
    """Generate writing-category prompt pairs."""
    pairs = []
    topics = [
        "artificial intelligence", "climate change", "space exploration",
        "remote work", "mental health", "sustainable living", "digital privacy",
        "creativity", "education reform", "ocean conservation", "music therapy",
        "urban gardening", "cryptocurrency", "social media impact",
        "cultural diversity", "renewable energy", "vintage fashion",
        "street food", "minimalism", "public libraries", "board games",
        "podcasting", "amateur astronomy", "fermentation", "urban cycling",
    ]

    subtopic_map = {
        "artificial intelligence": ["machine learning applications", "AI ethics", "automation impact", "AI in healthcare"],
        "climate change": ["carbon reduction strategies", "renewable solutions", "policy impacts", "individual actions"],
        "space exploration": ["Mars mission challenges", "satellite technology", "commercial spaceflight", "scientific discoveries"],
        "remote work": ["productivity tips", "team communication", "work-life balance", "digital tools"],
        "mental health": ["stress management", "workplace wellness", "therapy options", "self-care strategies"],
        "sustainable living": ["zero waste tips", "energy efficiency", "ethical consumption", "community initiatives"],
        "digital privacy": ["data protection", "encryption basics", "social media risks", "GDPR compliance"],
        "creativity": ["overcoming blocks", "daily practices", "collaborative methods", "divergent thinking"],
        "education reform": ["technology in classrooms", "assessment methods", "equity issues", "curriculum innovation"],
        "ocean conservation": ["plastic pollution", "marine biodiversity", "sustainable fishing", "coral reef protection"],
    }

    formats = ["essay", "blog post", "article", "opinion piece", "guide", "review", "story", "analysis"]
    for topic in topics:
        subs = subtopic_map.get(topic, ["key aspects", "current trends", "future outlook", "practical applications"])
        for _ in range(count // len(topics)):
            starts = random.choice(VAGUE_WRITING_STARTS)
            fmt = random.choice(formats)
            vague = f"{starts} {fmt} about {topic}"

            dims = random.sample(["length", "tone", "audience", "format", "specificity"], k=random.randint(2, 3))
            opt_parts = [f"Write a {random.choice(OPTIMIZATION_DIMS['length']).replace('in ', '')} {fmt} about {topic}."]

            for dim in dims:
                if dim == "tone":
                    opt_parts.append(random.choice(OPTIMIZATION_DIMS["tone"]))
                elif dim == "audience":
                    opt_parts.append(random.choice(OPTIMIZATION_DIMS["audience"]))
                elif dim == "format":
                    fmt_spec = random.choice(OPTIMIZATION_DIMS["format"])
                    opt_parts[0] = fmt_spec.capitalize().replace("As a", "Format as a") + f" about {topic}."
                elif dim == "specificity":
                    sub = random.choice(subs)
                    opt_parts.append(f"Focus on {sub}.")
                elif dim == "length" and "length" not in dims[:1]:
                    opt_parts.append(random.choice(OPTIMIZATION_DIMS["length"]))

            optimized = " ".join(opt_parts)
            pairs.append({"vague": vague, "optimized": optimized, "category": "writing"})

    return pairs[:count]


def generate_coding_pairs(count: int = 300) -> list[dict]:
    """Generate coding-category prompt pairs."""
    pairs = []
    tasks = [
        ("web scraper", "Write a Python web scraper using requests and BeautifulSoup that extracts {specifics}. Include retry logic with exponential backoff, rate limiting, proper error handling, and output as JSON.", ["article titles and dates from news sites", "product prices from e-commerce sites", "contact info from directory pages"]),
        ("REST API", "Design and implement a RESTful API using FastAPI for {domain}. Include: CRUD endpoints, JWT authentication, Pydantic models for input validation, proper HTTP status codes, and OpenAPI documentation.", ["task management system", "book library catalog", "event booking platform"]),
        ("authentication system", "Implement a secure authentication system in {language} with: user registration/login, password hashing (bcrypt), JWT token management, refresh token rotation, and rate limiting on login attempts. Include type hints.", ["Python with FastAPI", "Node.js with Express", "TypeScript"]),
        ("database migration tool", "Create a database migration tool that {specifics}. Support: up/down migrations, transaction safety, rollback on failure, and migration history tracking. Include unit tests.", ["handles schema changes for PostgreSQL", "manages data transformations between versions", "syncs schema across environments"]),
        ("caching layer", "Implement a caching layer with {specifics}. Include: TTL management, cache invalidation strategies, thread-safe operations, and graceful fallback when cache is empty.", ["LRU eviction policy and Redis backend", "In-memory store with configurable size limits", "Write-through cache with background sync"]),
        ("rate limiter", "Build a distributed rate limiter that {specifics}. Implement: sliding window algorithm, configurable limits per client, HTTP headers for remaining quota, and graceful 429 responses.", ["works with Redis for shared state across instances", "uses token bucket algorithm with configurable burst", "supports both IP and user-level limits"]),
        ("data validation library", "Create a data validation library that {specifics}. Support: nested schemas, custom validators, type coercion, detailed error messages, and both sync and async validation.", ["validates API request payloads against schemas", "validates configuration files at startup", "validates and transforms database query results"]),
        ("file processor", "Write a file processing pipeline that {specifics}. Include: progress tracking, error recovery, parallel processing where safe, and structured logging.", ["batch processes CSV files into a database", "converts images between formats with metadata preservation", "parses and validates large log files"]),
        ("notification system", "Implement a multi-channel notification system supporting {specifics}. Include: template management, delivery status tracking, retry logic, and preference management.", ["email, SMS, and push notifications", "scheduled and event-triggered notifications", "user-configurable notification preferences"]),
        ("error handling framework", "Build a centralized error handling framework for {specifics}. Include: custom exception hierarchy, structured error responses, logging integration, and automatic error reporting.", ["Python backend services", "microservices architecture", "REST API applications"]),
    ]

    vague_starts = ["fix my", "write a", "help me with", "create a", "implement a", "build a", "make a", "debug this", "optimize my", "refactor my"]

    for i in range(count):
        task_name, opt_template, specifics_list = random.choice(tasks)
        specifics = random.choice(specifics_list)
        vague = f"{random.choice(vague_starts)} {task_name}"
        optimized = opt_template.format(specifics=specifics, domain=specifics, language=random.choice(["Python", "TypeScript", "Go"]))
        pairs.append({"vague": vague, "optimized": optimized, "category": "coding"})

    return pairs[:count]


def generate_analysis_pairs(count: int = 250) -> list[dict]:
    """Generate analysis-category prompt pairs."""
    pairs = []
    analysis_templates = [
        ("analyze {topic}", "Analyze {topic} using a {framework} approach. Focus on: {focus_areas}. Present your findings as a {format} for {audience}. Include specific data points and actionable recommendations."),
        ("compare {topic}", "Compare {options} for {context}. Evaluate on: {criteria}. Provide a recommendation matrix with weighted scores, and give a final recommendation for {audience} with justification."),
        ("evaluate {topic}", "Conduct a thorough evaluation of {topic} covering: {aspects}. Prioritize findings by impact level (Critical/High/Medium/Low). For each finding, provide evidence, implications, and 2-3 specific remediation steps."),
        ("what do you think about {topic}", "Provide a balanced analysis of {topic} for {audience}. Cover: (1) current state and key trends, (2) opportunities and risks, (3) short-term and long-term outlook. Support each point with specific examples or data. Format as a structured brief."),
        ("give me insights on {topic}", "Generate actionable insights on {topic} based on {perspective}. For each insight: (1) state the observation, (2) explain the implication, (3) suggest a specific action. Prioritize by potential impact. Format as a numbered list with sub-headers."),
    ]

    topics = [
        ("quarterly sales data", "SWOT", ["revenue trends", "customer segments", "product performance"], "executive summary", "executives"),
        ("user engagement metrics", "funnel analysis", ["drop-off points", "feature adoption", "retention rates"], "dashboard report", "product managers"),
        ("market trends in SaaS", "Porter's Five Forces", ["competition intensity", "buyer power", "threat of substitutes"], "competitive brief", "strategy team"),
        ("our security posture", "risk assessment", ["vulnerabilities", "compliance gaps", "incident history"], "audit report", "CISO"),
        ("supply chain efficiency", "bottleneck analysis", ["lead times", "inventory levels", "supplier reliability"], "operations review", "supply chain team"),
        ("customer feedback patterns", "thematic analysis", ["pain points", "feature requests", "satisfaction drivers"], "insights memo", "product team"),
        ("team productivity metrics", "capacity planning", ["velocity trends", "blockers", "skill distribution"], "team retrospective", "engineering managers"),
        ("investment opportunities in AI", "cost-benefit", ["market size", "technical maturity", "regulatory landscape"], "investment thesis", "investment committee"),
        ("technology adoption rates", "diffusion of innovations", ["early adopters", "chasm crossing", "network effects"], "market study", "product strategy"),
        ("energy consumption patterns", "time-series decomposition", ["seasonal trends", "anomalies", "efficiency opportunities"], "sustainability report", "facilities team"),
    ]

    for i in range(count):
        tmpl_vague, tmpl_opt = random.choice(analysis_templates)
        topic_data = random.choice(topics)

        vague = tmpl_vague.format(topic=topic_data[0])
        optimized = tmpl_opt.format(
            topic=topic_data[0],
            framework=topic_data[1],
            focus_areas=", ".join(topic_data[2]),
            format=topic_data[3],
            audience=topic_data[4],
            options="Option A vs Option B",
            context="a mid-size tech company",
            criteria="cost, scalability, maturity, and team expertise",
            aspects=", ".join(topic_data[2]),
            perspective="a strategic lens",
        )

        pairs.append({"vague": vague, "optimized": optimized, "category": "analysis"})

    return pairs[:count]


def generate_translation_pairs(count: int = 200) -> list[dict]:
    """Generate translation-category prompt pairs."""
    pairs = []
    languages = [
        ("Finnish", "Finnish"),
        ("English", "English"),
        ("Swedish", "Swedish"),
        ("German", "German"),
        ("French", "French"),
        ("Spanish", "Spanish"),
        ("Japanese", "Japanese"),
    ]

    contexts = [
        ("business email", "formal register suitable for professional communication"),
        ("product description", "marketing-friendly tone, localized for the target market"),
        ("technical documentation", "precise terminology, maintain technical accuracy"),
        ("legal document", "formal legal register, preserve legal intent exactly"),
        ("casual chat message", "casual conversational tone with colloquial expressions"),
        ("academic paper abstract", "formal academic register, discipline-specific terminology"),
        ("website UI copy", "concise and user-friendly, maintain consistent terminology"),
        ("social media post", "engaging and informal tone, use platform-appropriate language"),
    ]

    instructions = [
        "Preserve the original meaning exactly\u2014do not add or remove information.",
        "Keep technical terms in the source language where no established translation exists.",
        "Adapt idioms and cultural references to be natural in the target language.",
        "Maintain the same level of formality as the source text.",
        "Use terminology consistent with {domain} industry standards.",
    ]

    for i in range(count):
        target_lang = random.choice(languages)
        context = random.choice(contexts)
        instruction = random.choice(instructions)

        vague = f"translate this to {target_lang[1]}"

        parts = [f"Translate the following {context[0]} to {target_lang[1]}."]
        parts.append(f"Use a {context[1]}.")
        parts.append(instruction.format(domain=random.choice(["technology", "finance", "healthcare", "marketing", "software"])))
        if random.random() > 0.5:
            parts.append("Provide the translation only, without explanations.")

        optimized = " ".join(parts)
        pairs.append({"vague": vague, "optimized": optimized, "category": "translation"})

    return pairs[:count]


def generate_qa_pairs(count: int = 250) -> list[dict]:
    """Generate Q&A-category prompt pairs."""
    pairs = []

    topics = [
        ("Kubernetes", "Explain Kubernetes to someone who understands containers but not orchestration. Cover pods, services, deployments, and when you'd actually need it. Use analogies from software development. Format as a structured explanation with headings."),
        ("blockchain", "Explain blockchain technology focusing on how consensus mechanisms work. Compare Proof of Work and Proof of Stake. Include: real-world examples, common misconceptions, and 3 practical use cases beyond cryptocurrency. Written for a business audience."),
        ("machine learning", "Explain machine learning for a developer who knows Python but no ML. Cover: supervised vs unsupervised learning, model training workflow, overfitting, and when to use ML vs rules. Include a concrete example with a housing price prediction scenario."),
        ("DNS", "Explain how DNS works step by step, from typing a URL to the page loading. Include: recursive and iterative queries, caching, TTL, and common DNS records (A, CNAME, MX, TXT). Use a real-world analogy. Format as a labeled diagram description followed by explanation."),
        ("OAuth 2.0", "Explain OAuth 2.0 authorization flow step by step. Include: roles (client, auth server, resource server, owner), authorization code grant flow, common pitfalls, and a concrete example using a food delivery app. Target audience: junior backend developer."),
        ("Git branching", "Explain Git branching strategies for a team of 5 developers. Cover: Git Flow, GitHub Flow, and Trunk-Based Development. Compare pros/cons of each. Recommend one for a small startup and justify. Include diagrams described in text."),
        ("microservices", "Explain microservices architecture for someone familiar with monoliths. Cover: service boundaries, communication patterns (sync vs async), data management, and when microservices are NOT the right choice. Use concrete examples from e-commerce."),
        ("load balancing", "Explain load balancing concepts for a junior DevOps engineer. Cover: Layer 4 vs Layer 7, algorithms (round-robin, least connections, IP hash), health checks, and session persistence. Include a practical Nginx configuration example."),
        ("SQL indexes", "Explain database indexing for a developer who writes SQL but doesn't understand performance. Cover: B-tree indexes, when to create indexes, composite indexes, covering indexes, and how to read EXPLAIN plans. Use a practical example with a 1M row users table."),
        ("CI/CD pipelines", "Explain CI/CD pipeline design for a team moving from manual deployment. Cover: build, test, deploy stages; environment promotion (dev/staging/prod); rollback strategies; and pipeline-as-code. Include a concrete GitHub Actions workflow example."),
    ]

    for i in range(count):
        topic_name, opt_content = random.choice(topics)
        vague_options = [
            f"what is {topic_name}",
            f"how does {topic_name} work",
            f"explain {topic_name}",
            f"tell me about {topic_name}",
            f"basics of {topic_name}",
        ]
        vague = random.choice(vague_options)
        pairs.append({"vague": vague, "optimized": opt_content, "category": "q_and_a"})

    # Add variations with different optimization dimensions
    for i in range(count - len(pairs)):
        topic_name, base_opt = random.choice(topics)
        vague = random.choice([f"what is {topic_name}", f"explain {topic_name}", f"how does {topic_name} work"])
        # Add random specificity
        extras = [
            " Include code examples where relevant.",
            " Provide a comparison table summarizing key points.",
            " End with 3 Frequently Asked Questions and answers.",
            " Use analogies from everyday life to explain technical concepts.",
            " Format as a numbered tutorial with progressive complexity.",
        ]
        optimized = base_opt + random.choice(extras)
        pairs.append({"vague": vague, "optimized": optimized, "category": "q_and_a"})

    return pairs[:count]


def generate_roleplay_pairs(count: int = 150) -> list[dict]:
    """Generate roleplay-category prompt pairs."""
    pairs = []

    roles = [
        ("career counselor", "Act as a career counselor with 15 years of experience in the tech industry. Your role: ask 2-3 clarifying questions first, then provide tailored advice on skills to develop, career paths, and decision frameworks. Be direct but supportive. Use real industry examples."),
        ("security auditor", "You are a senior security auditor conducting a penetration test review. Analyze provided materials for: misconfigurations, privilege escalation paths, and compliance gaps against ISO 27001. Format findings as a professional audit report with severity ratings (Critical/High/Medium/Low). Include remediation steps for each finding."),
        ("fitness coach", "Act as a certified personal trainer with expertise in beginner programming. Ask about: current fitness level, available equipment, time commitment, and any injuries. Then create a 4-week progressive plan with specific exercises, sets, reps, and rest periods. Include warm-up and cool-down routines."),
        ("UX researcher", "Act as a senior UX researcher conducting a usability review. Analyze the given interface for: information hierarchy, cognitive load, accessibility (WCAG 2.1), and user flow friction. Prioritize findings by severity and provide specific redesign recommendations with reasoning."),
        ("financial advisor", "Act as a fee-only financial planner. Review the provided financial situation and provide guidance on: budgeting, emergency fund targets, debt payment strategies, and investment allocation. Ask clarifying questions about goals, risk tolerance, and timeline before giving recommendations. Use specific numbers and percentages."),
        ("language tutor", "Act as a patient language tutor for Finnish. Assess the learner's current level through 3-5 test questions. Then provide a personalized lesson covering: vocabulary (10 words), grammar rule, and 2 practice exercises. Correct errors gently with explanations. Mix Finnish and English as appropriate."),
        ("startup mentor", "Act as a startup mentor who has advised 50+ early-stage companies. Review the provided pitch/idea and evaluate: market size, competitive landscape, business model viability, and go-to-market strategy. Be honest about weaknesses. Suggest 3 specific next actions with timelines."),
        ("project manager", "Act as an experienced project manager for a software team. Analyze the described project situation for: scope creep risks, resource bottlenecks, timeline feasibility, and stakeholder alignment. Provide a structured action plan with priorities, owners, and deadlines."),
    ]

    for i in range(count):
        role_name, optimized = random.choice(roles)
        vague_options = [f"act as a {role_name}", f"pretend you are a {role_name}", f"be a {role_name}", f"roleplay as a {role_name}"]
        vague = random.choice(vague_options)
        pairs.append({"vague": vague, "optimized": optimized, "category": "roleplay"})

    return pairs[:count]


def generate_summarization_pairs(count: int = 200) -> list[dict]:
    """Generate summarization-category prompt pairs."""
    pairs = []

    templates = [
        ("summarize this", "Summarize the following text in 3-5 bullet points for a busy executive. Focus on: key findings, business implications, and actionable recommendations. Maintain factual accuracy\u2014do not add information not present in the original text. Use clear, direct language."),
        ("tldr", "Provide a structured summary of the following content organized into: (1) a 2-sentence executive summary, (2) key decisions or conclusions, (3) action items with suggested owners, (4) unresolved questions or risks. Format for email distribution to stakeholders who weren't present."),
        ("make a summary", "Summarize this document in {length} for {audience}. Highlight: main arguments, supporting evidence, and conclusions. Use {format}. Preserve all critical data points and quotes. Omit filler and repetition. End with a one-sentence takeaway."),
        ("shorten this", "Condense the following text to {length} while preserving: all key facts, numerical data, proper nouns, and conclusions. Remove: redundancy, filler phrases, and tangential points. Format as {format}. Target audience: {audience}."),
        ("give me the key points", "Extract the key points from the following content. For each point, provide: (1) the main claim, (2) supporting evidence, (3) implications. Limit to the 5 most important points. Format as a numbered list. End with a one-paragraph synthesis."),
        ("recap the meeting", "Summarize this meeting transcript into: (1) a 2-sentence executive summary, (2) key decisions made with rationale, (3) action items with owners and deadlines, (4) unresolved questions for next meeting. Format as a structured document suitable for email distribution."),
        ("extract main ideas", "Extract and organize the main ideas from the following text. Group related concepts together. For each group, provide a heading, 2-3 supporting details, and one implication. Format as a structured outline. Priority order: most impactful ideas first."),
    ]

    lengths = ["200 words", "300 words", "500 words", "one page", "half a page", "3-5 sentences", "a single paragraph"]
    audiences = ["executives", "technical team leads", "general audience", "stakeholders", "new team members", "board members"]
    formats = ["bullet points", "numbered list", "structured sections with headers", "a comparison table", "a decision matrix", "narrative prose"]

    for i in range(count):
        vague, opt_template = random.choice(templates)
        optimized = opt_template.format(
            length=random.choice(lengths),
            audience=random.choice(audiences),
            format=random.choice(formats),
        )
        pairs.append({"vague": vague, "optimized": optimized, "category": "summarization"})

    return pairs[:count]


def generate_brainstorming_pairs(count: int = 150) -> list[dict]:
    """Generate brainstorming-category prompt pairs."""
    pairs = []

    templates = [
        ("give me ideas for {topic}", "Generate {count} creative ideas for {topic}. For each idea, provide: (1) a concise name, (2) the problem it solves, (3) target audience or demographic, (4) key differentiator from existing solutions, (5) estimated complexity to implement ({complexity_scale}). Prioritize ideas that {priority}."),
        ("brainstorm {topic}", "Brainstorm {count} approaches to {topic}. Categorize each as: Quick Win (< 1 week), Medium Effort (1-4 weeks), or Long Term (1+ month). For each approach, describe: what it involves, expected impact (1-10), and one potential risk. Format as a prioritized list."),
        ("help me think about {topic}", "Help me explore {topic} from multiple angles. Provide: (1) 5 conventional approaches, (2) 3 unconventional/creative approaches, (3) 2 approaches that involve other disciplines or fields, (4) 1 wild card idea. For each, assess feasibility (Low/Medium/High) and potential impact (Low/Medium/High)."),
        ("what could I do with {topic}", "Suggest {count} projects or experiments related to {topic}. For each: name, one-paragraph description, required resources, timeline estimate, and success metric. Rank by effort-to-impact ratio, best first."),
        ("creative ideas for {topic}", "Generate {count} innovative ideas for {topic} using the following creative lenses: (1) What if cost was no object? (2) What if we had to launch in 48 hours? (3) What if we could only use existing resources? (4) What if we targeted an unexpected audience? For each lens, provide 2-3 specific ideas."),
    ]

    topics = [
        ("a mobile app", "5-8", "Low/Medium/High", "can be built by a small team in under 3 months"),
        ("team building", "6-10", "Easy/Medium/Hard", "work well for remote teams"),
        ("content marketing", "10-12", "Low/Medium/High effort", "can be started with minimal budget"),
        ("cost reduction", "8-10", "-/-/-", "save at least 20% without quality loss"),
        ("employee engagement", "6-8", "Low/Medium/High", "are measurable and sustainable"),
        ("side hustles", "8-12", "-/-/-", "can generate income within 30 days"),
        ("sustainability initiatives", "6-8", "Easy/Medium/Hard", "have both environmental and business benefits"),
        ("learning projects", "8-10", "Beginner/Intermediate/Advanced", "build portfolio-worthy skills"),
        ("community events", "5-8", "-/-/-", "cost under $500 to organize"),
        ("portfolio projects", "10-12", "-/-/-", "demonstrate real-world skills to employers"),
    ]

    for i in range(count):
        topic_data = random.choice(topics)
        vague_tmpl, opt_tmpl = random.choice(templates)
        vague = vague_tmpl.format(topic=topic_data[0])
        optimized = opt_tmpl.format(
            topic=topic_data[0],
            count=random.choice(["6", "8", "10", "12"]),
            complexity_scale=topic_data[2],
            priority=topic_data[3],
        )
        pairs.append({"vague": vague, "optimized": optimized, "category": "brainstorming"})

    return pairs[:count]


def generate_instruction_pairs(count: int = 100) -> list[dict]:
    """Generate instruction-category prompt pairs."""
    pairs = []

    templates = [
        ("how to {topic}", "Write a step-by-step guide for {topic}. Target audience: {audience}. Include: prerequisites, each step as a numbered instruction with code blocks where applicable, verification steps after critical actions, common errors and their fixes, and next steps for further learning. Format as a markdown tutorial with headers."),
        ("teach me {topic}", "Create a structured learning guide for {topic}. Structure: (1) prerequisites and assumed knowledge, (2) core concepts explained with examples, (3) hands-on exercises progressing from easy to hard, (4) self-assessment questions, (5) resources for deeper learning. Assume the reader is {audience}."),
        ("guide for {topic}", "Write a practical guide for {topic} covering: setup and prerequisites, step-by-step implementation, testing and validation, troubleshooting common issues, and best practices. Include code examples in {language}. Target audience: {audience}. Estimated completion time: {time}."),
        ("tutorial on {topic}", "Create an interactive tutorial for {topic}. Each section should: explain the concept, show a working code example, provide a practice exercise, and include a 'common mistakes' callout box. Assume the reader knows {prerequisites}. Use {language} for all examples. Total length: suitable for {time} session."),
    ]

    guides = [
        ("setting up Docker", "developers new to containers", "Python", "2 hours", "basic command line"),
        ("writing unit tests", "junior Python developers", "Python", "1 hour", "Python basics"),
        ("deploying to AWS", "developers with no cloud experience", "YAML/CLI", "3 hours", "basic web development"),
        ("building a REST API", "intermediate Python developers", "Python", "2 hours", "HTTP basics and Python"),
        ("configuring a firewall", "junior sysadmins", "bash", "1.5 hours", "Linux command line"),
        ("implementing CI/CD", "developers new to DevOps", "YAML", "2 hours", "Git and basic scripting"),
        ("code review best practices", "team leads and senior developers", "none", "30 min", "2+ years programming"),
        ("database migration", "backend developers", "SQL/Python", "1.5 hours", "SQL and ORM basics"),
        ("setting up monitoring", "DevOps engineers", "YAML/Python", "2 hours", "Linux and networking basics"),
        ("creating a GitHub Action", "developers new to CI/CD", "YAML", "1 hour", "Git and GitHub basics"),
        ("writing documentation", "all developers", "Markdown", "45 min", "None"),
        ("implementing OAuth 2.0", "intermediate backend developers", "Python/Node", "2 hours", "HTTP and API basics"),
    ]

    for i in range(count):
        guide = random.choice(guides)
        vague_tmpl, opt_tmpl = random.choice(templates)
        vague = vague_tmpl.format(topic=guide[0])
        optimized = opt_tmpl.format(
            topic=guide[0],
            audience=guide[1],
            language=guide[2],
            time=guide[3],
            prerequisites=guide[4],
        )
        pairs.append({"vague": vague, "optimized": optimized, "category": "instruction"})

    return pairs[:count]


def generate_editing_pairs(count: int = 100) -> list[dict]:
    """Generate editing-category prompt pairs."""
    pairs = []

    templates = [
        ("improve this {doc_type}", "Improve the following {doc_type} by: (1) {action_1}, (2) {action_2}, (3) {action_3}. Target audience: {audience}. Maintain {preservation}. Style: {style}. Return only the improved version with changes highlighted in [brackets]."),
        ("rewrite {doc_type}", "Rewrite the following {doc_type} to be more {goal}. Specific requirements: {details}. Preserve all factual content and the original intent. The target audience is {audience}. Format the output as {format}."),
        ("make {doc_type} better", "Enhance this {doc_type} focusing on: clarity, conciseness, and impact. Reduce word count by approximately {reduction}%. Remove: passive voice, unnecessary adverbs, and filler phrases. Add: a clear call-to-action and stronger transitions. Target tone: {tone}."),
        ("polish {doc_type}", "Polish the following {doc_type} for {audience}. Apply these changes: {changes}. Ensure the final version is {quality}. Do not alter the core message or add information not present in the original. Output the revised text only."),
        ("fix the writing in {doc_type}", "Fix the writing quality of the following {doc_type}. Address: {issues}. Improve: {improvements}. The final version should read like it was written by a professional {writer_type}. Preserve the original meaning exactly."),
    ]

    doc_types = ["email", "cover letter", "blog post", "marketing copy", "product description",
                 "social media post", "report", "proposal", "README", "job posting"]

    for i in range(count):
        doc_type = random.choice(doc_types)
        vague_tmpl, opt_tmpl = random.choice(templates)
        vague = vague_tmpl.format(doc_type=doc_type)

        actions = {
            "email": [("removing unnecessary formality", "adding a clear call-to-action", "tightening the subject line"),
                      ("cutting word count by 30%", "adding a specific ask", "improving scannability")],
            "cover letter": [("highlighting relevant achievements", "reducing to one page", "matching the job description tone"),
                            ("adding specific metrics", "improving the opening hook", "strengthening the closing paragraph")],
            "blog post": [("adding a compelling introduction", "breaking into shorter paragraphs", "adding data points and examples"),
                          ("improving headline and subheads", "adding a conclusion with takeaways", "removing jargon")],
            "report": [("clarifying the executive summary", "adding data visualizations descriptions", "improving recommendations"),
                       ("removing redundancy", "strengthening causal claims", "better section transitions")],
        }

        default_actions = [("improving clarity", "removing redundancy", "strengthening transitions"),
                          ("adding specificity", "improving flow", "fixing grammar")]

        doc_actions = actions.get(doc_type, default_actions)
        action_set = random.choice(doc_actions)

        optimized = opt_tmpl.format(
            doc_type=doc_type,
            action_1=action_set[0],
            action_2=action_set[1],
            action_3=action_set[2],
            audience=random.choice(["executives", "team leads", "general audience", "technical professionals", "clients"]),
            preservation="all factual content and the original intent",
            style=random.choice(["professional and direct", "friendly but authoritative", "concise and impactful", "warm and approachable"]),
            goal=random.choice(["professional and concise", "engaging and persuasive", "clear and organized", "compelling and action-oriented"]),
            details=f"reduce by 30%, add {random.choice(['a clear call-to-action', 'specific examples', 'data points'])}",
            format=random.choice(["the same document type with tracked changes", "a clean revised version"]),
            reduction=random.choice(["20", "25", "30"]),
            tone=random.choice(["professional", "conversational but polished", "authoritative", "direct"]),
            changes=", ".join(action_set),
            quality=random.choice(["publishable-ready", "client-ready", "professional-grade"]),
            issues=random.choice(["wordiness, unclear transitions, passive voice", "jargon, lack of structure, weak verbs", "redundancy, vague claims, missing specificity"]),
            improvements="sentence variety, active voice, and stronger vocabulary",
            writer_type=random.choice(["copywriter", "technical writer", "business communicator", "content strategist"]),
        )

        pairs.append({"vague": vague, "optimized": optimized, "category": "editing"})

    return pairs[:count]


# ============================================================================
# Main generation pipeline
# ============================================================================

def main():
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("writing", generate_writing_pairs, 300),
        ("coding", generate_coding_pairs, 300),
        ("analysis", generate_analysis_pairs, 250),
        ("translation", generate_translation_pairs, 200),
        ("q_and_a", generate_qa_pairs, 250),
        ("roleplay", generate_roleplay_pairs, 150),
        ("summarization", generate_summarization_pairs, 200),
        ("brainstorming", generate_brainstorming_pairs, 150),
        ("instruction", generate_instruction_pairs, 100),
        ("editing", generate_editing_pairs, 100),
    ]

    all_pairs = []
    for cat_name, generator, count in generators:
        print(f"Generating {cat_name} ({count} pairs)...")
        pairs = generator(count)
        print(f"  Generated {len(pairs)} pairs")
        all_pairs.extend(pairs)

    # Load and merge seed data
    seed_path = output_dir / "seed_data.jsonl"
    if seed_path.exists():
        seed_pairs = []
        with open(seed_path) as f:
            for line in f:
                if line.strip():
                    seed_pairs.append(json.loads(line))
        print(f"\nLoaded {len(seed_pairs)} seed examples")
        all_pairs.extend(seed_pairs)

    # Deduplicate by vague prompt
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        key = p["vague"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    print(f"\nTotal before dedup: {len(all_pairs)}")
    print(f"After dedup by vague prompt: {len(unique_pairs)}")

    # Category stats
    from collections import Counter
    cat_counts = Counter(p.get("category", "unknown") for p in unique_pairs)
    print("\nCategory distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    # Save merged data
    output_path = output_dir / "generated_data.jsonl"
    with open(output_path, "w") as f:
        for p in unique_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(unique_pairs)} unique pairs to {output_path}")


if __name__ == "__main__":
    main()