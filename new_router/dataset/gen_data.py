

import argparse
import json
import random
import string
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Callable

NOUNS = [
    "cat", "river", "mountain", "algorithm", "telescope", "umbrella", "lantern",
    "compass", "fossil", "glacier", "harbor", "jungle", "kettle", "labyrinth",
    "mirror", "nebula", "oracle", "phantom", "quasar", "rapids", "satellite",
    "tornado", "ultralight", "vortex", "waterfall", "xenolith", "yardarm", "zenith",
    "abbey", "beacon", "canopy", "dagger", "eclipse", "fathom", "gorge", "hallway",
    "iceberg", "javelin", "keystone", "ledger", "mosaic", "nozzle", "outpost",
    "pinnacle", "quarry", "ravine", "scaffold", "threshold", "undertow", "vault",
]

ADJECTIVES = [
    "ancient", "brittle", "cobalt", "drifting", "ethereal", "fragile", "gilded",
    "hollow", "iridescent", "jagged", "kinetic", "luminous", "molten", "nimble",
    "obsidian", "petrified", "quiet", "resonant", "spectral", "turbulent",
    "ultraviolet", "volatile", "weathered", "xenolithic", "yellow", "zealous",
    "abrupt", "blazing", "crystalline", "dormant", "effervescent", "fleeting",
]

VERBS = [
    "calculates", "drifts", "echoes", "fractures", "generates", "hovers",
    "illuminates", "justifies", "kindles", "lingers", "manifests", "navigates",
    "oscillates", "permeates", "quantifies", "refracts", "scatters", "transcends",
    "unravels", "vibrates", "whispers", "yields", "accelerates", "bifurcates",
    "cascades", "diverges", "emanates", "fluctuates", "gravitates", "harmonizes",
]

DOMAINS = [
    "astrophysics", "mycology", "cartography", "numismatics", "dendrology",
    "speleology", "ichthyology", "campanology", "thalassography", "palynology",
    "zymurgy", "orology", "conchology", "glyptography", "hymenopterology",
    "nephology", "pteridology", "sphragistics", "xenobiology", "zoogeography",
    "aerology", "bryology", "cetology", "dosimetry", "ethnomusicology",
    "fulgurology", "geochronology", "helminthology", "iridology", "juvenology",
]

ANIMALS = [
    "narwhal", "axolotl", "quokka", "pangolin", "okapi", "fossa", "saola",
    "kakapo", "blobfish", "tardigrade", "mantis shrimp", "star-nosed mole",
    "aye-aye", "babirusa", "capybara", "dugong", "echidna", "flying squirrel",
    "goblin shark", "hagfish", "irrawaddy dolphin", "jerboa", "kinkajou",
    "leafy sea dragon", "maned wolf", "numbat", "ocelot", "platypus", "quoll",
]

COMPANIES = [
    "Meridian Dynamics", "Helix Ventures", "Stratum Labs", "Apex Forge",
    "Cobalt Systems", "Driftwood Analytics", "Ember Technologies", "Falcon Ridge",
    "Granite Peak", "Harbor Light", "Ironclad Solutions", "Jasper Networks",
    "Keystone Digital", "Lantern Software", "Mosaic Platforms", "Nova Circuit",
    "Obsidian Cloud", "Pinnacle Data", "Quorum Intelligence", "Radiant Logic",
]

MODULE_NAMES = [
    "auth", "billing", "cache", "dashboard", "events", "frontend", "gateway",
    "hooks", "ingestion", "jobs", "kafka", "logging", "metrics", "notifications",
    "orchestrator", "payments", "queue", "redis", "scheduler", "telemetry",
    "uploads", "validator", "webhooks", "xray", "yaml_parser", "zero_trust",
]

# ── Prompt generator functions ────────────────────────────────────────────────
# Each function takes a Random instance and returns a unique (prompt, max_tokens, temp, type)
# Structural category is preserved; surface form is randomized.

def unique_tag(rng: random.Random) -> str:
    """Generate a random 8-character alphanumeric tag to bust KV cache prefix matching."""
    return ''.join(rng.choices(string.ascii_uppercase + string.digits, k=8))


def make_spec_prompts(rng: random.Random, n: int) -> list:
    """Generate n unique speculative-favored prompts."""

    generators = [

        # ── Numeric sequences ─────────────────────────────────────────────
        lambda: (
            f"Print every integer from {rng.randint(1,50)} to {rng.randint(120,200)}, "
            f"one per line, no other text.",
            512, 0.0, "spec"
        ),
        lambda: (
            f"Count down from {rng.randint(150,250)} to {rng.randint(1,10)}, one number per line.",
            512, 0.0, "spec"
        ),
        lambda: (
            f"List every multiple of {rng.randint(2,9)} from "
            f"{rng.randint(2,9)} to {rng.randint(200,400)}, one per line.",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write the multiplication table for {rng.randint(2,15)} "
            f"from x1 to x{rng.randint(25,50)}, one equation per line.",
            400, 0.0, "spec"
        ),
        lambda: (
            f"List the first {rng.randint(60,120)} {'even' if rng.random()<0.5 else 'odd'} "
            f"numbers, one per line.",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write the squares of integers from {rng.randint(1,5)} to "
            f"{rng.randint(60,100)}, one per line as: N^2 = M.",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Print powers of {rng.choice([2,3,5])} from exponent 1 to "
            f"{rng.randint(20,35)}, one per line as: base^exp = result.",
            350, 0.0, "spec"
        ),

        # ── Pure repetition ───────────────────────────────────────────────
        lambda: (
            f"Repeat the word '{rng.choice(NOUNS)}' exactly {rng.randint(80,200)} "
            f"times, one per line. No other text.",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write the phrase '{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}' "
            f"exactly {rng.randint(50,100)} times, each on its own line.",
            400, 0.0, "spec"
        ),
        lambda: (
            "Repeat 'STATUS_" + str(rng.randint(100,999)) + ": " + rng.choice(['OK','PASS','DONE','ACK']) +
            "' exactly " + str(rng.randint(60,120)) + " times, one per line.",
            350, 0.0, "spec"
        ),
        lambda: (
            f"Repeat the IP address '10.{rng.randint(0,255)}.{rng.randint(0,255)}.1' "
            f"exactly {rng.randint(80,150)} times, one per line.",
            350, 0.0, "spec"
        ),
        lambda: (
            f"Write '0x{rng.randint(1000,9999):04X}' exactly {rng.randint(100,200)} "
            f"times separated by spaces.",
            400, 0.0, "spec"
        ),

        # ── Structured data ───────────────────────────────────────────────
        lambda: (
            f"Generate a CSV with columns id,{rng.choice(NOUNS)},score. "
            f"Fill {rng.randint(60,100)} rows with sequential data starting at id={rng.randint(1000,9999)}.",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write a JSON array of {rng.randint(30,60)} objects, each with: "
            f"id (int), {rng.choice(NOUNS)} (string '{rng.choice(ADJECTIVES)}'), active (bool true).",
            400, 0.0, "spec"
        ),
        lambda: (
            "Repeat this JSON " + str(rng.randint(30,60)) + " times on separate lines: "
            + '{"node": "worker-' + f'{rng.randint(1,99):02d}' + '", "status": "running", "cpu": ' + str(rng.randint(10,90)) + '}',
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write a YAML list of {rng.randint(40,80)} items: "
            f"- {rng.choice(NOUNS)}_N: {rng.choice(ADJECTIVES)}_value_N where N increments.",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Generate {rng.randint(40,70)} lines of nginx access log entries. "
            f"Use IP range 172.{rng.randint(16,31)}.x.x with sequential request IDs "
            f"starting at {rng.randint(10000,99999)}.",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write a .env file with {rng.randint(30,60)} variables: "
            f"{rng.choice(NOUNS).upper()}_VAR_N={rng.choice(ADJECTIVES)}_value_N.",
            350, 0.0, "spec"
        ),
        lambda: (
            f"Generate {rng.randint(30,60)} /etc/hosts entries: "
            f"192.168.{rng.randint(1,254)}.N  {rng.choice(NOUNS)}-N.{rng.choice(['local','internal','corp'])}",
            300, 0.0, "spec"
        ),
        lambda: (
            f"Generate a markdown table with columns: ID, {rng.choice(NOUNS).title()}, "
            f"{rng.choice(ADJECTIVES).title()}_Score. Fill {rng.randint(25,50)} rows sequentially.",
            400, 0.0, "spec"
        ),

        # ── Repetitive code ───────────────────────────────────────────────
        lambda: (
            f"Write {rng.randint(40,80)} lines of Python: "
            f"print('Processing {rng.choice(NOUNS)} N of {rng.randint(50,100)}') "
            f"where N increments from 1.",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write {rng.randint(30,60)} Python assert statements: "
            f"assert {rng.choice(NOUNS)}_N == expected_N, "
            f"'Test {rng.randint(100,999)}-N failed'",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write {rng.randint(30,50)} SQL INSERT INTO {rng.choice(NOUNS)}s "
            f"(id, name, value) VALUES (N, '{rng.choice(ADJECTIVES)}_N', N*{rng.randint(2,20)});",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write a bash for loop from {rng.randint(1,10)} to {rng.randint(80,150)} "
            f"that echoes 'Processing {rng.choice(NOUNS)} $i of {rng.randint(80,150)}'.",
            350, 0.0, "spec"
        ),
        lambda: (
            f"Write {rng.randint(40,70)} JavaScript lines: "
            f"console.log('{rng.choice(NOUNS).upper()}_' + N + ': ' + result_N);",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write {rng.randint(25,50)} HTML list items: "
            f"<li id='{rng.choice(NOUNS)}-N'>{rng.choice(ADJECTIVES).title()} item N: description N</li>",
            350, 0.0, "spec"
        ),
        lambda: (
            f"Write a Python class '{rng.choice(NOUNS).title()}Processor' with "
            f"{rng.randint(8,15)} methods. Each method is named handle_step_N "
            f"and prints its own name and the timestamp {rng.randint(1000000,9999999)}.",
            400, 0.0, "spec"
        ),

        # ── Fixed templates ───────────────────────────────────────────────
        lambda: (
            f"Write {rng.randint(40,70)} lines: "
            + "'" + f"[{rng.choice(['INFO','WARN','DEBUG'])}] Step N: "
            + f"{rng.choice(VERBS).title()} {rng.choice(NOUNS)} N and verify result N.'",
            400, 0.0, "spec"
        ),
        lambda: (
            f"List {rng.randint(50,90)} server hostnames: "
            f"{rng.choice(NOUNS)}-{{001..N}}.{rng.choice(['prod','staging','dev'])}."
            f"{rng.choice(['example','internal','corp'])}.com",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Write {rng.randint(40,70)} git commit messages: "
            f"'fix({rng.choice(MODULE_NAMES)}-N): resolve issue N in "
            f"{rng.choice(MODULE_NAMES)} at rev {rng.randint(1000,9999)}'",
            400, 0.0, "spec"
        ),
        lambda: (
            f"Generate {rng.randint(30,60)} ticket IDs: "
            f"{rng.choice(['PROJ','INFRA','BUG','FEAT'])}-{rng.randint(1000,9999)}-N: "
            f"Fix {rng.choice(NOUNS)} in {rng.choice(MODULE_NAMES)} module",
            350, 0.0, "spec"
        ),
        lambda: (
            f"Write {rng.randint(30,60)} cron entries: "
            f"*/{rng.randint(1,30)} * * * * /usr/bin/{rng.choice(NOUNS)}_job_N.sh "
            f">> /var/log/{rng.choice(NOUNS)}_N.log 2>&1",
            350, 0.0, "spec"
        ),
        lambda: (
            f"List {rng.randint(40,70)} API endpoints: "
            f"/api/v{rng.randint(1,4)}/{rng.choice(NOUNS)}s/N/{rng.choice(NOUNS)}s/N",
            350, 0.0, "spec"
        ),
        lambda: (
            f"Generate {rng.randint(30,60)} test function signatures: "
            f"def test_{rng.choice(NOUNS)}_N_{rng.choice(VERBS).replace(' ','_')}"
            f"_returns_{rng.choice(ADJECTIVES)}():",
            350, 0.0, "spec"
        ),
        lambda: (
            f"Write {rng.randint(40,60)} error log lines: "
            f"'[2024-{rng.randint(1,12):02d}-N] ERROR in {rng.choice(MODULE_NAMES)}: "
            f"{rng.choice(NOUNS).title()} {rng.choice(VERBS)} with code {rng.randint(400,599)}'",
            400, 0.0, "spec"
        ),
    ]

    results = []
    for _ in range(n):
        gen = rng.choice(generators)
        prompt, max_tokens, temp, req_type = gen()
        prompt = f"[{unique_tag(rng)}] {prompt}"
        results.append((prompt, max_tokens, temp, req_type))
    return results


def make_std_prompts(rng: random.Random, n: int) -> list:
    """Generate n unique standard-favored prompts."""

    generators = [

        # ── Creative one-liners ───────────────────────────────────────────
        lambda: (
            f"Give me a one-sentence tagline for a startup called "
            f"'{rng.choice(COMPANIES)}' that sells {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}s.",
            60, 1.0, "std"
        ),
        lambda: (
            f"In one sentence, what would a {rng.choice(ADJECTIVES)} {rng.choice(ANIMALS)} "
            f"say about {rng.choice(DOMAINS)}?",
            60, 1.0, "std"
        ),
        lambda: (
            f"Write a haiku about a {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} "
            f"at {rng.randint(1,4)}am.",
            40, 1.0, "std"
        ),
        lambda: (
            f"Suggest a name for a shop that sells {rng.choice(ADJECTIVES)} "
            f"{rng.choice(NOUNS)}s inside a {rng.choice(NOUNS)}. One name only.",
            25, 1.0, "std"
        ),
        lambda: (
            f"In one sentence, describe the taste of {rng.choice(ADJECTIVES)} "
            f"{rng.choice(NOUNS)}.",
            55, 1.0, "std"
        ),
        lambda: (
            f"What would a {rng.choice(ADJECTIVES)} {rng.choice(ANIMALS)}'s "
            f"midlife crisis look like? One sentence.",
            65, 1.0, "std"
        ),
        lambda: (
            f"Invent a word combining '{rng.choice(NOUNS)}' and '{rng.choice(VERBS)}' "
            f"and define it in one sentence.",
            55, 1.0, "std"
        ),
        lambda: (
            f"Write the worst possible fortune cookie message for someone studying "
            f"{rng.choice(DOMAINS)}.",
            35, 1.0, "std"
        ),
        lambda: (
            f"What is the worst superpower to have while doing {rng.choice(DOMAINS)}? "
            f"One sentence.",
            55, 1.0, "std"
        ),
        lambda: (
            f"Describe the sound of a {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} "
            f"using only {rng.choice(DOMAINS)} metaphors. One sentence.",
            65, 1.0, "std"
        ),
        lambda: (
            f"What would {rng.choice(['Plato','Nietzsche','Kant','Hegel','Wittgenstein'])} "
            f"say about {rng.choice(COMPANIES)}? One sentence.",
            65, 1.0, "std"
        ),
        lambda: (
            f"Invent a phobia name for the fear of {rng.choice(ADJECTIVES)} "
            f"{rng.choice(NOUNS)}s. Give the Latin name and one-sentence definition.",
            75, 1.0, "std"
        ),
        lambda: (
            f"Write a 2-sentence legal disclaimer for a service that rents "
            f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}s by the hour.",
            80, 1.0, "std"
        ),
        lambda: (
            f"Describe {rng.choice(DOMAINS)} using only "
            f"{rng.choice(['cooking','weather','sports','music','architecture'])} metaphors. "
            f"One sentence.",
            75, 1.0, "std"
        ),
        lambda: (
            f"What is the plot of a telenovela set inside a "
            f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}? Two sentences.",
            80, 1.0, "std"
        ),
        lambda: (
            f"Invent a cocktail named after {rng.choice(DOMAINS)}. "
            f"Give the name and a one-sentence description of the taste.",
            65, 1.0, "std"
        ),
        lambda: (
            f"Write the opening line of a nature documentary about "
            f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}s.",
            80, 1.0, "std"
        ),
        lambda: (
            f"What life advice would a {rng.choice(ADJECTIVES)} {rng.choice(ANIMALS)} "
            f"give to a {rng.choice(ADJECTIVES)} {rng.choice(ANIMALS)}? One sentence.",
            65, 1.0, "std"
        ),
        lambda: (
            f"Describe an economy built entirely around {rng.choice(ADJECTIVES)} "
            f"{rng.choice(NOUNS)}s. Two sentences.",
            80, 1.0, "std"
        ),
        lambda: (
            f"Write a 2-sentence weather forecast for a city inside a "
            f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}.",
            75, 1.0, "std"
        ),

        # ── Surreal / abstract ────────────────────────────────────────────
        lambda: (
            f"What color is the concept of {rng.choice(NOUNS)}? One sentence.",
            45, 1.2, "std"
        ),
        lambda: (
            f"What does {rng.choice(DOMAINS)} smell like on a "
            f"{rng.choice(ADJECTIVES)} day? One sentence.",
            55, 1.2, "std"
        ),
        lambda: (
            f"If {rng.choice(NOUNS)} were a texture, what would it feel like? One sentence.",
            50, 1.2, "std"
        ),
        lambda: (
            f"Describe the weight of a {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}. "
            f"One sentence.",
            50, 1.2, "std"
        ),
        lambda: (
            f"What does {rng.choice(ADJECTIVES)} taste like on a "
            f"{rng.choice(ADJECTIVES)} Tuesday? One sentence.",
            60, 1.2, "std"
        ),
        lambda: (
            f"If {rng.choice(NOUNS)} were a piece of furniture, what would it be? "
            f"One sentence.",
            55, 1.2, "std"
        ),
        lambda: (
            f"Describe {rng.choice(ADJECTIVES)} ambition as if it were a "
            f"{rng.choice(DOMAINS)} phenomenon. One sentence.",
            65, 1.2, "std"
        ),
        lambda: (
            f"What is the opposite of a {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}? "
            f"One sentence.",
            45, 1.2, "std"
        ),
        lambda: (
            f"If {rng.choice(NOUNS)} had a postal address, what neighborhood would it "
            f"live in? One sentence.",
            55, 1.2, "std"
        ),
        lambda: (
            f"What sound does {rng.choice(ADJECTIVES)} silence make in "
            f"{rng.choice(DOMAINS)}? One sentence.",
            55, 1.2, "std"
        ),

        # ── Unexpected combinations ───────────────────────────────────────
        lambda: (
            f"Describe the {rng.choice(['Battle of Thermopylae','French Revolution','Apollo 11','Black Death'])} "
            f"as a Yelp review written by a {rng.choice(ADJECTIVES)} {rng.choice(ANIMALS)}. "
            f"Two sentences.",
            80, 1.0, "std"
        ),
        lambda: (
            f"Explain {rng.choice(DOMAINS)} to a {rng.choice(ADJECTIVES)} "
            f"{rng.choice(ANIMALS)} using only gestures. One sentence.",
            70, 1.0, "std"
        ),
        lambda: (
            f"What would a {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}'s "
            f"LinkedIn headline say? One sentence.",
            60, 1.0, "std"
        ),
        lambda: (
            f"Write a one-sentence Yelp review of the concept of "
            f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}.",
            65, 1.0, "std"
        ),
        lambda: (
            f"Write a performance review for {rng.choice(ADJECTIVES)} "
            f"{rng.choice(NOUNS)} as if it were an employee at {rng.choice(COMPANIES)}. "
            f"Two sentences.",
            80, 1.0, "std"
        ),
        lambda: (
            f"Describe {rng.choice(DOMAINS)} as a heated argument between a "
            f"{rng.choice(ADJECTIVES)} {rng.choice(ANIMALS)} and a {rng.choice(ADJECTIVES)} "
            f"{rng.choice(NOUNS)}. One sentence.",
            75, 1.0, "std"
        ),
        lambda: (
            f"What would {rng.choice(COMPANIES)}'s mission statement be if it "
            f"was founded by a {rng.choice(ADJECTIVES)} {rng.choice(ANIMALS)}? One sentence.",
            70, 1.0, "std"
        ),
        lambda: (
            f"Write a {rng.choice(['Shakespearean','haiku-style','legal','medical'])} "
            f"description of {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}. One sentence.",
            75, 1.0, "std"
        ),

        # ── Short factual QA (short output kills spec even at temp=0) ─────
        lambda: (
            f"What is the capital city of {rng.choice(['Iceland','Bhutan','Suriname','Djibouti','Vanuatu','Palau','Nauru','Tuvalu'])}?",
            15, 0.0, "std"
        ),
        lambda: (
            f"In one sentence, define {rng.choice(DOMAINS)}.",
            50, 0.0, "std"
        ),
        lambda: (
            f"What is {rng.randint(2,99)} times {rng.randint(2,99)}? "
            f"Give only the number.",
            10, 0.0, "std"
        ),
        lambda: (
            f"What does the acronym {rng.choice(['RAID','BIOS','ASCII','MIME','CRUD','REST','SOAP','AJAX','CORS','JWT'])} stand for?",
            25, 0.0, "std"
        ),
        lambda: (
            f"Name the {rng.choice(['three','four','five'])} most common elements "
            f"in {rng.choice(['the human body','Earth crust','the atmosphere','seawater'])}.",
            40, 0.0, "std"
        ),
        lambda: (
            f"What is the time complexity of {rng.choice(['merge sort','heap sort','bubble sort','binary search','DFS','BFS'])} "
            f"in the worst case?",
            25, 0.0, "std"
        ),
        lambda: (
            f"In what year was {rng.choice(['Python','Java','C++','Rust','Go','Kotlin','Swift','TypeScript'])} first released?",
            15, 0.0, "std"
        ),
        lambda: (
            f"What is the SI unit of {rng.choice(['pressure','luminosity','electric current','magnetic flux','frequency'])}?",
            20, 0.0, "std"
        ),
    ]

    results = []
    for _ in range(n):
        gen = rng.choice(generators)
        prompt, max_tokens, temp, req_type = gen()
        prompt = f"[{unique_tag(rng)}] {prompt}"
        results.append((prompt, max_tokens, temp, req_type))
    return results


# ── Traffic phases ────────────────────────────────────────────────────────────
# (start_frac, end_frac, proportion_of_total, label)
BASE_PHASES = [
    (0/60,  5/60,  0.05, "low_load"),
    (5/60,  10/60, 0.10, "ramp_up"),
    (10/60, 15/60, 0.27, "burst_1"),
    (15/60, 30/60, 0.10, "recovery_1"),
    (30/60, 35/60, 0.08, "steady"),
    (35/60, 40/60, 0.30, "burst_2"),
    (40/60, 60/60, 0.10, "cool_down"),
]


def make_phases(duration_s: int, total_requests: int) -> List[tuple]:
    phases    = []
    allocated = 0
    for i, (start_frac, end_frac, proportion, label) in enumerate(BASE_PHASES):
        start_s = int(start_frac * duration_s)
        end_s   = int(end_frac   * duration_s)
        if i == len(BASE_PHASES) - 1:
            n = total_requests - allocated
        else:
            n = round(proportion * total_requests)
            allocated += n
        phases.append((start_s, end_s, n, label))
    return phases


# ── Core dataclass ────────────────────────────────────────────────────────────

@dataclass
class TraceRequest:
    request_id:   int
    arrival_time: float
    prompt:       str
    max_tokens:   int
    temperature:  float
    request_type: str   # "spec" or "std"
    phase:        str


# ── Generation ────────────────────────────────────────────────────────────────

def generate_trace(duration_s: int = 3600,
                   total_requests: int = 300,
                   seed: int = 42) -> List[TraceRequest]:
    rng = random.Random(seed)
    np.random.seed(seed)
    phases = make_phases(duration_s, total_requests)

    # Pre-generate all unique prompts up front
    n_spec = total_requests // 2
    n_std  = total_requests - n_spec
    spec_pool = make_spec_prompts(rng, n_spec)
    std_pool  = make_std_prompts(rng, n_std)
    rng.shuffle(spec_pool)
    rng.shuffle(std_pool)

    spec_iter = iter(spec_pool)
    std_iter  = iter(std_pool)

    requests = []
    req_id   = 0

    for start, end, n_requests, phase_label in phases:
        if n_requests == 0:
            continue
        duration = end - start
        rate     = n_requests / duration

        inter_arrivals = np.random.exponential(1.0 / rate, size=n_requests * 4)
        timestamps     = np.cumsum(inter_arrivals) + start
        timestamps     = timestamps[timestamps < end][:n_requests]

        if len(timestamps) < n_requests:
            extra      = np.linspace(start, end, n_requests - len(timestamps) + 2)[1:-1]
            timestamps = np.sort(np.concatenate([timestamps, extra]))[:n_requests]

        for t in timestamps:
            # Alternate spec/std to keep 50/50 mix in every phase
            if req_id % 2 == 0:
                try:
                    prompt, max_tokens, temperature, req_type = next(spec_iter)
                except StopIteration:
                    prompt, max_tokens, temperature, req_type = next(std_iter)
            else:
                try:
                    prompt, max_tokens, temperature, req_type = next(std_iter)
                except StopIteration:
                    prompt, max_tokens, temperature, req_type = next(spec_iter)

            requests.append(TraceRequest(
                request_id   = req_id,
                arrival_time = round(float(t), 3),
                prompt       = prompt,
                max_tokens   = max_tokens,
                temperature  = temperature,
                request_type = req_type,
                phase        = phase_label,
            ))
            req_id += 1

    requests.sort(key=lambda r: r.arrival_time)
    return requests


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_trace(requests: List[TraceRequest], path: str,
               duration_s: int, total_requests: int):
    phases = make_phases(duration_s, total_requests)
    data = {
        "metadata": {
            "total_requests": len(requests),
            "spec_requests":  len([r for r in requests if r.request_type == "spec"]),
            "std_requests":   len([r for r in requests if r.request_type == "std"]),
            "duration_s":     duration_s,
            "duration_min":   duration_s / 60,
            "phases":         [[s, e, n, l] for s, e, n, l in phases],
        },
        "requests": [asdict(r) for r in requests],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(requests)} requests to {path}")


def load_trace(path: str) -> List[TraceRequest]:
    with open(path) as f:
        data = json.load(f)
    return [TraceRequest(**r) for r in data["requests"]]


# ── Summary ───────────────────────────────────────────────────────────────────

def print_trace_summary(requests: List[TraceRequest]):
    phase_order = ["low_load", "ramp_up", "burst_1", "recovery_1",
                   "steady", "burst_2", "cool_down"]
    dur = max(r.arrival_time for r in requests)
    print(f"\n  Total requests : {len(requests)}")
    print(f"  Spec-favored   : {len([r for r in requests if r.request_type == 'spec'])}")
    print(f"  Std-favored    : {len([r for r in requests if r.request_type == 'std'])}")
    print(f"  Duration       : {dur:.1f}s ({dur/60:.1f} min)")
    print(f"  Unique prompts : {len(set(r.prompt for r in requests))} / {len(requests)}")
    print(f"\n  {'Phase':<15} {'Reqs':>6} {'Start':>9} {'End':>9} {'Gap':>12}")
    print(f"  {'─'*15} {'─'*6} {'─'*9} {'─'*9} {'─'*12}")
    for phase in phase_order:
        group = [r for r in requests if r.phase == phase]
        if not group:
            continue
        t0  = min(r.arrival_time for r in group)
        t1  = max(r.arrival_time for r in group)
        gap = (t1 - t0) / len(group) if len(group) > 1 else 0
        print(f"  {phase:<15} {len(group):>6} {t0:>8.1f}s {t1:>8.1f}s {gap:>10.1f}s/req")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartSpec trace generator")
    parser.add_argument("--output",   type=str, default="trace.json",
                        help="Output JSON file (default: trace.json)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Total duration in MINUTES (default: 60)")
    parser.add_argument("--requests", type=int, default=300,
                        help="Total number of requests (default: 300)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    duration_s = args.duration * 60
    print(f"\n  Generating trace: {args.duration} min | "
          f"{args.requests} requests | seed={args.seed}")

    requests = generate_trace(duration_s=duration_s,
                              total_requests=args.requests,
                              seed=args.seed)
    print_trace_summary(requests)
    save_trace(requests, args.output, duration_s, args.requests)