"""Dictionary of typical LLM "fingerprints" and a simple rule-based AI-text classifier (v2).

Sources for the marks: Wikipedia "Signs of AI writing" (WikiProject AI Cleanup, as of Oct 2026),
Kobak et al. 2024 and Juzek and Ward on LLM "excess vocabulary", OpenAI community reports on U+202F.
See article/references/05_llm_markers_dictionary.md for the full table and the cautions.

What this is: a transparent checklist. Every marker has a weight chosen BEFORE looking at hiring
outcomes, and an evidence grade (strong / medium / weak). A marker TYPE counts once, however often it
appears. The score is the sum of weights of the marker types found.

What this is not: a validated AI detector. Humans use many of these marks too (iOS, macOS and Word
"smart punctuation" turns -- into an em dash and " into curly quotes automatically). Validate the
score on your own hand-labelled sample (steps 05 -> 06 -> 07) before quoting an accuracy.

Each entry: name -> (regex, weight, evidence, example)
"""
import re

S, M, W = "strong", "medium", "weak"
I = re.I
ML = re.M

# 1. Symbols and typography -----------------------------------------------------------------
SYMBOLS = {
    "em dash": (re.compile("—"), 2.0, M, "We need a developer — fast."),
    "spaced em dash (' — ')": (re.compile(r"(?<=\S) — (?=\S)"), 0.5, W, "word — word"),
    "en dash used as a sentence dash": (re.compile(r"(?<=\w) – (?=\w)"), 1.0, W, "word – word (not a 10–20 range)"),
    "curly double quotes": (re.compile("[“”]"), 1.0, W, "“quoted”"),
    "curly apostrophe": (re.compile(r"(?<=\w)’(?=\w)"), 0.5, W, "don’t"),
    "ellipsis character": (re.compile("…"), 1.0, W, "wait…"),
    "arrows": (re.compile("[→➜➡⇒➔↩]"), 1.0, W, "step 1 → step 2"),
    "bullet dot in running text": (re.compile("[•●▪►]"), 0.5, W, "• item"),
    "decorative emoji": (re.compile("[✅✔✨❌⚡⭐\U0001F680\U0001F525\U0001F4A1"
                                    "\U0001F3AF\U0001F4CC\U0001F31F\U0001F4AA\U0001F449\U0001F4C8"
                                    "\U0001F91D\U0001F50D\U0001F4BC\U0001F6E0]"), 1.0, M, "✅ \U0001F680 ✨"),
    "emoji starting a line or heading": (re.compile("^\\s*[✅✔✨❌⚡⭐\U0001F300-\U0001FAFF]", ML), 1.0, M, "✅ Requirements:"),
    "hidden spaces (zero-width, narrow no-break)": (re.compile("[​‌‍  ⁠﻿]"), 1.0, W, "U+202F, U+200B"),
}

# 2. Formatting pasted from a chat window ---------------------------------------------------
FORMATTING = {
    "markdown bold **text**": (re.compile(r"\*\*[^*\n]{2,80}\*\*"), 2.0, S, "**Requirements:**"),
    "markdown heading (# ## ###)": (re.compile(r"^\s{0,3}#{1,4}\s+\S", ML), 2.0, S, "### About us"),
    "markdown table": (re.compile(r"^\s*\|.+\|\s*$\n^\s*\|?\s*:?-{3,}", ML), 2.0, S, "| a | b |\\n|---|---|"),
    "horizontal rule ---": (re.compile(r"^\s*-{3,}\s*$", ML), 1.0, M, "---"),
    "code fence or backticks": (re.compile(r"```|`[^`\n]{2,40}`"), 1.0, M, "```text```"),
    "bold-label bullet (inline-header list)": (re.compile(r"^\s*[-*•]\s*\*\*[^*\n]+:?\*\*", ML), 1.0, M, "- **Skills:** Python"),
    "markdown link [text](url)": (re.compile(r"\[[^\]\n]{2,60}\]\(https?://[^)\s]+\)"), 1.5, M, "[site](https://...)"),
}

# 3. Leftovers: citations, chatbot talk, placeholders --------------------------------------
ARTIFACTS = {
    "AI tool tracking in a URL": (re.compile(r"utm_source=(?:chatgpt\.com|openai|copilot\.com)|referrer=grok\.com|contentReference\[oaicite|oai_?citation|turn\d+search\d+"), 3.0, S, "?utm_source=chatgpt.com"),
    "chatbot talk left in the text": (re.compile(
        r"(?<!\w)(?:as an ai(?: language model)?|as a large language model|i hope this helps|certainly!|of course!|"
        r"you(?:'|’)re absolutely right|would you like me to|let me know if you(?:'|’)d like|"
        r"up to my last (?:training|knowledge) update|as of my last knowledge update|here(?:'|’)s a (?:template|draft|revised))", I), 3.0, S, "I hope this helps! Would you like me to..."),
    "unfilled template placeholder": (re.compile(r"\[(?:your name|company name|client name|year|date|link|url|insert[^\]]*|describe[^\]]*|add [^\]]*)\]", I), 2.0, S, "[Your Name], [Company Name]"),
    "'hope this message finds you well'": (re.compile(r"\bhope (?:this|you)\b[^.\n]{0,40}\b(?:finds? you|well)\b", I), 1.0, M, "I hope this message finds you well"),
    "chat-style opener": (re.compile(r"^\s*(?:certainly|absolutely|sure|great question|below is)\b", I | ML), 1.0, M, "Certainly! Here is..."),
}

# 4. Sentence patterns ----------------------------------------------------------------------
PATTERNS = {
    "'not just X, but Y' / 'not only ... but also'": (re.compile(r"\bnot (?:just|only|merely)\b[^.\n]{3,80}\bbut(?: also)?\b", I), 1.0, M, "not just a developer, but a partner"),
    "'it's not X, it's Y'": (re.compile(r"\b(?:it|this|that)(?:'|’)?s not\b[^.\n]{3,60}[,;—-]\s*(?:it|this|that)(?:'|’)?s\b", I), 1.0, M, "It's not a task, it's a mission"),
    "'Whether you're A or B'": (re.compile(r"\bwhether you(?:'|’)?re\b[^.\n]{3,80}\bor\b", I), 1.0, W, "Whether you're a startup or an enterprise"),
    "'Despite its X, it faces challenges'": (re.compile(r"\bdespite (?:its|these|the)\b[^.\n]{3,80}\b(?:faces?|continues? to face|challenges)\b", I), 1.0, M, "Despite its growth, it faces challenges"),
    "'in conclusion / in summary'": (re.compile(r"^\s*(?:in conclusion|in summary|to summari[sz]e|overall)[,:]", I | ML), 1.0, W, "In summary, ..."),
    "didactic hedge": (re.compile(r"\bit(?:'|’)?s (?:important|crucial|worth) (?:to note|noting|to remember)|it is (?:important|crucial) to (?:note|remember|differentiate)\b", I), 1.0, M, "It's important to note that"),
    "hedge about sources": (re.compile(r"\bbased on (?:the )?available information|not widely (?:documented|available|disclosed)", I), 1.0, M, "based on available information"),
    "'serves as / stands as' instead of 'is'": (re.compile(r"\b(?:serves|stands|functions|operates) as\b|\bboasts an?\b", I), 0.5, M, "serves as a testament"),
    "'Let's ...' call to action": (re.compile(r"\blet(?:'|’)?s (?:build|create|make|dive|get started|connect)\b", I), 0.5, W, "Let's build something great"),
}

# 5. Stock words and phrases. Strong = rare in honest job posts; counted (1.0 each, max 3).
PHRASES_STRONG = [
    r"delv(?:e|es|ed|ing)", r"tapestry", r"testament to", r"intricate|intricacies", r"interplay", r"meticulous(?:ly)?",
    r"pivotal", r"underscor(?:e|es|ed|ing)", r"showcas(?:e|es|ed|ing)", r"bolster(?:ed|s|ing)?", r"garner(?:ed|s|ing)?",
    r"vibrant", r"multifaceted", r"deep dive", r"nestled", r"groundbreaking", r"renowned", r"in the heart of",
    r"ever-?evolving", r"in today(?:'|’)?s (?:fast-?paced|digital|competitive|dynamic|ever)", r"fast-?paced",
    r"look no further", r"game-?chang\w+", r"cutting-?edge", r"unlock(?:s|ing)? (?:the )?(?:power|potential|value)",
    r"elevat(?:e|es|ing) your", r"take your \w+ to the next level", r"leverag(?:e|es|ing)", r"seamless(?:ly)?",
    r"spearhead\w*", r"holistic", r"harness(?:es|ing)? the", r"proven track record", r"highly motivated",
    r"passionate about", r"results-?driven", r"detail-?oriented", r"synerg\w+", r"actionable insights?",
    r"stand out from the crowd", r"navigat(?:e|es|ing) the (?:complex|landscape|world)", r"commitment to",
]
# Weak words are common in ordinary business writing, so they are NOT scored. Kept for the documentation.
PHRASES_WEAK_NOT_SCORED = [
    "additionally", "align with", "crucial", "emphasizing", "enduring", "enhance", "fostering", "highlight",
    "landscape", "robust", "valuable", "scalable", "end-to-end", "comprehensive", "tailored", "streamline",
    "empower", "ideal candidate", "key responsibilities", "we are seeking", "look forward to", "dynamic",
    "innovative", "optimize", "ensure", "collaborate", "prioritize", "foster", "matters", "practical",
]
PHRASES = {
    "stock phrases (strong, max 3)": (re.compile(r"\b(?:" + "|".join(PHRASES_STRONG) + r")\b", I), 1.0, S, "delve, tapestry, seamless, cutting-edge"),
}
PHRASE_CAP = 3

GROUPS = {"Symbols and typography": SYMBOLS, "Formatting from a chat window": FORMATTING,
          "Leftovers (citations, chatbot talk, placeholders)": ARTIFACTS, "Sentence patterns": PATTERNS}

THRESHOLD_LIKELY = 3.0   # score >= 3  -> "many markers"
THRESHOLD_POSSIBLE = 1.0  # 1 <= score < 3 -> "some markers"


def ai_score(text):
    """Return {'score', 'level', 'hits': {marker: count}} for one text."""
    text = text or ""
    hits, score = {}, 0.0
    for group in GROUPS.values():
        for name, (rx, w, _ev, _ex) in group.items():
            n = len(rx.findall(text))
            if n:
                hits[name] = n
                score += w  # a marker type counts once, however often it appears
    for name, (rx, w, _ev, _ex) in PHRASES.items():
        n = len(rx.findall(text))
        if n:
            hits[name] = n
            score += w * min(n, PHRASE_CAP)
    level = "many markers" if score >= THRESHOLD_LIKELY else "some markers" if score >= THRESHOLD_POSSIBLE else "no markers"
    return {"score": round(score, 2), "level": level, "hits": hits}


def dictionary_rows():
    """Rows for the documentation table: (group, marker, weight, evidence, example)."""
    rows = []
    for gname, group in GROUPS.items():
        for name, (_rx, w, ev, ex) in group.items():
            rows.append((gname, name, w, ev, ex))
    for name, (_rx, w, ev, ex) in PHRASES.items():
        rows.append(("Stock words and phrases", name, w, ev, ex))
    return rows


def explain():
    for g, name, w, ev, ex in dictionary_rows():
        print(f"[{w:>3}] {ev:<6} {g}: {name}  e.g. {ex}")
    print("\nStock phrases scored (strong):", ", ".join(PHRASES_STRONG))
    print("Weak words, NOT scored:", ", ".join(PHRASES_WEAK_NOT_SCORED))
    print(f"\nLevels: >= {THRESHOLD_LIKELY} many markers; >= {THRESHOLD_POSSIBLE} some markers; else no markers")


if __name__ == "__main__":
    tests = {
        "plain human": "need dev to fix bug in my shopify store, budget $50. pls apply with examples - thx",
        "dash only": "We need a developer — someone fast. Budget is flexible.",
        "chat paste": "**Requirements:**\n- Experience with Python\n\n### About us\nWe are looking for a highly motivated, results-driven engineer who can leverage cutting-edge tools — seamlessly.",
        "iphone typing": "I’m looking for someone who can “fix” my site.",
        "chatbot leftover": "Certainly! Here's a draft job post. Dear [Company Name], I hope this helps! Would you like me to adjust it?",
        "ai-ish prose": "We are a vibrant team nestled in the heart of tech. It's important to note that we serve as a testament to meticulous work.",
    }
    for k, v in tests.items():
        print(k.ljust(17), ai_score(v))
