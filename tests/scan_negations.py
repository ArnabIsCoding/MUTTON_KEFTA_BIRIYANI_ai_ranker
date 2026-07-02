
import json
import re
import sys
from collections import Counter, defaultdict

IMPORTANT_KEYWORDS = [
    "embeddings", "embedding", "vector database", "vector search",
    "retrieval", "ranking", "re-ranking", "recommendation",
    "search engine", "bm25", "faiss", "elasticsearch", "pinecone",
    "qdrant", "milvus", "weaviate", "hnsw", "semantic search",
    "ndcg", "mrr", "a/b test", "precision", "recall",
    "deployed", "deployment", "production", "scaled", "inference",
    "latency", "real-time", "microservice", "kubernetes",
    "llm", "rag", "fine-tuning", "lora", "qlora", "peft",
    "transformer", "python", "fastapi", "pytorch", "tensorflow",
    "ml", "machine learning", "deep learning", "nlp",
    "search", "recommendation system",
]

NEGATION_WORDS = ["not", "never", "didn't", "don't", "doesn't", "wasn't", 
                  "weren't", "aren't", "no", "without", "lack", "lacking",
                  "haven't", "hasn't", "hadn't", "isn't", "cannot", "can't"]

def find_negation_near_keywords(text, window=40):
    text_lower = text.lower()
    findings = []
    
    for neg_word in NEGATION_WORDS:
        neg_pattern = re.compile(r'\b' + re.escape(neg_word) + r'\b')
        for neg_match in neg_pattern.finditer(text_lower):
            neg_start = neg_match.start()
            neg_end = neg_match.end()
            
            search_region = text_lower[neg_end:neg_end + window]
            
            for kw in IMPORTANT_KEYWORDS:
                kw_pattern = re.compile(r'\b' + re.escape(kw) + r'\b')
                kw_match = kw_pattern.search(search_region)
                if kw_match:
                    ctx_start = max(0, neg_start - 20)
                    ctx_end = min(len(text), neg_end + kw_match.end() + 20)
                    context = text[ctx_start:ctx_end].replace('\n', ' ').strip()
                    
                    findings.append({
                        "negation": neg_word,
                        "keyword": kw,
                        "context": context,
                        "distance": kw_match.start(),  
                    })
    
    return findings


def classify_finding(finding):
    ctx = finding["context"].lower()
    neg = finding["negation"]
    kw = finding["keyword"]
    
    true_negation_patterns = [
        f"{neg} {kw}",                    
        f"{neg} any {kw}",                
        f"{neg} experience in {kw}",      
        f"{neg} experience with {kw}",    
        f"{neg} worked on {kw}",          
        f"{neg} built {kw}",              
        f"{neg} used {kw}",               
        f"{neg} involved in {kw}",        
        f"limited {kw}",                  
        f"{neg} hands-on {kw}",           
        f"{neg} direct {kw}",             
    ]
    
    for pattern in true_negation_patterns:
        if pattern in ctx:
            return "TRUE_NEGATION"
    
    false_negation_patterns = [
        f"no downtime",           
        f"no single point",       
        f"no issues",             
        f"not a poc",             
        f"not a proof",          
        f"not limited to",        
        f"no competition",        
        f"not just",              
        f"no data loss",
        f"not only",
        f"without downtime",
        f"without interruption",
        f"without any issues",
        f"without compromising",
    ]
    
    for pattern in false_negation_patterns:
        if pattern in ctx:
            return "FALSE_NEGATION"
    
    return "AMBIGUOUS"


def main():
    candidates_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\VS\India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    
    print(f"Loading candidates from {candidates_path}...")
    
    all_findings = []
    total_candidates = 0
    candidates_with_findings = 0
    total_descriptions = 0
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            cand = json.loads(line)
            total_candidates += 1
            cand_id = cand.get("candidate_id", "")
            
            texts = []
            for role in cand.get("career_history", []):
                desc = role.get("description", "")
                if desc:
                    texts.append(desc)
                    total_descriptions += 1
            
            profile = cand.get("profile", {})
            summary = profile.get("summary", "")
            if summary:
                texts.append(summary)
            
            cand_findings = []
            for text in texts:
                findings = find_negation_near_keywords(text)
                for f_item in findings:
                    f_item["candidate_id"] = cand_id
                    f_item["classification"] = classify_finding(f_item)
                    cand_findings.append(f_item)
            
            if cand_findings:
                candidates_with_findings += 1
                all_findings.extend(cand_findings)
            
            if line_num % 20000 == 0:
                print(f"  Processed {line_num} candidates... ({len(all_findings)} findings so far)")
    
    print(f"\n{'='*70}")
    print(f"NEGATION DETECTION ANALYSIS — {total_candidates} candidates, {total_descriptions} career descriptions")
    print(f"{'='*70}")
    
    classifications = Counter(f["classification"] for f in all_findings)
    
    print(f"\nTotal findings (negation word near a keyword): {len(all_findings)}")
    print(f"Candidates with at least one finding: {candidates_with_findings} ({candidates_with_findings/total_candidates*100:.2f}%)")
    print(f"\nClassification breakdown:")
    for cls, count in classifications.most_common():
        print(f"  {cls}: {count} ({count/len(all_findings)*100:.1f}%)")
    
    true_negs = [f for f in all_findings if f["classification"] == "TRUE_NEGATION"]
    print(f"\n{'='*70}")
    print(f"TRUE NEGATIONS (cases where removing negation detection would hurt):")
    print(f"{'='*70}")
    print(f"Count: {len(true_negs)}")
    if true_negs:
        seen_contexts = set()
        shown = 0
        for tn in true_negs:
            ctx_key = tn["context"][:60]
            if ctx_key not in seen_contexts:
                seen_contexts.add(ctx_key)
                print(f"\n  [{tn['candidate_id']}] neg='{tn['negation']}' kw='{tn['keyword']}'")
                print(f"    \"{tn['context']}\"")
                shown += 1
                if shown >= 30:
                    break
    
    false_negs = [f for f in all_findings if f["classification"] == "FALSE_NEGATION"]
    print(f"\n{'='*70}")
    print(f"FALSE NEGATIONS (cases where negation detection HURTS — suppresses true matches):")
    print(f"{'='*70}")
    print(f"Count: {len(false_negs)}")
    if false_negs:
        seen_contexts = set()
        shown = 0
        for fn in false_negs:
            ctx_key = fn["context"][:60]
            if ctx_key not in seen_contexts:
                seen_contexts.add(ctx_key)
                print(f"\n  [{fn['candidate_id']}] neg='{fn['negation']}' kw='{fn['keyword']}'")
                print(f"    \"{fn['context']}\"")
                shown += 1
                if shown >= 30:
                    break
    
    ambiguous = [f for f in all_findings if f["classification"] == "AMBIGUOUS"]
    print(f"\n{'='*70}")
    print(f"AMBIGUOUS (unclear if negation applies to keyword):")
    print(f"{'='*70}")
    print(f"Count: {len(ambiguous)}")
    if ambiguous:
        seen_contexts = set()
        shown = 0
        for amb in ambiguous:
            ctx_key = amb["context"][:60]
            if ctx_key not in seen_contexts:
                seen_contexts.add(ctx_key)
                print(f"\n  [{amb['candidate_id']}] neg='{amb['negation']}' kw='{amb['keyword']}'")
                print(f"    \"{amb['context']}\"")
                shown += 1
                if shown >= 50:
                    break
    
    print(f"\n{'='*70}")
    print(f"KEYWORDS MOST OFTEN NEAR NEGATION WORDS:")
    print(f"{'='*70}")
    kw_counts = Counter(f["keyword"] for f in all_findings)
    for kw, count in kw_counts.most_common(20):
        true_count = sum(1 for f in all_findings if f["keyword"] == kw and f["classification"] == "TRUE_NEGATION")
        false_count = sum(1 for f in all_findings if f["keyword"] == kw and f["classification"] == "FALSE_NEGATION")
        amb_count = sum(1 for f in all_findings if f["keyword"] == kw and f["classification"] == "AMBIGUOUS")
        print(f"  '{kw}': {count} total (TRUE_NEG={true_count}, FALSE_NEG={false_count}, AMBIGUOUS={amb_count})")
    
    print(f"\n{'='*70}")
    print(f"NEGATION WORDS MOST COMMON:")
    print(f"{'='*70}")
    neg_counts = Counter(f["negation"] for f in all_findings)
    for nw, count in neg_counts.most_common():
        print(f"  '{nw}': {count}")
    
    print(f"\n{'='*70}")
    print(f"VERDICT:")
    print(f"{'='*70}")
    total = len(all_findings)
    if total == 0:
        print("  No negation-near-keyword cases found. Negation detection is unnecessary.")
    else:
        true_pct = len(true_negs) / total * 100 if total else 0
        false_pct = len(false_negs) / total * 100 if total else 0
        print(f"  True negations (would lose by removing): {len(true_negs)} ({true_pct:.1f}%)")
        print(f"  False negations (currently hurting us):  {len(false_negs)} ({false_pct:.1f}%)")
        print(f"  Ambiguous:                              {len(ambiguous)}")
        
        if len(false_negs) >= len(true_negs):
            print(f"\n  >>> RECOMMENDATION: REMOVE negation detection.")
            print(f"      False negations ({len(false_negs)}) >= True negations ({len(true_negs)}).")
            print(f"      The feature hurts more than it helps.")
        else:
            print(f"\n  >>> RECOMMENDATION: KEEP negation detection but review false negation cases.")
            print(f"      True negations ({len(true_negs)}) > False negations ({len(false_negs)}).")
            print(f"      But verify the true negation examples are real.")


if __name__ == "__main__":
    main()