"""Sanity test for the HRR core — runs with numpy only, no model downloads."""
from src.hrr import RoleFillerEncoder

enc = RoleFillerEncoder(dim=2048)

chunk_trace = enc.encode_structure([
    ("subject", "nolan"), ("action", "direct"), ("object", "inception"),
])
distractor_trace = enc.encode_structure([
    ("subject", "dream"), ("action", "fascinate"), ("object", "humanity"),
])

print("Probe chunk for 'subject':    ", enc.probe(chunk_trace, "subject", top_k=2))
print("Probe distractor for 'subject':", enc.probe(distractor_trace, "subject", top_k=2))

top_name, top_sim = enc.probe(chunk_trace, "subject", top_k=1)[0]
assert top_name == "filler::nolan", f"expected nolan, got {top_name}"
assert top_sim > 0.15, f"cleanup too weak: {top_sim}"
print("\nHRR core: PASS")
