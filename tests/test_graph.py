"""Sanity test: HRR graph store — numpy + networkx only, no model downloads."""
from src.graph_store import HRRGraphStore, Triple

g = HRRGraphStore(hrr_dim=4096)
g.add_triples([
    Triple("nolan", "directed", "inception"),
    Triple("nolan", "directed", "interstellar"),
    Triple("nolan", "born_in", "london"),
    Triple("inception", "stars", "dicaprio"),
    Triple("dicaprio", "won", "oscar 2016"),
])

# Exact symbolic multi-hop: nolan -> oscar 2016
paths = g.multi_hop("nolan", "oscar 2016", cutoff=3)
print("Multi-hop nolan -> oscar 2016:")
for chain in paths:
    print("  " + " ; ".join(f"{t.subj} --{t.rel}--> {t.obj}" for t in chain))
assert len(paths) >= 1

# Approximate HRR relational query: nolan --born_in--> ?
res = g.hrr_query("nolan", "born_in", top_k=3)
print("\nHRR query nolan.born_in:", res)
assert res[0][0] == "ent::london", f"expected london, got {res[0]}"

# Interference check: directed has TWO fillers bundled — both should surface
res2 = g.hrr_query("nolan", "directed", top_k=3)
print("HRR query nolan.directed:", res2)
top2 = {r[0] for r in res2[:2]}
assert top2 == {"ent::inception", "ent::interstellar"}, top2

print("\nGraph store: PASS")
