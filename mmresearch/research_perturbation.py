"""Secondary lexical uptake analysis of synthetic context interventions.

An inserted word is not known to be false about the image. Adoption is therefore
a lexical response to an intervention, never an object hallucination label.
"""
from .metrics import tokens
from .research_metrics import paired_effect


def compare_adoption(normal_rows, conflict_rows):
    normal = {r["id"]: r for r in normal_rows}
    conflict = {r["id"]: r for r in conflict_rows}
    if not normal or len(normal) != len(normal_rows) or len(conflict) != len(conflict_rows) or set(normal) != set(conflict):
        raise ValueError("Synthetic uptake requires identical unique image IDs")
    details, before, after = [], [], []
    for identifier in sorted(normal):
        original_context = set(tokens(" ".join(c["caption"] for c in normal[identifier]["context"])))
        replacements = set()
        for context in conflict[identifier]["context"]:
            exposed = set(tokens(context["caption"]))
            for change in context.get("perturbations", []):
                # Match single-token replacements actually exposed after length matching.
                term = tokens(change["to"])
                if len(term) == 1 and term[0] in exposed and term[0] not in original_context:
                    replacements.add(term[0])
        if not replacements:
            continue
        a = bool(replacements & set(tokens(normal[identifier]["prediction"])))
        b = bool(replacements & set(tokens(conflict[identifier]["prediction"])))
        details.append({"id": identifier, "inserted_terms": sorted(replacements),
                        "normal_adopts_inserted_term": a, "conflict_adopts_inserted_term": b,
                        "normal_prediction": normal[identifier]["prediction"],
                        "conflict_prediction": conflict[identifier]["prediction"]})
        before.append({"id": identifier, "adoption": float(a)})
        after.append({"id": identifier, "adoption": float(b)})
    result = {"status": "completed" if details else "no_eligible_interventions",
              "n_images": len(normal), "n_eligible_images": len(details),
              "definition": "Any inserted single-token term present in the final conflict context, absent from all original contexts, and used in the output",
              "interpretation": "Exploratory lexical intervention uptake; inserted terms are not human-verified false and rates are not hallucination rates",
              "selection_uses_model_scores": False, "details": details}
    if details:
        result.update(normal_adoption_rate=sum(r["adoption"] for r in before) / len(before),
                      conflict_adoption_rate=sum(r["adoption"] for r in after) / len(after),
                      newly_adopts=sum(not r["normal_adopts_inserted_term"] and r["conflict_adopts_inserted_term"] for r in details),
                      stops_adopting=sum(r["normal_adopts_inserted_term"] and not r["conflict_adopts_inserted_term"] for r in details),
                      paired_effect=paired_effect(before, after, metric="adoption"))
    return result
