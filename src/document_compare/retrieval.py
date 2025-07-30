import difflib

def compare_documents(text1: str, text2: str) -> dict:
    """
    Compare two documents and return similarity ratio and differences.
    """
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    diff = list(difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm=''))
    added = [line[1:] for line in diff if line.startswith('+') and not line.startswith('+++')]
    removed = [line[1:] for line in diff if line.startswith('-') and not line.startswith('---')]
    changed = [line for line in diff if line.startswith('?')]
    summary = []
    if added:
        summary.append(f"Added lines: {len(added)}")
        summary.extend(added[:10])
    if removed:
        summary.append(f"Removed lines: {len(removed)}")
        summary.extend(removed[:10])
    if changed:
        summary.append(f"Changed lines: {len(changed)}")
        summary.extend(changed[:10])
    if not summary:
        summary.append("No significant differences found.")
    return {
        "similarity": similarity,
        "differences": summary
    }
