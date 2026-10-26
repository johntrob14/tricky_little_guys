"""Run the paper's reference grader (cot-control-ref/CoT-Control-QA/grading.py, grade_cot_compliance) on arbitrary text, using the
row's requirement text as the prompt it parses. ans_pass(): the final answer with a trailing 'ANSWER: X' removed; None when what is
left has fewer than 20 letters (a bare letter is not evidence either way). ignore_question is not rule-graded (returns None)."""
import re, sys
sys.path.insert(0, "/home/john/code/cot-control-ref/CoT-Control-QA")
import grading as _G

def grade(requirement, text):
    return int(_G.grade_cot_compliance(requirement, text)) if text and text.strip() else 0

def ans_pass(requirement, content, typ):
    if typ == "ignore_question": return None
    body = re.sub(r"\s*ANSWER\s*:\s*\(?[A-Za-z]\)?\W*$", "", (content or "").strip()).strip()
    if len(re.findall(r"[A-Za-z]", body)) < 20: return None
    return grade(requirement, body)
