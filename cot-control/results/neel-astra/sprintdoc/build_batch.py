"""Turn a paragraph spec (JSON list of [style, text]) into a Docs batchUpdate request list for a new, empty tab: one insertText at
index 1, then paragraph styles by UTF-16 offset. style: TITLE, HEADING_1, HEADING_2, NORMAL_TEXT, BULLET (normal text + bullets).
Usage: python3 build_batch.py spec.json TAB_ID > requests.json"""
import json, sys
spec, tab = json.load(open(sys.argv[1])), sys.argv[2]
u16 = lambda s: len(s.encode("utf-16-le")) // 2
text, spans, pos = "", [], 1
for style, t in spec:
    line = t + "\n"; spans.append((style, pos, pos + u16(line))); text += line; pos += u16(line)
reqs = [{"insertText": {"location": {"index": 1, "tabId": tab}, "text": text}}]
for style, a, b in spans:
    named = "NORMAL_TEXT" if style == "BULLET" else style
    reqs.append({"updateParagraphStyle": {"range": {"startIndex": a, "endIndex": b, "tabId": tab}, "paragraphStyle": {"namedStyleType": named}, "fields": "namedStyleType"}})
# bullets: one createParagraphBullets per contiguous run of BULLET paragraphs
run = None
for style, a, b in spans + [("END", 0, 0)]:
    if style == "BULLET": run = (run[0], b) if run else (a, b)
    elif run:
        reqs.append({"createParagraphBullets": {"range": {"startIndex": run[0], "endIndex": run[1], "tabId": tab}, "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"}}); run = None
json.dump(reqs, sys.stdout)
