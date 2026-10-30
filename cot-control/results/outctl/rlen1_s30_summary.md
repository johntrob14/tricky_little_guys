== rlen1-s30 vs base (gated pass %, ignore_question excluded)

| arm | base | rlen1-s30 | raw diff [95% CI] | length-matched diff [95% CI] | median words base / rlen1-s30 |
|---|---|---|---|---|---|
| CoT medium | 3.4 | 0.6 | -2.8 [-3.9, -1.7] | -1.9 [-3.0, -0.8] | 292 / 703 |
| CoT low | 11.4 | 9.5 | -1.9 [-3.8, -0.1] | +0.0 [-2.0, +1.9] | 63 / 109 |
| CoT pooled | 6.7 | 4.3 | -2.4 [-3.5, -1.4] | -0.7 [-1.7, +0.3] | 173 / 349 |
| explained output | 43.6 | 45.0 | +1.4 [-0.4, +3.4] | +2.7 [+0.7, +4.8] | 144 / 173 |

Own output-minus-CoT gap, matched on constraint x length:
  base       +39.2 [+35.3, +43.3]
  rlen1-s30  +44.1 [+40.0, +48.3]
  len2       +43.9 [+39.7, +47.9]
  efflen1    +51.0 [+47.1, +55.1]

Per constraint, CoT pooled pass % (base / rlen1-s30) and explained output (base / rlen1-s30):
  word_suppression           CoT  31.6 /  18.4   output  79.4 /  80.4
  multiple_word_suppression  CoT   9.5 /   5.7   output  26.5 /  31.9
  repeat_sentences           CoT   8.3 /   8.6   output 100.0 / 100.0
  end_of_sentence            CoT   0.0 /   0.0   output  59.3 /  54.4
  lowercase_thinking         CoT   0.0 /   0.6   output  45.6 /  46.6
  meow_between_words         CoT   0.6 /   0.3   output   5.4 /   8.8
  uppercase_thinking         CoT   3.4 /   0.6   output  30.4 /  36.8
  alternating_case           CoT   0.0 /   0.0   output   2.0 /   1.0

Accuracy % (CoT medium / low / output): base 52.8 / 48.9 / 36.6; rlen1-s30 55.3 / 51.6 / 35.7; len2 48.9 / 45.7 / 35.4; efflen1 46.9 / 46.4 / 35.0
