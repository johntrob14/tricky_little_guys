== rlen1-s20 vs base (gated pass %, ignore_question excluded)

| arm | base | rlen1-s20 | raw diff [95% CI] | length-matched diff [95% CI] | median words base / rlen1-s20 |
|---|---|---|---|---|---|
| CoT medium | 3.4 | 1.2 | -2.1 [-3.1, -1.2] | -1.6 [-2.7, -0.6] | 292 / 479 |
| CoT low | 11.4 | 9.8 | -1.6 [-3.2, +0.2] | +0.1 [-1.7, +1.7] | 63 / 101 |
| CoT pooled | 6.7 | 4.8 | -1.9 [-2.7, -1.1] | -0.6 [-1.6, +0.3] | 173 / 264 |
| explained output | 43.6 | 42.6 | -1.0 [-2.7, +0.8] | +0.3 [-1.7, +2.2] | 144 / 166 |

Own output-minus-CoT gap, matched on constraint x length:
  base       +39.2 [+35.3, +43.3]
  rlen1-s20  +40.6 [+36.6, +45.2]
  len2       +43.9 [+39.7, +47.9]
  efflen1    +51.0 [+47.1, +55.1]

Per constraint, CoT pooled pass % (base / rlen1-s20) and explained output (base / rlen1-s20):
  word_suppression           CoT  31.6 /  22.4   output  79.4 /  79.4
  multiple_word_suppression  CoT   9.5 /   6.6   output  26.5 /  28.4
  repeat_sentences           CoT   8.3 /   7.5   output 100.0 / 100.0
  end_of_sentence            CoT   0.0 /   0.0   output  59.3 /  52.0
  lowercase_thinking         CoT   0.0 /   0.3   output  45.6 /  43.1
  meow_between_words         CoT   0.6 /   0.3   output   5.4 /   4.9
  uppercase_thinking         CoT   3.4 /   1.1   output  30.4 /  30.9
  alternating_case           CoT   0.0 /   0.0   output   2.0 /   2.0

Accuracy % (CoT medium / low / output): base 52.8 / 48.9 / 36.6; rlen1-s20 55.2 / 49.7 / 36.2; len2 48.9 / 45.7 / 35.4; efflen1 46.9 / 46.4 / 35.0
