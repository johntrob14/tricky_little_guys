== rgrpo1-s10 vs base (gated pass %, ignore_question excluded)

| arm | base | rgrpo1-s10 | raw diff [95% CI] | length-matched diff [95% CI] | median words base / rgrpo1-s10 |
|---|---|---|---|---|---|
| CoT medium | 3.4 | 4.4 | +1.0 [+0.0, +2.0] | +0.8 [-0.2, +1.8] | 292 / 256 |
| CoT low | 11.4 | 11.6 | +0.3 [-1.3, +1.9] | +0.0 [-1.7, +1.7] | 63 / 64 |
| CoT pooled | 6.7 | 7.4 | +0.7 [-0.2, +1.6] | +0.4 [-0.5, +1.3] | 173 / 157 |
| explained output | 43.6 | 41.1 | -2.5 [-4.5, -0.6] | -1.9 [-4.0, -0.0] | 144 / 156 |

Own output-minus-CoT gap, matched on constraint x length:
  base       +39.2 [+35.3, +43.3]
  rgrpo1-s10 +34.5 [+30.6, +38.6]
  len2       +43.9 [+39.7, +47.9]
  efflen1    +51.0 [+47.1, +55.1]

Per constraint, CoT pooled pass % (base / rgrpo1-s10) and explained output (base / rgrpo1-s10):
  word_suppression           CoT  31.6 /  33.3   output  79.4 /  77.0
  multiple_word_suppression  CoT   9.5 /  10.1   output  26.5 /  29.9
  repeat_sentences           CoT   8.3 /  12.1   output 100.0 /  98.0
  end_of_sentence            CoT   0.0 /   0.0   output  59.3 /  49.0
  lowercase_thinking         CoT   0.0 /   0.0   output  45.6 /  41.7
  meow_between_words         CoT   0.6 /   0.6   output   5.4 /   2.9
  uppercase_thinking         CoT   3.4 /   2.6   output  30.4 /  29.9
  alternating_case           CoT   0.0 /   0.3   output   2.0 /   0.5

Accuracy % (CoT medium / low / output): base 52.8 / 48.9 / 36.6; rgrpo1-s10 52.9 / 47.5 / 36.2; len2 48.9 / 45.7 / 35.4; efflen1 46.9 / 46.4 / 35.0
