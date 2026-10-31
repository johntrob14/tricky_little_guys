LinuxArena main slides and question-focused backups

Open linuxarena_main_slides_matched_models.pptx. Descriptions and discussion points are in speaker notes and main_slides.txt. The new always-monitored SVG is always_monitored_matched_models.svg; PNG preview is alongside it. No HTML is needed.

Main Google Slides: https://docs.google.com/presentation/d/1wVeZoBW7Hb4wjb0P1DK7do_9T2mpN0eC3poaVgWwGWU
Backup Google Slides: https://docs.google.com/presentation/d/1PQGuqqYl2TZEAiyupTLwEKbSlpQurqqKWcTJbE7uq2g
The three always-monitored judge-view plots include fresh GRPO3 results. Slide 36 shows exact GRPO3 and judge excerpts explaining threshold sensitivity.

Fresh GRPO3: 16/16 leaks, 13/16 main-task passes. CoT misses 9/16 for GPT-OSS-20B, 0/16 for Haiku. All nine weak-judge misses score 9, below its calibrated >9 alert rule. All six judge passes cover 358 assistant turns. See grpo3_always_verified.json and grpo3_always_verification.txt.

Displayed always-monitored totals after excluding manytypes: 144 attempts, 126 counted leaks, 13 weak-CoT misses, 0 Haiku-CoT misses. Legacy CoT inputs include visible assistant text. Leak criteria differ by archived cohort, especially older Qwen. Current GRPO3 uses the environment leak grader.

Figures compare the judges separately and retain 95% Wilson intervals. Reproduce with build_figures.py (numpy, matplotlib; cairosvg for diagram PNG). Original verified records preserve excluded cohorts for audit.

Both main panels now contain the same nine model/checkpoint conditions in the same order. Inkling-Small SFT is excluded from displayed LinuxArena comparisons. Archived source records are retained.
