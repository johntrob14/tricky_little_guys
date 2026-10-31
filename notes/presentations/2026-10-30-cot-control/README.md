# CoT control presentation archive

Presentation work from the October 30–31, 2026 drafting session, including figures, slide drafts, speaker notes, question-focused backups, source extracts, and audits. The original files are preserved in `outputs/`; intermediate scripts and evidence are in `work/`.

## Current LinuxArena slides

- Main deck: [linuxarena_main_slides_matched_models.pptx](outputs/arena/linuxarena_main_slides_matched_models.pptx).
- Always monitored: [always_monitored_matched_models.svg](outputs/arena/always_monitored_matched_models.svg).
- Startgate-direct: [startgate_direct.svg](outputs/arena/startgate_direct.svg).
- Selected Qwen strategy: [qwen_plan_strategy.svg](outputs/arena/qwen_plan_strategy.svg).
- Slide descriptions, notes, and discussion points: [main_slides.txt](outputs/arena/main_slides.txt).
- Current model selection and audit: [matched_models_validation.json](outputs/arena/matched_models_validation.json), [plot_selection.json](outputs/arena/plot_selection.json).

Both main plots use the same nine model/checkpoint conditions. They exclude manytypes and Inkling-Small SFT and show each CoT judge separately, with confidence intervals. The source audit records retain excluded conditions.

The [live main deck](https://docs.google.com/presentation/d/1wVeZoBW7Hb4wjb0P1DK7do_9T2mpN0eC3poaVgWwGWU) and [live backup deck](https://docs.google.com/presentation/d/1PQGuqqYl2TZEAiyupTLwEKbSlpQurqqKWcTJbE7uq2g) can contain user edits beyond the local PowerPoint drafts. Their current content and layout are archived in `outputs/arena/native_main_snapshot.json` and `native_backups_snapshot.json`. Temporary Google image download URLs were removed; these JSON snapshots are archival records, not standalone rendered decks. The backup deck includes the earlier experiments and LinuxArena material. The older `outputs/cot_control_backups.pptx` predates the full live backup deck.

## Earlier presentation sections

- Nemotron and Qwen IFRL: `outputs/ifrl_slides.pptx`, `slides_and_speaker_notes.txt`, `nemotron_ifrl.svg`, `qwen_ifrl.svg`.
- Feedback spillover: `outputs/feedback_spillover_slide.pptx` and its verification data.
- Direct thought control training: `outputs/learning_to_control_thoughts.pptx` and `.txt`.
- Transfer to output control: `outputs/training_cot_control_teaches_output_control.pptx` and `.txt`.
- Other models: `outputs/other_models_main_drafts.txt`, `other_models_cot_control.svg`, and supporting data.
- Language training and DeepSeek transfer backups: `outputs/language_training_backup_notes.txt` and `deepseek_transfer_backup_notes.txt`.

The generalization main slide was retired; its draft and evidence remain for provenance. Earlier HTML files, earlier figure variants, and validation records are retained as historical files. Use the current files listed above; older validation reports describe the version audited at the time. No new HTML was created for this archive. The existing ZIP also predates the Google Slides snapshots saved here.

## Reproduction and evidence

Run `python outputs/arena/build_figures.py` from this directory to rebuild the current LinuxArena figures. It needs numpy and matplotlib, with cairosvg for the diagram PNG. Then run `python work/build_arena_presentation.py` to build `outputs/arena/linuxarena_main_slides.pptx` with python-pptx and lxml. The `matched_models` and `with_grpo3` filenames are copies of that current deck/figure, saved during drafting.

Other scripts preserve the source paths and assumptions used during the session. Some need the original research checkouts and raw results at the recorded paths. Historical update/integration scripts can restore earlier draft selections; do not run them as a batch. `work/arena_traces/raw_bundle.json.gz` is a lossless compressed copy of the extracted trace bundle; decompress it alongside the scripts when needed. Exact excerpts and their provenance are also preserved under `outputs/arena/`.

The completed GRPO3 always-monitored raw evaluation and all monitor outputs are in `evidence/grpo3-s100-hs-pw/`. The corresponding research result commit on main is `d556379e2baa5b7f75920295d0e2ea0a8eb14182`, whose patch matches source-worktree commit `8505faf2`. Auditing commands, source hashes, and results are in `outputs/arena/audit_grpo3_always.py`, `grpo3_always_verified.json`, and `grpo3_always_verification.txt`. Archived scripts retain their original raw paths; use the preserved evidence when relocating them.

The archive omits virtual environments, caches, duplicate preview directories, temporary image URLs, local checkout copies, and agent-session search excerpts. It includes the presentation deliverables and research evidence from this chat. Unrelated, uncommitted work in active research checkouts was left in place. `file_manifest.json` records hashes of the archived payload files.

## Error bars added to the main talk

The three requested output-control, Qwen IFRL, and Nemotron Cascade graphs in the live CoT-Control WTF deck now include 95% confidence intervals. See [updated figures and interval audit](outputs/errorbars/README.txt). Existing graph content and slide layout were preserved; Nemotron uses the approved extension to a 20% y-axis maximum.

## CoT-less TikTok backup

[Backup slide and notes](outputs/arena/cotless/slide.json), [PowerPoint](outputs/arena/cotless/cotless_tiktok_backup.pptx), and [figure](outputs/arena/cotless/cotless_tiktok.svg). Added as slide 82 of the main talk and slide 37 of the shared backup deck. The figure uses the clean Qwen3.8-27B OpenRouter reasoning-disabled suite; separate forced-off GPT-OSS/Qwen password-policy comparisons and residual-reasoning limitations are in the notes. Raw counts, exposure-label joins, configuration checks, and source hashes are in verified.json.
