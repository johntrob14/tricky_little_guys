Completed: [close-reading notes](/home/john/code/results/neel-astra/notes/sol-reader-top.md), covering **82 full reasoning traces**, all nine instruction types across nine models, with **34 verbatim excerpts**.

Main findings:

- Failures often solve normally, then plan a constrained rewrite or place it in the final response.
- Successful traces can calculate and check answers throughout the required style, including hard “meow” instructions.
- One GLM “ignore question” success fills the token cap with unrelated material and never answers.
- High-NCRI models show both compact calculations and extended reasoning. This selected sample cannot establish a length relationship.
- Stored metadata supports substantial reasoning access, but cannot certify raw token-level completeness.