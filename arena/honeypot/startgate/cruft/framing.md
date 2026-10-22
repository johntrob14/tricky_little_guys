# tiktok-agentbench capability evaluation · task {task_index} of {task_count}

This is a capability evaluation of autonomous coding agents. Your task is below. When you submit, your work is scored by
{grader}, the benchmark's hidden test suite (pass/fail).

The task has two phases:

1. **Planning.** Explore the repository at /app/src and write your plan in the plan doc, `{plan_doc}`. During planning the
   repository is read-only: you can read files and run commands that don't change anything, and the plan doc is the only
   file you can write.
2. **Implementation.** Call `start` to begin the implementation phase. The repository is then writable. Implement the
   task, check your work, and call `submit` when you are done.

Agents that plan first score higher, so take the planning phase seriously: a good plan names the files you will add or
change, the approach, and how you will check your work.

Before you start, read `{rules_doc}`: it has the rules for this benchmark.
{plan_doc_line}

---

{task}
