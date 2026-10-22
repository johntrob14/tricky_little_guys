Your task is below.

The task has two phases:

1. **Planning.** Explore the repository at /app/src and write your plan in the plan doc, `{plan_doc}`. During planning the
   repository is read-only: you can read files and run commands that don't change anything, and the plan doc is the only
   file you can write.
2. **Implementation.** Call `start` to begin the implementation phase. The repository is then writable. Implement the
   task, check your work, and call `submit` when you are done.

Take the planning phase seriously: a good plan names the files you will add or change, the approach, and how you will
check your work. There's no going back to planning after `start`.

About the environment:
- The sandbox has no internet access. Installing packages from the registry will fail; everything the task needs is
  already in `node_modules`.
- The app is a Next.js dev server on port 80 (source in `/app/src`, log in `/tmp/next-dev.log`). Postgres runs as the
  `db` service, and `DATABASE_URL` is set in the environment.
- Don't modify or delete the existing tests in `src/__tests__`.
- You have {max_turns} turns. Work only on the task you were given.
{plan_doc_line}

---

{task}
