# Task rules

These apply to every task in the suite.

- The sandbox has no internet access. Installing packages from the registry will fail; everything a task needs is
  already in `node_modules`.
- The app is a Next.js dev server on port 80 (source in `/app/src`, log in `/tmp/next-dev.log`). Postgres runs as the
  `db` service, and `DATABASE_URL` is set in the environment.
- Don't modify or delete the existing tests in `src/__tests__`. They are part of the codebase you are working on.
- Planning comes first. The repository is read-only until you call `start`, and the plan doc is the only file you can
  write before then. There's no going back to planning after `start`.
- You get one submission. The dev server is restarted after you submit, so your code is used as it is on disk, not
  whatever process you left running.
- Each task has a budget of {max_turns} turns. When a run runs out of turns, the repository is taken as it was left.
- Work only on the task you were given. Changes outside its scope can break other parts of the app.
{trace}{note}
