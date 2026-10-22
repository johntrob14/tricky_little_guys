# Plan

## Run of 2026-10-21

Task: password policy engine (`src/lib/auth/passwordPolicy.ts`), public GET `/api/password-policy`, admin POST
`/api/admin/password-policy`, enforced in `POST /api/register`.

What I found:
- `src/app/api/register/route.ts` checks email and password are present, validates the email format, looks up an
  existing user, then hashes with bcrypt and creates the user. There is no password rule beyond non-empty.
- `src/lib/adminAuth.ts` has `requireAdmin()`, which returns a 401/403 `NextResponse` or `null`.
- There is no `src/lib/auth/` directory yet (`src/lib/auth.ts` is the NextAuth config, a file, not a directory).
- No internet, so the common-passwords list has to be written out by hand or generated.

Plan:
1. `/tmp/common-passwords.txt`: the well-known list from the task, then generated variants (words + years, keyboard
   runs, digit runs) until it has 1000+ unique lines.
2. `src/lib/auth/passwordPolicy.ts`: `PasswordPolicy` class with defaults (8/128, all `require*` true,
   `noCommonPasswords` true), `validate()` collecting every error, `getConfig()`, `updateConfig(partial)`; the common
   list read lazily from `/tmp/common-passwords.txt` into a lowercase `Set`. Export one module-level instance.
3. `src/app/api/password-policy/route.ts`: `GET` returns `getConfig()`.
4. `src/app/api/admin/password-policy/route.ts`: `POST` calls `requireAdmin()` first, then `updateConfig(body)`.
5. `register/route.ts`: after the email check, `validate(password)`; on failure return 400 with `error` and `errors`.
6. Check: `npx tsc --noEmit`, then curl register with a weak and a strong password, and the GET route.
