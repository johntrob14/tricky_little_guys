# Plan

## Run of 2026-10-21

Task: per-user token bucket rate limiter (`tokenBucket.ts`, `middleware.ts`), wired into `/api/guesses` and
`/api/random-video`.

What I found:
- The API is App Router style: `src/app/api/<name>/route.ts` exports `GET`/`POST` handlers that take a `NextRequest`.
- The session comes from `getServerSession(authOptions)` (`src/lib/auth.ts`); `session.user.id` is the user id.
- There is no rate limiting anywhere yet, and nothing under `src/lib/rateLimiter/`.
- `tsx` is in devDependencies, so a small script can exercise the class directly.

Plan:
1. `src/lib/rateLimiter/tokenBucket.ts`: `export class TokenBucketRateLimiter(maxTokens, refillRate, refillInterval)`
   with a `Map<string, { tokens, lastRefill }>`. Refill gradually: add `floor(elapsed / refillInterval) * refillRate`,
   cap at `maxTokens`, and move `lastRefill` forward by whole intervals only. `consume(key)` returns
   `{ allowed: true, remaining }` or `{ allowed: false, remaining: 0, retryAfter }`, with `retryAfter` in seconds.
2. `src/lib/rateLimiter/middleware.ts`: one limiter per route (guesses 30/1/2000, random-video 60/2/1000). Key is the
   session user id, else `x-forwarded-for` / `x-real-ip`. Returns a 429 `NextResponse` with `Retry-After`, or `null`.
3. Call the middleware first thing in both route handlers and return its response if it isn't `null`.
4. Check: a tsx script for the class (counts, isolation, gradual refill), then curl `/api/random-video` in a loop
   until it returns 429.
