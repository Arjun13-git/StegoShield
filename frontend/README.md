# StegoShield web (Next.js)

Frontend for the StegoShield FastAPI backend: **Analyze**, **Encode** and **Research** pages. It contains no ML logic; every
score, indicator and metric comes from the API or from `src/data/research-summary.json`.

## Run

```bash
# 1. backend (repository root, virtual environment active). The browser talks to the API directly,
#    so the backend must allow the frontend's origin:
cd backend && CORS_ORIGINS=http://localhost:3000 uvicorn app.main:app --port 8000

# 2. frontend
cd frontend
cp .env.example .env.local        # optional; NEXT_PUBLIC_API_BASE_URL defaults to http://127.0.0.1:8000
npm install
npm run dev                       # http://localhost:3000
```

`NEXT_PUBLIC_API_BASE_URL` is embedded in the browser bundle: never put secrets in it.

## Scripts

| command | purpose |
|---|---|
| `npm run lint` | ESLint (Next.js core-web-vitals + TypeScript rules) |
| `npm test` | Vitest + Testing Library (API client, error mapping, upload/encode validation, result rendering) |
| `npm run build` | production build |

## Research data

`src/data/research-summary.json` holds aggregate Phase 1A/1B results. It is generated, not hand-written:

```bash
python scripts/export_research_summary.py   # from the repository root; reads data/reports/ (git-ignored)
```

## Notes

- `stego_score` is an uncalibrated model score, never a probability; risk levels and verdicts are shown exactly as the API returns them.
- Client-side file checks are advisory; the backend validates every upload.
- Toolchain pins: TypeScript 6 and ESLint 9, because `typescript-eslint` does not yet support TypeScript 7 and `eslint-plugin-react` does not yet support ESLint 10.
