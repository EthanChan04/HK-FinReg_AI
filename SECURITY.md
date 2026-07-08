# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## ⚠️ Disclaimer

This project is an **academic proof-of-concept** (PoC) for demonstrating Multi-Agent Compliance AI powered by LangGraph. It is **NOT** intended for production use in regulated financial environments without significant additional security hardening.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub Issue
2. Email the maintainer directly at the address listed in the repository profile
3. Include a detailed description of the vulnerability and steps to reproduce

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Security Measures

### API Key Management
- All API keys are loaded from `.env` files which are **excluded from version control** via `.gitignore`
- `.env.example` files provide templates with placeholder values only
- The backend is configured to use `Authorization: Bearer <key>` protection by default
- The frontend must not expose backend tokens via `NEXT_PUBLIC_*`; use server-side proxy env vars like `BACKEND_API_KEY` only

### CORS Policy
- Cross-Origin Resource Sharing is restricted to explicitly allowed origins
- No wildcard (`*`) origins are permitted in the default configuration

### Data Privacy
- PII (Personally Identifiable Information) is scrubbed before being sent to LLM providers
- HKID numbers, phone numbers, and email addresses are automatically redacted via regex filters
- PII scrubbing is applied globally across all business endpoints (SVF, Bank Account, Cross-Border, SME, Copilot, Research, KAG)
- Review-queue runtime registry files must not persist raw user submissions or be committed to the repository

### Rate Limiting
- API rate limiting is enforced via `RateLimitMiddleware`
- Default limits: 60 requests/minute, 500 requests/hour per client IP
- Health check endpoint (`/api/v1/health`) is exempt from rate limiting
- Configure via `RATE_LIMIT_RPM` and `RATE_LIMIT_RPH` in `.env`

### Startup Security Checks
- The application performs security configuration checks on startup
- Warnings are printed for: disabled API key auth, empty API key, DEBUG mode, wildcard CORS, disabled rate limiting

### Production Deployment Recommendations
- Keep `API_KEY_ENABLED=True` and set a strong `API_KEY` in `.env`
- Set `DEBUG=False` to disable Swagger documentation endpoints
- Configure `CORS_ORIGINS` to only include your production frontend domain
- Use HTTPS for all API communications
- Configure rate limiting (`RATE_LIMIT_RPM`, `RATE_LIMIT_RPH`) appropriately
- Regularly rotate all API keys and tokens (see Key Rotation Policy below)

### Key Rotation Policy

| Key | Rotation Frequency | Where to Rotate |
| --- | --- | --- |
| `API_KEY` (backend bearer) | Every 90 days | Generate new: `python -c "import secrets; print(secrets.token_urlsafe(32))"`, update `backend/.env` and `frontend/.env.local` |
| `ZHIPU_API_KEY` / `LONGCAT_API_KEY` | Every 90 days | Provider dashboard |
| `COHERE_API_KEY` | Every 90 days | [Cohere dashboard](https://dashboard.cohere.com/api-keys) |
| `LANGCHAIN_API_KEY` | Every 90 days | [LangSmith settings](https://smith.langchain.com) |
| `EMBEDDING_API_KEY` | Every 90 days | Provider dashboard |

After rotation:
1. Update the `.env` file on the server
2. Restart the backend service
3. Verify with `GET /api/v1/health`
4. Revoke old keys at the provider dashboard immediately
