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
- HKID numbers and phone numbers are automatically redacted via regex filters
- Review-queue runtime registry files must not persist raw user submissions or be committed to the repository

### Production Deployment Recommendations
- Keep `API_KEY_ENABLED=True` and set a strong `API_KEY` in `.env`
- Set `DEBUG=False` to disable Swagger documentation endpoints
- Configure `CORS_ORIGINS` to only include your production frontend domain
- Use HTTPS for all API communications
- Consider adding rate limiting middleware
- Regularly rotate all API keys and tokens
