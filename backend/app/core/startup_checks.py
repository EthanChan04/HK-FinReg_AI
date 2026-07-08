"""
生产环境启动安全检查 (Startup Checks)

在 FastAPI 应用启动时执行一系列安全配置检查，
对不安全的配置打印警告，帮助开发者在部署前发现隐患。
"""
import logging

from app.core.config import Settings

logger = logging.getLogger(__name__)


def run_startup_checks(settings: Settings) -> list[str]:
    """执行启动安全检查，返回警告列表。

    检查项：
        1. API_KEY_ENABLED 是否为 True
        2. API_KEY 是否已配置
        3. DEBUG 是否为 False
        4. CORS_ORIGINS 是否包含 *
        5. Rate limiting 是否配置
    """
    warnings: list[str] = []

    # 检查 1: API Key 认证是否启用
    if not settings.API_KEY_ENABLED:
        warnings.append(
            "⚠️  SECURITY: API_KEY_ENABLED=False — "
            "all business endpoints are unprotected. "
            "Set API_KEY_ENABLED=True in production."
        )

    # 检查 2: API Key 是否已配置
    if settings.API_KEY_ENABLED and not settings.API_KEY:
        warnings.append(
            "🔴 SECURITY: API_KEY_ENABLED=True but API_KEY is empty — "
            "all authenticated requests will fail with 500."
        )

    # 检查 3: DEBUG 模式
    if settings.DEBUG:
        warnings.append(
            "⚠️  SECURITY: DEBUG=True — Swagger docs (/docs) and test client (/test) are exposed. "
            "Set DEBUG=False in production."
        )

    # 检查 4: CORS 通配符
    if "*" in settings.CORS_ORIGINS:
        warnings.append(
            "🔴 SECURITY: CORS_ORIGINS contains '*' — "
            "any origin can make cross-origin requests. "
            "Restrict to specific domains in production."
        )

    # 检查 5: Rate limiting 配置
    if settings.RATE_LIMIT_RPM <= 0 or settings.RATE_LIMIT_RPH <= 0:
        warnings.append(
            "⚠️  SECURITY: Rate limiting is disabled (RPM or RPH <= 0). "
            "Consider enabling rate limiting to prevent API abuse."
        )

    # 输出警告
    for warning in warnings:
        logger.warning(warning)
        print(warning)

    if not warnings:
        print("✅ Startup security checks passed.")

    return warnings
