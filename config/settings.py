ALLOWED_HOSTS = ["*"]
CORS_ALLOW_ALL_ORIGINS = True

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    # existing middleware...
]
INSTALLED_APPS = [
    # existing apps...
    "corsheaders",
    "api",
]

STATIC_ROOT = BASE_DIR / "staticfiles"