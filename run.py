import sys
import uvicorn
from app.core.config import settings

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if __name__ == "__main__":
    print("=" * 65)
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"📖 Interactive Swagger UI: http://127.0.0.1:{settings.port}/docs")
    print(f"📚 ReDoc Documentation:    http://127.0.0.1:{settings.port}/redoc")
    print(f"🔍 OpenAPI JSON Schema:    http://127.0.0.1:{settings.port}/openapi.json")
    print("=" * 65)
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
