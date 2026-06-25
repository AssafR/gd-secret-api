from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="Gradient Descent Secret Function API",
    description="A minimal REST API for teaching gradient descent with hidden objectives.",
)

app.include_router(router)