"""Authentication routes — login / register / logout."""

import re

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from web.core.security import hash_password, verify_password, create_access_token
from app.db.dal import query_one, execute

router = APIRouter(tags=["auth"])

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _templates(request: Request):
    return request.app.state.templates


def _user(request: Request):
    return getattr(request.state, "user", None)


# ── Login ────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _user(request):
        return RedirectResponse("/", status_code=303)
    return _templates(request).TemplateResponse(request, "pages/login.html", {
        "error": None,
    })


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    user = query_one("SELECT id, username, password_hash FROM users WHERE username = %s", (username,))
    if not user or not verify_password(password, user["password_hash"]):
        return _templates(request).TemplateResponse(request, "pages/login.html", {
            "error": "Sai tên đăng nhập hoặc mật khẩu.",
        })

    token = create_access_token({"sub": str(user["id"]), "username": user["username"]})
    # Redirect to ?next= param or home (validate same-origin to prevent open redirect)
    next_url = request.query_params.get("next", "/")
    if not next_url.startswith("/") or next_url.startswith("//"):
        next_url = "/"
    response = RedirectResponse(next_url, status_code=303)
    response.set_cookie(
        "access_token", token,
        httponly=True, max_age=7 * 24 * 3600, samesite="lax",
    )
    return response


# ── Register ─────────────────────────────────────────────

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if _user(request):
        return RedirectResponse("/", status_code=303)
    return _templates(request).TemplateResponse(request, "pages/register.html", {
        "error": None,
    })


@router.post("/register", response_class=HTMLResponse)
async def register_submit(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
):
    # Validate inputs
    if not _USERNAME_RE.match(username):
        return _templates(request).TemplateResponse(request, "pages/register.html", {
            "error": "Tên đăng nhập chỉ gồm chữ cái, số, dấu gạch dưới (3-30 ký tự).",
        })
    if not _EMAIL_RE.match(email):
        return _templates(request).TemplateResponse(request, "pages/register.html", {
            "error": "Email không hợp lệ.",
        })
    if len(password) < 6:
        return _templates(request).TemplateResponse(request, "pages/register.html", {
            "error": "Mật khẩu cần ít nhất 6 ký tự.",
        })

    # Check uniqueness
    existing = query_one(
        "SELECT id FROM users WHERE username = %s OR email = %s",
        (username, email),
    )
    if existing:
        return _templates(request).TemplateResponse(request, "pages/register.html", {
            "error": "Tên đăng nhập hoặc email đã được sử dụng.",
        })

    hashed = hash_password(password)
    execute(
        "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
        (username, email, hashed),
    )

    # Auto-login after registration
    user = query_one("SELECT id, username FROM users WHERE username = %s", (username,))
    token = create_access_token({"sub": str(user["id"]), "username": user["username"]})
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        "access_token", token,
        httponly=True, max_age=7 * 24 * 3600, samesite="lax",
    )
    return response


# ── Logout ───────────────────────────────────────────────

@router.get("/logout")
async def logout():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("access_token")
    return response
