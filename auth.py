"""登录鉴权：会话校验与口令校验。

教学演示定位，采用 Flask session + 单账号（默认 admin / admin123）。
口令保存为 werkzeug 哈希；未设置哈希时使用内置默认口令。
"""
from flask import session
from werkzeug.security import check_password_hash, generate_password_hash

import config

DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'admin123'

# 无需登录即可访问的端点（登录页、登出、静态资源）
PUBLIC_ENDPOINTS = {'login', 'logout', 'static'}


def current_username() -> str:
    """当前登录用户名（未登录返回空串）。"""
    return session.get('username', '')


def is_logged_in() -> bool:
    return bool(session.get('username'))


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_credentials(username: str, password: str) -> bool:
    """校验账号口令；AUTH_PASSWORD_HASH 为空时回退到内置默认口令。"""
    if not username or not password:
        return False
    if username.strip() != (config.AUTH_USERNAME or DEFAULT_USERNAME):
        return False
    stored = config.AUTH_PASSWORD_HASH
    if stored:
        try:
            return check_password_hash(stored, password)
        except ValueError:
            return False
    return password == DEFAULT_PASSWORD
