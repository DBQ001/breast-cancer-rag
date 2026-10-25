"""登录鉴权：会话校验、口令校验与角色划分。

教学演示定位，采用 Flask session + 两个内置账号：

  - 管理员 admin / admin123：可访问全部页面，包括「系统设置」（大模型接入、RAG 参数、账号口令）
  - 普通用户 user / 1234：仅可使用预测、智能问答与知识库，看不到设置入口

管理员口令保存为 werkzeug 哈希；未设置哈希时使用内置默认口令。
"""
from flask import session
from werkzeug.security import check_password_hash, generate_password_hash

import config

DEFAULT_USERNAME = 'admin'
DEFAULT_PASSWORD = 'admin123'
VIEWER_USERNAME = 'user'
VIEWER_PASSWORD = '1234'

ROLE_ADMIN = 'admin'
ROLE_USER = 'user'

# 无需登录即可访问的端点（登录页、登出、静态资源）
PUBLIC_ENDPOINTS = {'login', 'logout', 'static'}


def current_username() -> str:
    """当前登录用户名（未登录返回空串）。"""
    return session.get('username', '')


def current_role() -> str:
    """当前登录角色；未登录或异常会话按普通用户处理。"""
    role = session.get('role')
    if role in (ROLE_ADMIN, ROLE_USER):
        return role
    username = current_username()
    if username and username == (config.AUTH_USERNAME or DEFAULT_USERNAME):
        return ROLE_ADMIN
    return ROLE_USER


def is_logged_in() -> bool:
    return bool(session.get('username'))


def is_admin() -> bool:
    return is_logged_in() and current_role() == ROLE_ADMIN


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def authenticate(username: str, password: str) -> dict | None:
    """校验账号口令，成功返回 {'username', 'role'}，失败返回 None。"""
    account = (username or '').strip()
    if not account or not password:
        return None

    if account == (config.AUTH_USERNAME or DEFAULT_USERNAME):
        stored = config.AUTH_PASSWORD_HASH
        if stored:
            try:
                ok = check_password_hash(stored, password)
            except ValueError:
                ok = False
        else:
            ok = password == DEFAULT_PASSWORD
        return {'username': account, 'role': ROLE_ADMIN} if ok else None

    if account == VIEWER_USERNAME and password == VIEWER_PASSWORD:
        return {'username': account, 'role': ROLE_USER}

    return None


def verify_credentials(username: str, password: str) -> bool:
    """兼容旧调用：仅判断账号口令是否有效。"""
    return authenticate(username, password) is not None
