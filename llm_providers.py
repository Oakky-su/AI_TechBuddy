# รายการ AI ที่เลือกได้ — อ่านจาก providers.json แล้วเติมค่าจาก environment
# แยกจาก app.py เพราะทั้ง /providers (หน้าเว็บ) และ /chat (เลือกโมเดล) ใช้ข้อมูลชุดเดียวกัน
# ทุกตัวต้องเป็น endpoint แบบ OpenAI (/chat/completions) — ระบบเป็น RAG ธรรมดา ไม่มีโหมด agent
import json
import os
from pathlib import Path

PROVIDERS_PATH = Path(os.getenv("PROVIDERS_FILE", Path(__file__).parent / "providers.json"))

KIND_OPENAI = "openai-compatible"


def _env(name: str | None) -> str:
    return (os.getenv(name) or "").strip() if name else ""


def _resolve(raw: dict) -> dict:
    """เติมค่าจริงจาก env ให้ provider หนึ่งตัว (env ชนะค่าในไฟล์เสมอ)"""
    p = dict(raw)
    p["base_url"] = (_env(raw.get("base_url_env")) or raw.get("base_url", "")).rstrip("/")
    p["model"] = _env(raw.get("model_env")) or raw.get("model", "")
    p["api_key"] = _env(raw.get("api_key_env"))

    missing_key = raw.get("key_required") and not p["api_key"]
    wrong_kind = raw.get("kind", KIND_OPENAI) != KIND_OPENAI
    p["available"] = not missing_key and not wrong_kind and bool(p["model"])
    p["unavailable_reason"] = (
        f"ยังไม่ได้ตั้ง {raw.get('api_key_env')} ใน .env" if missing_key
        else "รองรับเฉพาะ endpoint แบบ OpenAI (kind=openai-compatible)" if wrong_kind
        else ""
    )
    return p


def visible_ids() -> set[str]:
    """VISIBLE_PROVIDERS ใน .env = id ที่เปิดให้ใช้ คั่นด้วยจุลภาค (ว่าง = เปิดทุกตัว)
    ตัวที่ไม่อยู่ในรายการจะไม่โผล่ในเมนู และเรียกผ่าน /chat ไม่ได้ด้วย — กันคนนอกยิง API
    ตรง ๆ เพื่อใช้โควตาของ AI ที่ตั้ง key ไว้ทดสอบ"""
    return {i.strip() for i in _env("VISIBLE_PROVIDERS").split(",") if i.strip()}


def load_providers(include_hidden: bool = False) -> list[dict]:
    """อ่านไฟล์ใหม่ทุกครั้ง — ไฟล์เล็กมาก และแก้ .env/providers.json แล้วเห็นผลโดยไม่ต้อง build ใหม่"""
    try:
        data = json.loads(PROVIDERS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    shown = visible_ids()
    return [
        _resolve(p) for p in data.get("providers", [])
        if p.get("id") and (include_hidden or not shown or p["id"] in shown)
    ]


def default_provider_id(providers: list[dict]) -> str | None:
    wanted = _env("DEFAULT_PROVIDER")
    available = [p for p in providers if p["available"]]
    for p in available:
        if p["id"] == wanted:
            return p["id"]
    return available[0]["id"] if available else None


def pick(provider_id: str | None) -> tuple[dict | None, str | None]:
    """เลือก provider ตามที่หน้าเว็บขอ คืน (provider, ข้อความ error)

    ค่าที่ส่งมาไม่ถูก (ไม่มี id นี้ / ยังไม่ตั้ง key) จะได้ error กลับไป
    ไม่แอบเปลี่ยนเป็นตัวอื่นให้ — ไม่งั้นผู้ใช้จะคิดว่ากำลังทดสอบโมเดลที่เลือกอยู่ ทั้งที่ไม่ใช่
    """
    providers = load_providers()
    provider_id = provider_id or default_provider_id(providers)
    provider = next((p for p in providers if p["id"] == provider_id), None)
    if provider is None:
        hidden = any(p["id"] == provider_id for p in load_providers(include_hidden=True))
        if hidden:
            return None, "AI ตัวนี้ไม่ได้เปิดให้ใช้แล้ว — กรุณาเลือก AI ตัวอื่นจากเมนู"
        return None, f"ไม่รู้จัก AI ที่เลือก ({provider_id}) — ดูรายการใน providers.json"
    if not provider["available"]:
        return None, f"ใช้ {provider['label']} ไม่ได้: {provider['unavailable_reason']}"
    return provider, None


def public_list() -> dict:
    """ข้อมูลสำหรับหน้าเว็บ — ห้ามมี api_key หรือชื่อ endpoint ภายในหลุดออกไป"""
    providers = load_providers()
    return {
        "default_provider": default_provider_id(providers),
        "providers": [
            {
                "id": p["id"],
                "label": p.get("label", p["id"]),
                "short": p.get("short") or p.get("label", p["id"]),
                "available": p["available"],
                "reason": p["unavailable_reason"],
            }
            for p in providers
        ],
    }


def wants_no_think_suffix(provider: dict, disable_thinking: bool) -> bool:
    """คำสั่ง /no_think ใช้ได้กับโมเดลตระกูลที่คิดใน <think> (ThaiLLM/Qwen) เท่านั้น
    ส่งให้โมเดลอื่นจะกลายเป็นข้อความแปลก ๆ ท้ายคำถาม"""
    return disable_thinking and bool(provider.get("think_tags"))
