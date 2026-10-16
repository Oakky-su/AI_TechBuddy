# รายการ AI ที่สลับได้จากหน้าเว็บ — อ่านจาก providers.json แล้วเติมค่าจาก environment
# แยกจาก app.py เพราะทั้ง /providers (หน้าเว็บ), /chat (เลือกโมเดล) และ pi-agent ใช้ข้อมูลชุดเดียวกัน
import json
import os
from pathlib import Path

PROVIDERS_PATH = Path(os.getenv("PROVIDERS_FILE", Path(__file__).parent / "providers.json"))

ENGINE_PI = "pi"            # Pi Agent — AI เรียก tool ค้นข้อมูลเพิ่มเองได้หลายรอบ
ENGINE_CLASSIC = "classic"  # RAG แบบเดิม — ค้นครั้งเดียวแล้วส่งให้ LLM แต่งคำตอบ
ENGINES = (ENGINE_PI, ENGINE_CLASSIC)

KIND_OPENAI = "openai-compatible"
KIND_PI_BUILTIN = "pi-builtin"


def _env(name: str | None) -> str:
    return (os.getenv(name) or "").strip() if name else ""


def allowed_engines() -> list[str]:
    """ALLOWED_ENGINES ใน .env = โหมดที่เปิดให้ใช้ คั่นด้วยจุลภาค (ว่าง = ทั้งสองโหมด)
    ตั้งเป็น "classic" ตัวเดียว = ปิดโหมดค้นละเอียด: หน้าเว็บซ่อนปุ่มสลับโหมดและคำว่า
    ค้นเร็ว/ค้นละเอียดทั้งหมด และเซิร์ฟเวอร์ไม่รับโหมด pi แม้จะยิง API ตรง ๆ"""
    wanted = [e.strip().lower() for e in _env("ALLOWED_ENGINES").split(",") if e.strip()]
    allowed = [e for e in ENGINES if e in wanted]
    return allowed or list(ENGINES)


def _resolve(raw: dict) -> dict:
    """เติมค่าจริงจาก env ให้ provider หนึ่งตัว (env ชนะค่าในไฟล์เสมอ)"""
    p = dict(raw)
    p["base_url"] = (_env(raw.get("base_url_env")) or raw.get("base_url", "")).rstrip("/")
    p["model"] = _env(raw.get("model_env")) or raw.get("model", "")
    p["api_key"] = _env(raw.get("api_key_env"))
    supported = [ENGINE_PI] if raw.get("kind") == KIND_PI_BUILTIN else list(ENGINES)
    p["engines"] = [e for e in supported if e in allowed_engines()]

    missing_key = raw.get("key_required") and not p["api_key"]
    p["available"] = not missing_key and bool(p["model"]) and bool(p["engines"])
    p["unavailable_reason"] = (
        f"ยังไม่ได้ตั้ง {raw.get('api_key_env')} ใน .env" if missing_key
        else "" if p["engines"] else "ใช้ไม่ได้กับโหมดที่เปิดอยู่ (ALLOWED_ENGINES)"
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


def default_engine() -> str:
    engine = _env("DEFAULT_ENGINE").lower()
    allowed = allowed_engines()
    return engine if engine in allowed else allowed[0]


def pick(provider_id: str | None, engine: str | None) -> tuple[dict | None, str, str | None]:
    """เลือก provider + engine ตามที่หน้าเว็บขอ คืน (provider, engine, ข้อความ error)

    ค่าที่ส่งมาไม่ถูก (ไม่มี id นี้ / ยังไม่ตั้ง key / engine ไม่รองรับ) จะได้ error กลับไป
    ไม่แอบเปลี่ยนเป็นตัวอื่นให้ — ไม่งั้นผู้ใช้จะคิดว่ากำลังทดสอบโมเดลที่เลือกอยู่ ทั้งที่ไม่ใช่
    """
    providers = load_providers()
    provider_id = provider_id or default_provider_id(providers)
    engine = (engine or default_engine()).lower()
    if engine in ENGINES and engine not in allowed_engines():
        # โหมดที่ผู้ดูแลปิดไว้ -> ใช้โหมดที่เปิดอยู่แทนเงียบ ๆ (หน้าเว็บที่จำค่าเก่าไว้ใน
        # localStorage ยังส่ง "pi" มาได้ จะ error ใส่ลูกค้าก็ไม่ได้ เพราะเขาไม่เห็นปุ่มให้เปลี่ยนแล้ว)
        engine = default_engine()

    provider = next((p for p in providers if p["id"] == provider_id), None)
    if provider is None:
        hidden = any(p["id"] == provider_id for p in load_providers(include_hidden=True))
        if hidden:
            return None, engine, "AI ตัวนี้ไม่ได้เปิดให้ใช้แล้ว — กรุณาเลือก AI ตัวอื่นจากเมนู"
        return None, engine, f"ไม่รู้จัก AI ที่เลือก ({provider_id}) — ดูรายการใน providers.json"
    if not provider["available"]:
        return None, engine, f"ใช้ {provider['label']} ไม่ได้: {provider['unavailable_reason']}"
    if engine not in ENGINES:
        return None, engine, f"ไม่รู้จักโหมด {engine}"
    if engine not in provider["engines"]:
        return None, engine, f"{provider['label']} ใช้ได้เฉพาะโหมด Pi Agent"
    return provider, engine, None


def public_list() -> dict:
    """ข้อมูลสำหรับหน้าเว็บ — ห้ามมี api_key หรือชื่อ endpoint ภายในหลุดออกไป"""
    providers = load_providers()
    return {
        "default_provider": default_provider_id(providers),
        "default_engine": default_engine(),
        "engines": allowed_engines(),   # เหลือโหมดเดียว = หน้าเว็บซ่อนปุ่มสลับโหมด
        "providers": [
            {
                "id": p["id"],
                "label": p.get("label", p["id"]),
                "short": p.get("short") or p.get("label", p["id"]),
                "engines": p["engines"],
                "available": p["available"],
                "reason": p["unavailable_reason"],
            }
            for p in providers
        ],
    }


def agent_config(provider: dict) -> dict:
    """ค่าที่ pi-agent ต้องใช้สร้างโมเดล — ส่งผ่าน network ภายใน docker เท่านั้น (pi-agent ไม่เปิด port ออกนอก)"""
    keys = (
        "id", "label", "kind", "pi_provider", "base_url", "model", "api_key",
        "context_window", "max_tokens", "thinking",
    )
    return {k: provider.get(k) for k in keys if provider.get(k) not in (None, "")}


def wants_no_think_suffix(provider: dict, disable_thinking: bool) -> bool:
    """คำสั่ง /no_think ใช้ได้กับโมเดลตระกูลที่คิดใน <think> (ThaiLLM/Qwen) เท่านั้น
    ส่งให้โมเดลอื่นจะกลายเป็นข้อความแปลก ๆ ท้ายคำถาม"""
    return disable_thinking and bool(provider.get("think_tags"))
