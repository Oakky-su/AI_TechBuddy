// Pi Agent sidecar ของ Tech Buddy — ใช้ agent loop ของ pi (pi.dev, @earendil-works/pi-agent-core)
//
// หน้าที่: รับ prompt ที่ teddy (Flask) ค้นข้อมูลมาให้แล้วหนึ่งรอบ จากนั้นให้ LLM ตัดสินใจเองว่า
// ต้องค้นเพิ่มไหม ถ้าต้องก็เรียก tool ซึ่งยิงกลับไปที่ teddy /internal/tools/<ชื่อ> (ค้นจาก Chroma
// และ laptops.json ชุดเดียวกับโหมด classic) วนได้หลายรอบจนได้คำตอบ แล้วส่งคำตอบสุดท้ายกลับ
// POST /chat ตอบเป็น JSON ก้อนเดียว ส่วน POST /chat/stream ทยอยส่ง event (เรียก tool / ข้อความที่
// กำลังพิมพ์) เป็น NDJSON ให้หน้าเว็บโชว์ความคืบหน้า แทนที่ผู้ใช้จะจ้องจุดสามจุดนาน 10-20 วินาที
//
// LLM เลือกได้ต่อ request: endpoint แบบ OpenAI (vLLM / ThaiLLM / Ollama) หรือ provider ที่ pi รู้จัก
// (anthropic / openai / google ...) — config ทั้งหมดมาจาก providers.json ฝั่ง teddy
//
// service นี้ไม่เปิด port ออกนอก docker network เพราะ request มี API key ของ provider ติดมาด้วย

import http from "node:http";
import { Agent } from "@earendil-works/pi-agent-core";
import { Type, createModels, createProvider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { builtinProviders } from "@earendil-works/pi-ai/providers/all";

const PORT = Number(process.env.PORT || 7070);
const TOOLS_URL = (process.env.TOOLS_URL || "http://teddy:5000").replace(/\/$/, "");
const INTERNAL_TOKEN = process.env.INTERNAL_TOKEN || "";
// กัน agent วนเรียก tool ไม่จบ: ครบโควตาแล้วถอด tool ออก บังคับให้ตอบจากข้อมูลที่มี
const MAX_TOOL_CALLS = Number(process.env.PI_MAX_TOOL_CALLS || 6);
const MAX_TURNS = Number(process.env.PI_MAX_TURNS || 8);
// ต้องน้อยกว่า PI_AGENT_TIMEOUT ฝั่ง teddy (150 วิ) จะได้ตอบ error ที่อ่านรู้เรื่องก่อนฝั่งนั้นตัดสาย
const RUN_TIMEOUT_MS = Number(process.env.PI_RUN_TIMEOUT_MS || 120_000);
// tool แต่ละครั้ง (ยิงกลับไป teddy) — ปกติไม่ถึงวินาที ถ้า teddy ค้าง (เช่น worker กำลังรีโหลด)
// ให้ LLM เห็นเป็น error แล้วตอบจากข้อมูลที่มี ไม่ใช่รอจนหมดเวลาทั้งรอบ
const TOOL_TIMEOUT_MS = Number(process.env.PI_TOOL_TIMEOUT_MS || 20_000);
const TEMPERATURE = 0.1;  // เท่ากับโหมด classic — ตอบนิ่ง รูปแบบคงเส้นคงวา
const MAX_BODY_BYTES = 1024 * 1024;
// PI_DEBUG=1 -> log สรุป payload ที่ส่งให้ LLM ทุกครั้ง (จำนวนข้อความ / tool ที่ประกาศ) ไว้ไล่ปัญหา
const DEBUG = process.env.PI_DEBUG === "1";

const models = createModels();
for (const provider of builtinProviders()) models.setProvider(provider);

// ---------------------------------------------------------------- models

const customModels = new Map();  // signature ของ config -> Model (สร้าง provider ครั้งเดียวต่อ config)

function openAICompatibleModel(cfg) {
  const providerId = `techbuddy-${cfg.id}`;
  const signature = JSON.stringify([cfg.base_url, cfg.model, cfg.context_window, cfg.max_tokens]);
  const cached = customModels.get(providerId);
  if (cached?.signature === signature) return cached.model;

  const model = {
    id: cfg.model,
    name: cfg.label || cfg.model,
    api: "openai-completions",
    provider: providerId,
    baseUrl: cfg.base_url,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: cfg.context_window || 32768,
    maxTokens: cfg.max_tokens || 2048,
    // vLLM / ThaiLLM ไม่รู้จัก role "developer", reasoning_effort, store และ max_completion_tokens
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStore: false,
      maxTokensField: "max_tokens",
    },
  };
  models.setProvider(createProvider({
    id: providerId,
    name: cfg.label || providerId,
    baseUrl: cfg.base_url,
    // endpoint ที่ไม่ใช้ key (เช่น vLLM ภายใน) ยังต้องมีค่าอะไรสักอย่าง เพราะ openai-completions
    // ของ pi ไม่ยอมยิงถ้า key ว่าง — vLLM ที่ไม่ได้ตั้ง --api-key จะไม่สนค่านี้
    // ถ้ามี key จริง getApiKey ของ Agent จะส่งค่านั้นมาแทน (ค่าที่ส่งมาตอนเรียกชนะค่านี้เสมอ)
    auth: { apiKey: { name: cfg.label || providerId, resolve: async () => ({ auth: { apiKey: "no-key" } }) } },
    models: [model],
    api: openAICompletionsApi(),
  }));
  customModels.set(providerId, { signature, model });
  return model;
}

function resolveModel(cfg) {
  if (cfg.kind === "pi-builtin") {
    const model = models.getModel(cfg.pi_provider, cfg.model);
    if (!model) throw new HttpError(400, `pi ไม่รู้จักโมเดล ${cfg.pi_provider}/${cfg.model}`);
    return model;
  }
  if (!cfg.base_url || !cfg.model) throw new HttpError(400, "provider ต้องมี base_url และ model");
  return openAICompatibleModel(cfg);
}

// ---------------------------------------------------------------- tools
// ชื่อ/ความหมายต้องตรงกับ TOOLS และ PI_TOOL_GUIDE ใน app.py
// ใช้ Type.Number แทน Integer — โมเดลบางตัวส่ง 25000.0 มา แล้วจะโดน validation ตีกลับเปล่า ๆ

const price = {
  max_price: Type.Optional(Type.Number({ description: "ราคาสูงสุด (บาท) — ใส่ทุกครั้งที่ลูกค้าบอกงบ" })),
  min_price: Type.Optional(Type.Number({ description: "ราคาต่ำสุด (บาท) — ใช้เมื่อลูกค้าถามเป็นช่วงราคา" })),
};
const limit = Type.Optional(Type.Number({ description: "จำนวนรุ่นที่ต้องการ (สูงสุด 15)" }));

async function callTeddyTool(name, params, signal) {
  const timeout = AbortSignal.timeout(TOOL_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(`${TOOLS_URL}/internal/tools/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Token": INTERNAL_TOKEN },
      body: JSON.stringify(params ?? {}),
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
  } catch (error) {
    if (timeout.aborted && !signal?.aborted) {
      throw new Error(`tool ${name} ไม่ตอบภายใน ${TOOL_TIMEOUT_MS / 1000} วินาที`);
    }
    throw error;
  }
  const data = await res.json().catch(() => ({}));
  // throw = pi ส่งผลลัพธ์แบบ isError ให้ LLM เห็น แล้วมันจะแก้ argument เองในรอบถัดไป
  if (!res.ok) throw new Error(data.error || `tool ${name} ตอบ HTTP ${res.status}`);
  return { content: [{ type: "text", text: String(data.text ?? "") }], details: { chars: String(data.text ?? "").length } };
}

function teddyTool(name, label, description, parameters) {
  return {
    name,
    label,
    description,
    parameters,
    execute: (_toolCallId, params, signal) => callTeddyTool(name, params, signal),
  };
}

const TOOLS = [
  teddyTool(
    "filter_laptops",
    "กรองตามสเปก",
    "กรองโน้ตบุ๊คทุกรุ่นในข้อมูลด้วยสเปกจริง แล้วเรียงผล ใช้เมื่อคำถามมีเงื่อนไขชัดเจน เช่น การ์ดจอ/ซีพียูรุ่นที่ระบุ " +
      "RAM ขั้นต่ำ คะแนนเบนช์มาร์กขั้นต่ำ หรือช่วงราคา หรือเมื่อต้องการรายการ 'ทุกรุ่น' ที่ตรงเงื่อนไข " +
      "ช่องข้อความ (gpu, cpu, display, storage, os) จับคู่แบบมีคำนั้นอยู่ในสเปก เช่น gpu='RTX 4060'",
    Type.Object({
      gpu: Type.Optional(Type.String({ description: "คำที่ต้องมีในชื่อการ์ดจอ เช่น 'RTX 4060', 'RTX 50', 'Arc'" })),
      cpu: Type.Optional(Type.String({ description: "คำที่ต้องมีในชื่อซีพียู เช่น 'Ryzen 7', 'Core Ultra 9'" })),
      display: Type.Optional(Type.String({ description: "คำที่ต้องมีในสเปกหน้าจอ เช่น 'OLED', '16.0', '165Hz'" })),
      storage: Type.Optional(Type.String({ description: "คำที่ต้องมีในสเปกความจุ เช่น '1TB'" })),
      os: Type.Optional(Type.String({ description: "คำที่ต้องมีในระบบปฏิบัติการ เช่น 'Windows 11 Pro'" })),
      min_ram_gb: Type.Optional(Type.Number({ description: "RAM ขั้นต่ำ (GB)" })),
      min_cpu_score: Type.Optional(Type.Number({ description: "คะแนน CPU ขั้นต่ำ" })),
      min_gpu_score: Type.Optional(Type.Number({ description: "คะแนน GPU ขั้นต่ำ" })),
      ...price,
      sort_by: Type.Optional(Type.Union(
        [Type.Literal("price_asc"), Type.Literal("price_desc"), Type.Literal("gpu_score"), Type.Literal("cpu_score")],
        { description: "เรียงผล: price_asc (ถูกก่อน, ค่าเริ่มต้น), price_desc, gpu_score, cpu_score" },
      )),
      limit,
    }),
  ),
  teddyTool(
    "search_laptops",
    "ค้นตามความหมาย",
    "ค้นเอกสารสเปกและรีวิวผู้ใช้จริงตามความหมายของคำถาม เช่น การใช้งาน (เล่นเกม ตัดต่อ เรียน) " +
      "หรือความรู้สึกจากรีวิว (แบตอึด เครื่องเบา จอสวย เสียงพัดลม) คืนสเปก รีวิว และส่วนต่างงบที่คำนวณแล้ว",
    Type.Object({
      query: Type.String({ description: "สิ่งที่ต้องการค้น เป็นภาษาไทยหรืออังกฤษก็ได้" }),
      ...price,
      limit,
    }),
  ),
  teddyTool(
    "get_laptop_details",
    "ดูรายละเอียดรุ่น",
    "ดูสเปกเต็มและรีวิวทั้งหมดของรุ่นที่ระบุชื่อ (พิมพ์บางส่วนของชื่อได้ เช่น 'Legion 5' หรือ 'TUF A15')",
    Type.Object({
      name: Type.String({ description: "ชื่อรุ่นหรือบางส่วนของชื่อรุ่น" }),
    }),
  ),
];

// ---------------------------------------------------------------- agent run

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function assistantText(message) {
  return (message?.content ?? [])
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("")
    .trim();
}

// แปลง event ของ pi เป็นบรรทัดสั้น ๆ ให้ teddy ส่งต่อถึงหน้าเว็บ (/chat/stream)
// ไม่ส่ง message ทั้งก้อนทุก token — pi แนบ partial message มากับทุก event ซึ่งยาวขึ้นเรื่อย ๆ
// message_start: เริ่มข้อความใหม่ของ AI — ข้อความรอบก่อน (เช่น "ขอค้นเพิ่มนะครับ" ก่อนเรียก tool)
// ไม่ใช่คำตอบสุดท้าย ฝั่ง teddy จะล้างทิ้งแล้วเริ่มนับใหม่ ตรงกับ assistantText(last) ตอนจบ
function streamEvent(event) {
  switch (event.type) {
    case "message_start":
      return event.message?.role === "assistant" ? { type: "message_start" } : null;
    case "message_update": {
      const e = event.assistantMessageEvent;
      return e?.type === "text_delta" && e.delta ? { type: "text", delta: e.delta } : null;
    }
    case "tool_execution_start":
      return { type: "tool_start", id: event.toolCallId, name: event.toolName, args: event.args ?? {} };
    case "tool_execution_end":
      return {
        type: "tool_end", id: event.toolCallId, name: event.toolName,
        ok: !event.isError, ms: event.durationMs ?? null,
      };
    default:
      return null;
  }
}

async function runChat({ provider: cfg, system_prompt: systemPrompt, prompt }, { onEvent, signal } = {}) {
  if (!cfg?.id) throw new HttpError(400, "ต้องส่ง provider มาด้วย");
  if (!prompt) throw new HttpError(400, "ต้องส่ง prompt มาด้วย");

  const model = resolveModel(cfg);
  const toolsUsed = [];
  let toolCalls = 0;
  let turns = 0;

  const agent = new Agent({
    initialState: {
      systemPrompt: systemPrompt || "",
      model,
      thinkingLevel: cfg.thinking || "off",
      tools: TOOLS,
    },
    streamFn: (m, context, options) =>
      models.streamSimple(m, context, { ...options, temperature: TEMPERATURE, maxTokens: cfg.max_tokens || 2048 }),
    getApiKey: () => cfg.api_key || undefined,
    onPayload: DEBUG
      ? (payload) => {
          const roles = (payload?.messages ?? []).map((m) => m.role).join(",");
          const tools = (payload?.tools ?? []).map((t) => t.function?.name ?? t.name).join(",");
          console.log(`payload model=${payload?.model} roles=${roles} tools=${tools || "-"} ` +
            `tool_choice=${JSON.stringify(payload?.tool_choice ?? null)}`);
          return undefined;
        }
      : undefined,
    beforeToolCall: async () => {
      toolCalls += 1;
      // ครบโควตา -> ถอด tool ออกจากรอบถัดไป โมเดลจะเหลือทางเดียวคือตอบจากข้อมูลที่ค้นมาแล้ว
      if (toolCalls >= MAX_TOOL_CALLS) agent.state.tools = [];
      return undefined;
    },
    finishTurn: async ({ message }) => {
      if (message.stopReason === "error" || message.stopReason === "aborted") return undefined;
      turns += 1;
      return turns >= MAX_TURNS ? { action: "end" } : undefined;
    },
  });

  agent.subscribe((event) => {
    if (event.type === "tool_execution_end") {
      toolsUsed.push({ name: event.toolName, ok: !event.isError, ms: event.durationMs ?? null });
    }
    const out = onEvent && streamEvent(event);
    if (out) onEvent(out);
  });

  // ผู้เรียกตัดสาย (ผู้ใช้ปิดหน้าเว็บกลางคัน) -> หยุด agent ไม่ให้เผาโควตา LLM ต่อโดยไม่มีคนรอ
  const stop = () => agent.abort();
  signal?.addEventListener("abort", stop, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    stop();
  }, RUN_TIMEOUT_MS);
  try {
    await agent.prompt(prompt);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", stop);
  }

  const last = [...agent.state.messages].reverse().find((m) => m.role === "assistant");
  if (signal?.aborted) throw new HttpError(499, "teddy ปิดการเชื่อมต่อก่อนตอบเสร็จ");
  // หมดเวลาแล้ว abort ตรงไหนก็ได้ (ระหว่าง LLM ตอบหรือรอ tool) — stopReason อาจเป็น "error"
  // พร้อมข้อความ "This operation was aborted" ซึ่งอ่านไม่รู้เรื่อง จึงเช็คจาก timer ของเราเองก่อน
  if (timedOut || last?.stopReason === "aborted") {
    throw new HttpError(504, `ใช้เวลาเกิน ${RUN_TIMEOUT_MS / 1000} วินาที`);
  }
  if (!last) throw new HttpError(502, agent.state.errorMessage || "โมเดลไม่ได้ตอบกลับ");
  if (last.stopReason === "error") throw new HttpError(502, last.errorMessage || "โมเดลตอบ error");

  return { message: assistantText(last), tools: toolsUsed, turns, model: `${model.provider}/${model.id}` };
}

// ---------------------------------------------------------------- http

function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new HttpError(413, "request ใหญ่เกินไป"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"));
      } catch {
        reject(new HttpError(400, "body ต้องเป็น JSON"));
      }
    });
    req.on("error", reject);
  });
}

function send(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

// /chat/stream: ตอบเป็น NDJSON (JSON หนึ่งบรรทัดต่อ event) ทยอยส่งระหว่าง agent ทำงาน
// แล้วปิดท้ายด้วย {"type":"done", ...ผลแบบเดียวกับ /chat} หรือ {"type":"error", status, error}
// teddy ส่งต่อให้หน้าเว็บโชว์ว่ากำลังเรียก tool อะไร และพิมพ์คำตอบให้เห็นทีละส่วน
async function streamChat(body, res, signal) {
  res.writeHead(200, { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-cache" });
  const write = (event) => {
    if (!res.writableEnded && !res.destroyed) res.write(JSON.stringify(event) + "\n");
  };
  const result = await runChat(body, { onEvent: write, signal });
  write({ type: "done", ...result });
  res.end();
  return result;
}

function logOk(kind, providerId, result, started) {
  console.log(`${kind} ok provider=${providerId} tools=${result.tools.map((t) => t.name).join(",") || "-"} ` +
    `turns=${result.turns} ms=${Date.now() - started}`);
}

const server = http.createServer(async (req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    return send(res, 200, { ok: true, tools: TOOLS.map((t) => t.name) });
  }
  const streaming = req.url === "/chat/stream";
  if (req.method !== "POST" || !(streaming || req.url === "/chat")) return send(res, 404, { error: "not found" });

  const kind = streaming ? "stream" : "chat";
  const started = Date.now();
  let providerId = "?";
  // close ก่อนตอบจบ = teddy ตัดสาย (ผู้ใช้ปิดหน้า / กด "หยุด" / worker รีสตาร์ต) -> สั่งหยุด agent
  const cancel = new AbortController();
  res.on("close", () => {
    if (!res.writableFinished) cancel.abort();
  });
  try {
    const body = await readJson(req);
    providerId = body?.provider?.id ?? "?";
    if (streaming) {
      logOk(kind, providerId, await streamChat(body, res, cancel.signal), started);
    } else {
      const result = await runChat(body, { signal: cancel.signal });
      logOk(kind, providerId, result, started);
      send(res, 200, result);
    }
  } catch (error) {
    const status = error instanceof HttpError ? error.status : 500;
    console.error(`${kind} fail provider=${providerId} status=${status} ms=${Date.now() - started}: ${error.message}`);
    // stream ส่ง header 200 ไปแล้ว เปลี่ยน status ไม่ได้ -> แจ้ง error เป็น event สุดท้ายแทน
    if (res.headersSent) {
      if (!res.writableEnded && !res.destroyed) res.end(JSON.stringify({ type: "error", status, error: error.message }) + "\n");
    } else {
      send(res, status, { error: error.message });
    }
  }
});

server.listen(PORT, () => {
  console.log(`pi-agent พร้อมที่ :${PORT} (tools -> ${TOOLS_URL}, สูงสุด ${MAX_TOOL_CALLS} tool calls / ${MAX_TURNS} turns)`);
  if (!INTERNAL_TOKEN) console.warn("คำเตือน: ไม่ได้ตั้ง INTERNAL_TOKEN — teddy จะปฏิเสธทุก tool call");
});
