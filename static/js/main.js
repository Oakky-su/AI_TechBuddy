/* Tech Buddy — ฝั่งหน้าเว็บ
   หน้าที่: ส่งข้อความไป /chat/stream (โชว์สถานะ + คำตอบที่กำลังพิมพ์), เรนเดอร์คำตอบ (markdown),
   จัดการธีม และเก็บประวัติแชทใน localStorage */

const messagesEl   = document.getElementById('messages');
const emptyState   = document.getElementById('emptyState');
const userInput    = document.getElementById('userInput');
const sendBtn      = document.getElementById('sendBtn');
const newChatBtn   = document.getElementById('newChatBtn');
const themeToggle  = document.getElementById('themeToggle');
const historyList  = document.getElementById('historyList');
const historyEmpty = document.getElementById('historyEmpty');
const clearAllBtn  = document.getElementById('clearAllBtn');
const clearAllLabel = document.getElementById('clearAllLabel');
const aiPicker     = document.getElementById('aiPicker');
const aiSelect     = document.getElementById('aiSelect');
const engineToggle = document.getElementById('engineToggle');
const engineText   = engineToggle.querySelector('.engine-text');

const STORE_KEY = 'tb-chats';
const AI_KEY = 'tb-ai';
let chats = [];        // [{ id, title, messages: [{role, content, meta?}] }]
let currentId = null;
let isLoading = false;

/* AI ที่เลือก — provider = id ใน providers.json, engine = 'pi' (Pi Agent) | 'classic' (RAG)
   ถ้าโหลดรายการไม่ได้ จะส่ง null ไปให้เซิร์ฟเวอร์ใช้ค่าเริ่มต้นเอง */
let providers = [];
let aiChoice = { provider: null, engine: null };

// ชื่อโหมดชุดเดียว ใช้ทั้งปุ่มสลับ, tooltip และบรรทัด meta ใต้คำตอบ
// ชื่อโหมดที่ผู้ใช้เห็น (id 'pi' / 'classic' คือค่าที่ส่งให้เซิร์ฟเวอร์ ห้ามเปลี่ยน)
// ใช้คำธรรมดาแทน "Pi Agent / RAG" เพราะคนซื้อโน้ตบุ๊คไม่รู้จักศัพท์พวกนั้น
const ENGINE_NAME = { pi: 'ค้นละเอียด', classic: 'ค้นเร็ว' };
// โหมดที่เซิร์ฟเวอร์เปิดให้ใช้ (ALLOWED_ENGINES) — เหลือโหมดเดียวเมื่อไหร่ หน้าเว็บจะไม่พูดถึงโหมดเลย
let serverEngines = ['pi', 'classic'];
const TOOL_NAME = {
  filter_laptops: 'กรองสเปก',
  search_laptops: 'ค้นสเปก/รีวิว',
  get_laptop_details: 'ดูรายละเอียดรุ่น',
};

/* ───────────── ธีมสว่าง / มืด ───────────── */
// aria-pressed = โหมดมืดเปิดอยู่ไหม (ธีมเริ่มต้นถูกตั้งด้วยสคริปต์ใน <head> ก่อนไฟล์นี้)
const syncThemeToggle = () =>
  themeToggle.setAttribute('aria-pressed', String(document.documentElement.dataset.theme === 'dark'));
syncThemeToggle();

themeToggle.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  syncThemeToggle();
  try { localStorage.setItem('tb-theme', next); } catch { /* โหมดส่วนตัวบางเบราว์เซอร์เขียนไม่ได้ */ }
});

/* ───────────── เรนเดอร์ markdown อย่างปลอดภัย ─────────────
   คำตอบจากโมเดลเป็น markdown (**ตัวหนา**, bullet) แต่ห้ามเอาไปใส่ innerHTML ดิบ ๆ
   ขั้นตอน: escape HTML ก่อน -> แล้วค่อยแปลง markdown เป็นแท็กที่เราสร้างเอง
   ทำแบบนี้แท็กที่ผู้ใช้หรือโมเดลพิมพ์มาจะกลายเป็นข้อความธรรมดา ไม่ถูกรันเป็น HTML */
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function inlineMd(s) {
  return s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split('\n');
  let html = '', openList = null;
  const closeList = () => { if (openList) { html += `</${openList}>`; openList = null; } };

  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;

    if ((m = line.match(/^\s*#{1,6}\s+(.*)$/))) {            // หัวข้อ
      closeList(); html += `<h4>${inlineMd(m[1])}</h4>`;
    } else if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {      // bullet
      if (openList !== 'ul') { closeList(); html += '<ul>'; openList = 'ul'; }
      html += `<li>${inlineMd(m[1])}</li>`;
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {    // ลำดับตัวเลข
      if (openList !== 'ol') { closeList(); html += '<ol>'; openList = 'ol'; }
      html += `<li>${inlineMd(m[1])}</li>`;
    } else if (!line.trim()) {
      closeList();
    } else {
      closeList(); html += `<p>${inlineMd(line)}</p>`;
    }
  }
  closeList();
  return html;
}

/* ───────────── วาดข้อความลงหน้าจอ ───────────── */
const BOT_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="12" rx="2"/>
  <path d="M2 20h20"/></svg>`;

/* "Wave AI · ค้นละเอียด · ค้นเพิ่ม 2 ครั้ง · 3.4 วิ" — บอกว่าคำตอบนี้มาจาก AI ตัวไหน โหมดไหน */
function metaText(meta) {
  const showMode = serverEngines.length > 1;   // โหมดเดียว = ไม่ต้องบอกชื่อโหมด
  const parts = [meta.provider_label, showMode ? ENGINE_NAME[meta.engine] || meta.engine : null];
  if (showMode && meta.engine === 'pi') {
    const tools = meta.tools || [];
    parts.push(tools.length ? `ค้นเพิ่ม ${tools.length} ครั้ง` : 'ตอบจากข้อมูลที่ค้นไว้');
  }
  if (meta.elapsed != null) parts.push(`${meta.elapsed} วิ`);
  return parts.filter(Boolean).join(' · ');
}

function appendMsg(role, content, isError = false, meta = null) {
  hideEmptyState();
  const el = buildMsg(role, content, isError, meta);
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

/* เลื่อนลงให้มากที่สุดเท่าที่หัวข้อความ el ยังอยู่ในจอ — คำตอบสั้นก็เลื่อนสุดล่างตามปกติ
   คำตอบยาว (โดยเฉพาะบนมือถือ) จะหยุดที่บรรทัดแรก ไม่พาผู้ใช้ไปโผล่ท้ายคำตอบ */
function revealMsg(el) {
  const top = el.getBoundingClientRect().top - messagesEl.getBoundingClientRect().top + messagesEl.scrollTop;
  messagesEl.scrollTop = Math.min(messagesEl.scrollHeight, top - 12);
}

function buildMsg(role, content, isError = false, meta = null) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role === 'user' ? 'user-msg' : 'bot-msg'}`;

  const av = document.createElement('div');
  av.className = `avatar ${role === 'user' ? 'user-avatar' : 'bot-avatar'}`;
  if (role === 'user') av.textContent = 'คุณ';
  else av.innerHTML = BOT_ICON;

  const bubble = document.createElement('div');
  bubble.className = 'bubble' + (isError ? ' error-bubble' : '');
  // ข้อความผู้ใช้ใส่เป็น text ล้วน ส่วนคำตอบ AI ค่อยแปลง markdown (escape มาแล้ว)
  if (role === 'user') bubble.textContent = content;
  else bubble.innerHTML = renderMarkdown(content);

  if (meta) {
    const line = document.createElement('p');
    line.className = 'bubble-meta';
    line.textContent = metaText(meta);
    const tools = (meta.tools || []).map(t => TOOL_NAME[t] || t);
    if (tools.length) line.title = `เครื่องมือที่ใช้: ${tools.join(', ')}`;
    bubble.appendChild(line);
  }

  wrap.append(av, bubble);
  return wrap;
}

/* ───────────── คำตอบที่กำลังพิมพ์ (จาก /chat/stream) ─────────────
   ระหว่างรอ: จุดสามจุด + สถานะจากเซิร์ฟเวอร์ เช่น "กำลังกรองสเปก: RTX 4060 · ไม่เกิน 40,000 บาท…"
   พอ AI เริ่มพิมพ์: โชว์ข้อความแทนสถานะ แล้วตอนจบแทนที่ด้วยข้อความปกติ (หน้าตาเดียวกับ appendMsg)
   endpoint ของ LLM ส่ง token มาเป็นก้อน ๆ ห่างกันเกือบวินาที จึงค่อย ๆ เผยข้อความทีละนิดให้ลื่นตา */
const nearBottom = () => messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 80;

function createLiveReply() {
  hideEmptyState();
  const wrap = document.createElement('div');
  wrap.className = 'msg bot-msg live-msg';
  wrap.innerHTML =
    `<div class="avatar bot-avatar">${BOT_ICON}</div>
     <div class="bubble">
       <div class="live-status" role="status">
         <span class="dot-anim"></span><span class="dot-anim"></span><span class="dot-anim"></span>
         <span class="typing-label"></span>
       </div>
       <div class="live-body" hidden></div>
     </div>`;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  const statusEl = wrap.querySelector('.live-status');
  const labelEl = wrap.querySelector('.typing-label');
  const bodyEl = wrap.querySelector('.live-body');
  const instant = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let target = '';   // ข้อความที่ได้รับมาแล้วทั้งหมด
  let shown = 0;     // โชว์ไปแล้วกี่ตัวอักษร
  let frame = 0;

  // สถานะเข้าคิว โชว์อย่างน้อย 0.6 วิต่ออัน — tool บางตัวเสร็จในเสี้ยววินาที ถ้าเปลี่ยนทันทีจะอ่านไม่ทัน
  const STATUS_MIN_MS = 600;
  const statusQueue = [];
  let lastStatus = '', statusTimer = 0;
  function nextStatus() {
    statusTimer = 0;
    if (!statusQueue.length) return;
    labelEl.textContent = statusQueue.shift();
    statusTimer = setTimeout(nextStatus, STATUS_MIN_MS);
  }

  function paint() {
    frame = 0;
    const left = target.length - shown;
    // ไล่ตามทีละ 1/8 ของที่ค้าง — ก้อนใหญ่ก็ตามทันในเสี้ยววินาที ก้อนเล็กดูเหมือนพิมพ์
    if (left > 0) shown += instant ? left : Math.max(2, Math.ceil(left / 8));
    if (shown < target.length && /[\uD800-\uDBFF]/.test(target[shown - 1])) shown++;  // ไม่ตัดกลางอีโมจิ

    const stick = nearBottom();
    statusEl.hidden = shown > 0;
    bodyEl.hidden = shown === 0;
    bodyEl.innerHTML = renderMarkdown(target.slice(0, shown));
    if (stick) revealMsg(wrap);
    if (shown < target.length) frame = requestAnimationFrame(paint);
  }
  const schedule = () => { if (!frame) frame = requestAnimationFrame(paint); };

  // แทนที่กล่องนี้ด้วยข้อความปกติ (คำตอบสุดท้าย / error / หยุดกลางคัน)
  function settle(el) {
    cancelAnimationFrame(frame);
    clearTimeout(statusTimer);
    const stick = nearBottom();
    wrap.replaceWith(el);
    if (stick) revealMsg(el);
    return el;
  }

  return {
    setStatus(text) {
      if (text === lastStatus) return;
      lastStatus = text;
      statusQueue.push(text);
      if (!statusTimer) nextStatus();
    },
    append(text) { target += text; schedule(); },
    // เซิร์ฟเวอร์ทิ้งข้อความเดิม (AI ขึ้นข้อความใหม่หลังเรียก tool / เจอ <think>) — ถ้าต่อจากที่โชว์อยู่ได้ก็ไม่ต้องเริ่มใหม่
    replace(text) {
      shown = text.startsWith(target.slice(0, shown)) ? Math.min(shown, text.length) : 0;
      target = text;
      schedule();
    },
    finish(message, meta, debug) { settle(withDebug(buildMsg('bot', message, false, meta), debug)); },
    fail(error, onRetry, debug) { return settle(withDebug(buildError(error, onRetry), debug)); },
    // ผู้ใช้กดหยุด — เก็บส่วนที่พิมพ์มาแล้วไว้ให้อ่าน แต่บอกชัดว่าไม่ครบและไม่ได้บันทึก
    stop() {
      const text = target.trim();
      const el = buildMsg('bot', text || 'หยุดแล้ว');
      const note = document.createElement('p');
      note.className = 'bubble-meta';
      note.textContent = text ? 'หยุดก่อนตอบเสร็จ — คำตอบนี้ไม่ถูกบันทึก' : 'แก้คำถามในช่องพิมพ์แล้วส่งใหม่ได้เลย';
      el.querySelector('.bubble').appendChild(note);
      settle(el);
    },
  };
}

/* ───────────── error ─────────────
   แปลง error ให้ผู้ใช้อ่านรู้เรื่อง ข้อความดิบเก็บไว้เป็นบรรทัด "รายละเอียด" ตัวเล็ก (ไว้แจ้งผู้ดูแล)
   ข้อความที่เซิร์ฟเวอร์เขียนเป็นภาษาไทยถึงผู้ใช้อยู่แล้ว (ถามถี่เกิน, AI ที่ยังไม่ตั้งค่า ...) ใช้ตามเดิม */
const FRIENDLY_ERRORS = [
  [/ใช้เวลาเกิน|timed? ?out/i, 'AI ใช้เวลานานเกินไป ลองอีกครั้ง หรือถามให้สั้นลงนะครับ'],
  [/pi-agent/i, `โหมด${ENGINE_NAME.pi}ไม่พร้อมใช้งานตอนนี้ ลองอีกครั้ง หรือสลับไปโหมด${ENGINE_NAME.classic}ที่ปุ่มด้านบน`],
  [/^Vector search error/, 'ระบบค้นข้อมูลสินค้าขัดข้องชั่วคราว ลองอีกครั้งในอีกสักครู่นะครับ'],
  [/error/i, 'AI ตอบไม่สำเร็จ ลองอีกครั้ง หรือเปลี่ยนไปใช้ AI ตัวอื่นจากเมนูด้านบน'],
];
const NET_DOWN = 'เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ ตรวจสอบอินเทอร์เน็ตแล้วลองอีกครั้งนะครับ';
const NET_DROPPED = 'การเชื่อมต่อหลุดระหว่างตอบ ลองอีกครั้งนะครับ';

function friendlyError(raw) {
  for (const [pattern, text] of FRIENDLY_ERRORS) {
    if (pattern.test(raw)) return { text, detail: raw };
  }
  return /[฀-๿]/.test(raw) ? { text: raw } : { text: 'เกิดข้อผิดพลาดที่ไม่คาดคิด ลองอีกครั้งนะครับ', detail: raw };
}

function buildError({ text, detail }, onRetry) {
  const el = buildMsg('bot', text, true);
  const bubble = el.querySelector('.bubble');
  if (onRetry) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'retry-btn';
    btn.textContent = 'ลองอีกครั้ง';
    btn.addEventListener('click', onRetry);
    bubble.appendChild(btn);
  }
  if (detail) {
    const line = document.createElement('p');
    line.className = 'bubble-meta';
    line.textContent = `รายละเอียด: ${detail}`;
    bubble.appendChild(line);
  }
  return el;
}

/* ───────────── โหมด debug: แผงกระบวนการคิดของ AI (ด้านขวา) ─────────────
   ไล่ตามลำดับที่ระบบทำจริง: ตีความคำถาม -> embed -> ค้นฐานข้อมูลเวกเตอร์ -> ปรับผลค้น
   -> ข้อมูลที่ส่งให้ AI -> prompt -> AI ตอบ — อยู่ในแผงข้างแชท ไม่ปนกับบทสนทนา
   ทุกคำตอบมีปุ่ม "กระบวนการคิด" ในบรรทัดเล็กใต้คำตอบ กดแล้วแผงเปิดไปที่คำตอบนั้น (กดซ้ำ = ปิด)
   ไม่มีสวิตช์เปิด/ปิดบนหน้าเว็บ — ผู้ดูแลเปิดทั้งระบบด้วย DEBUG_PANEL=true ใน .env
   ไม่เก็บลงประวัติแชท (ก้อนใหญ่ ล้น localStorage) เปิดแชทเก่าจึงไม่มีปุ่มนี้ */
let debugOn = false;               // เซิร์ฟเวอร์เปิด DEBUG_PANEL ไว้ไหม (อ่านจาก /providers)
let lastChip = null;               // ปุ่มที่เปิดแผงล่าสุด — ปิดแผงแล้วคืนโฟกัสให้ปุ่มนี้
const debugDrawer   = document.getElementById('debugDrawer');
const debugBody     = document.getElementById('debugBody');
const debugQuestion = document.getElementById('debugQuestion');
const debugTotals   = document.getElementById('debugTotals');
const debugClose    = document.getElementById('debugClose');
// จอกว้างพอวางแผงข้างแชท (sidebar 268 + แชท ~470 + แผง 440) — แคบกว่านี้แผงเลื่อนทับแชทแทน
const wideScreen = window.matchMedia('(min-width: 1180px)');
const chipEntry = new WeakMap();   // ปุ่มใต้คำตอบ -> { question, trace } ของคำตอบนั้น
let shownEntry = null;             // คำตอบที่แผงโชว์อยู่

const SOURCE_NAME = {
  vector: 'ค้นเวกเตอร์', exact: 'ตรงชื่อรุ่น/สเปก', near_price: 'ราคาใกล้งบ',
  extreme: 'ถูก/แพงสุด', summary: 'ภาพรวมร้าน',
};
const KIND_NAME = { spec: 'สเปก', review: 'รีวิว', summary: 'ภาพรวม', note: 'ผลค้นตรง' };
const FINISH_NAME = { stop: 'ตอบจบปกติ', length: 'โทเคนหมดก่อนตอบจบ' };

const fmtNum = n => (n == null ? '–' : Number(n).toLocaleString('en-US'));
const short = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);
// value ต้องเป็น HTML ที่ escape แล้ว
const debugRow = (label, value) => `<dt>${escapeHtml(label)}</dt><dd>${value}</dd>`;

function debugSearch(s) {
  const rows = s.hits.map((h, i) => {
    const pct = Math.max(0, Math.min(100, h.similarity * 100)).toFixed(0);
    return `<tr><td class="num">${i + 1}</td><td>${escapeHtml(h.name || '')}</td>
      <td class="num">${fmtNum(h.price)}</td>
      <td class="num"><span class="dbg-bar" style="--w:${pct}%"></span>${h.similarity.toFixed(3)}</td></tr>`;
  }).join('');
  return `<p class="dbg-sub">${escapeHtml(KIND_NAME[s.type] || s.type)} · ขอ ${s.want} อันดับ · ${escapeHtml(s.filter)}</p>
    <div class="dbg-scroll"><table class="dbg-table">
      <thead><tr><th>#</th><th>รุ่น</th><th>ราคา</th><th>ความใกล้เคียง</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="4">ไม่เจอเอกสารที่ตรงเงื่อนไข</td></tr>'}</tbody>
    </table></div>`;
}

/* ขั้นตอนทั้งหมดเป็น HTML (escape แล้ว) + ตัวเลขสรุปบนหัวแผง */
function debugStepsHtml(t) {
  const q = t.query || {};
  const ai = t.llm || {};
  const usage = ai.usage || {};
  const list = arr => (arr?.length ? arr.map(escapeHtml).join(', ') : '–');
  const yesNo = v => (v ? 'ใช่' : 'ไม่');
  const budgetFrom = { message: 'อ่านจากข้อความนี้', session: 'จำไว้จากข้อความก่อนหน้า' }[q.budget_from];
  const steps = [];

  steps.push(['ตีความคำถาม', `<dl class="dbg-kv">
    ${debugRow('งบ (เพดาน)', q.budget != null
      ? `${fmtNum(q.budget)} บาท${budgetFrom ? ` <span class="dbg-dim">· ${budgetFrom}</span>` : ''}` : 'ไม่ได้บอก')}
    ${debugRow('ราคาขั้นต่ำ', q.min_price != null ? `${fmtNum(q.min_price)} บาท` : '–')}
    ${debugRow('ขอเทียบกี่รุ่น', q.model_count ? `${q.model_count} รุ่น` : '–')}
    ${debugRow('ถามรุ่นถูกที่สุด / แพงที่สุด', `${yesNo(q.cheapest)} / ${yesNo(q.priciest)}`)}
    ${debugRow('การ์ดจอที่จับได้', list(q.gpu_terms?.map(g => g.toUpperCase())))}
    ${debugRow('ชื่อรุ่นที่จับได้', list(q.asked_models))}
    ${debugRow('ประวัติแชทที่ส่งให้ AI', `${q.history_msgs || 0} ข้อความ`)}
  </dl>`]);

  if (t.embed) {
    const rest = t.embed.dims - t.embed.preview.length;
    steps.push(['แปลงคำถามเป็นเวกเตอร์ (embedding)',
      `<p>${escapeHtml(t.embed.model)} · ${fmtNum(t.embed.dims)} มิติ · ${fmtNum(t.embed.ms)} ms${t.embed.load_ms
        ? ` <span class="dbg-dim">(ก่อนหน้านั้นโหลดโมเดลครั้งแรก ${fmtNum(t.embed.load_ms)} ms)</span>` : ''}</p>
       <code class="dbg-vec">[${t.embed.preview.map(v => v.toFixed(4)).join(', ')}, … อีก ${fmtNum(rest)} ค่า]</code>`]);
  }
  if (t.searches) {
    steps.push(['ค้นในฐานข้อมูลเวกเตอร์', t.searches.map(debugSearch).join('')
      + '<p class="dbg-dim">ความใกล้เคียง = cosine similarity ระหว่างเวกเตอร์คำถามกับเอกสาร (1 = ความหมายตรงกันที่สุด)</p>']);
  }
  if (t.adjust) {
    steps.push(['ปรับผลค้นด้วยกฎของระบบ', t.adjust.length
      ? `<ul>${t.adjust.map(a => `<li>${escapeHtml(a)}</li>`).join('')}</ul>`
      : '<p class="dbg-dim">ไม่มี ใช้ผลค้นตามลำดับความใกล้เคียงตรง ๆ</p>']);
  }
  if (t.context) {
    steps.push([`ข้อมูลที่ส่งให้ AI (${t.context.length} ชิ้น ตามลำดับใน prompt)`,
      `<ol class="dbg-docs">${t.context.map(c => `<li><details><summary>
        <span class="dbg-tag">${escapeHtml(SOURCE_NAME[c.source] || c.source)}</span>
        <span class="dbg-tag dbg-tag-soft">${escapeHtml(KIND_NAME[c.kind] || c.kind)}</span>
        ${escapeHtml(short(c.text.split(' | ')[0], 80))}</summary><p>${escapeHtml(c.text)}</p></details></li>`).join('')}</ol>`]);
  }
  if (t.prompt) {
    steps.push(['Prompt ที่ส่งให้ AI', `<details><summary>ดู prompt ทั้งก้อน (${fmtNum(t.prompt.length)} ตัวอักษร)</summary>
      <button type="button" class="dbg-copy">คัดลอก prompt</button><pre class="dbg-pre">${escapeHtml(t.prompt)}</pre></details>`]);
  }
  if (t.llm) {
    const tokens = usage.total_tokens != null
      ? `${fmtNum(usage.prompt_tokens)} + ${fmtNum(usage.completion_tokens)} = ${fmtNum(usage.total_tokens)}` : '–';
    const thinking = ai.thinking
      ? `<details open><summary>ความคิดของ AI ก่อนตอบ</summary><pre class="dbg-pre">${escapeHtml(ai.thinking)}</pre></details>`
      : '<p class="dbg-dim">AI ตัวนี้ไม่ได้ส่งขั้นตอนคิดออกมา ตอบจาก prompt ด้านบนตรง ๆ</p>';
    steps.push(['AI เรียบเรียงคำตอบ', `<dl class="dbg-kv">
      ${debugRow('AI', escapeHtml(`${ai.provider} (${ai.model})`))}
      ${debugRow('temperature / max_tokens', `${ai.temperature} / ${fmtNum(ai.max_tokens)}`)}
      ${debugRow('เวลา', ai.ms != null ? `${fmtNum(ai.ms)} ms` : 'ยังไม่ได้คำตอบ')}
      ${debugRow('โทเคน (prompt + คำตอบ)', tokens)}
      ${debugRow('จบเพราะ', escapeHtml(FINISH_NAME[ai.finish_reason] || ai.finish_reason || '–'))}
    </dl>${thinking}`]);
  } else if (t.engine === 'pi') {
    const tools = (t.tools || []).map(x => TOOL_NAME[x.name] || x.name);
    steps.push(['AI โหมดค้นละเอียด', `<p class="dbg-dim">prompt และเวลาของ AI อยู่ฝั่ง Pi Agent ดูได้แค่เครื่องมือที่เรียกเพิ่ม</p>
      <p>${tools.length ? escapeHtml(tools.join(', ')) : 'ไม่ได้เรียกเครื่องมือเพิ่ม'}</p>`]);
  }

  const totals = [
    t.search_ms != null && `ค้น ${fmtNum(t.search_ms)} ms`,
    ai.ms != null && `AI ${fmtNum(ai.ms)} ms`,
    usage.total_tokens != null && `${fmtNum(usage.total_tokens)} โทเคน`,
  ].filter(Boolean).join(' · ');

  const html = `<ol class="dbg-steps">${steps.map(([title, body]) =>
    `<li><h3>${escapeHtml(title)}</h3>${body}</li>`).join('')}</ol>`;
  return { html, totals };
}

/* วาดแผงของคำตอบที่เลือก — entry = { question, trace } */
function renderDebug(entry) {
  shownEntry = entry;
  const { html, totals } = debugStepsHtml(entry.trace);
  debugQuestion.textContent = `“${entry.question}”`;
  debugTotals.textContent = totals;
  debugBody.innerHTML = html;
  debugBody.scrollTop = 0;
  debugBody.querySelector('.dbg-copy')?.addEventListener('click', async e => {
    try {
      await navigator.clipboard.writeText(entry.trace.prompt);
      e.target.textContent = 'คัดลอกแล้ว';
    } catch {
      e.target.textContent = 'คัดลอกไม่ได้ ลากคลุมข้อความแทนนะ';
    }
  });
}

/* ปุ่มใต้คำตอบไหนตรงกับที่แผงโชว์อยู่ ให้เห็นว่ากำลังดูคำตอบไหน */
function syncDebugChips() {
  messagesEl.querySelectorAll('.debug-chip').forEach(b =>
    b.setAttribute('aria-expanded', String(!debugDrawer.hidden && chipEntry.get(b) === shownEntry)));
}

function openDebugDrawer(open, { focus = false } = {}) {
  debugDrawer.hidden = !open;
  syncDebugChips();
  if (open && focus) debugClose.focus();
}

/* ปิดแผงและลืมคำตอบที่โชว์อยู่ — ใช้ตอนคำตอบหายจากจอ (แชทใหม่ / เปิดแชทเก่า) */
function closeDebug() {
  shownEntry = null;
  lastChip = null;
  openDebugDrawer(false);
}

/* ปุ่ม "กระบวนการคิด" ในบรรทัดเล็กใต้คำตอบ (ไม่มีข้อมูล debug = คืนกล่องเดิม) */
function withDebug(el, entry) {
  if (!entry?.trace) return el;
  const bubble = el.querySelector('.bubble');
  let line = bubble.querySelector('.bubble-meta:last-of-type');
  if (!line) {
    line = document.createElement('p');
    line.className = 'bubble-meta';
    bubble.appendChild(line);
  }
  const chip = document.createElement('button');
  chip.type = 'button';
  chip.className = 'debug-chip';
  chip.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 18h6M10 21h4"/>
    <path d="M12 3a6 6 0 00-3.6 10.8c.6.5 1 1.2 1 2V16h5.2v-.2c0-.8.4-1.5 1-2A6 6 0 0012 3z"/></svg>กระบวนการคิด`;
  chip.title = 'ดูว่าคำตอบนี้ AI ค้นเจออะไร เห็นข้อมูลอะไร และคิดยังไงก่อนตอบ';
  chip.setAttribute('aria-controls', 'debugDrawer');
  chip.setAttribute('aria-expanded', 'false');
  chipEntry.set(chip, entry);
  chip.addEventListener('click', () => {
    // กดซ้ำที่คำตอบที่เปิดดูอยู่ = ปิดแผง
    if (!debugDrawer.hidden && shownEntry === entry) {
      openDebugDrawer(false);
      return;
    }
    lastChip = chip;
    renderDebug(entry);
    openDebugDrawer(true, { focus: !wideScreen.matches });
  });
  line.appendChild(chip);
  return el;
}

debugClose.addEventListener('click', () => {
  openDebugDrawer(false);
  // แผงทับแชทอยู่ -> คืนโฟกัสให้ปุ่มใต้คำตอบที่เปิดมัน
  if (!wideScreen.matches && lastChip?.isConnected) lastChip.focus();
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !debugDrawer.hidden && !wideScreen.matches) debugClose.click();
});

/* อ่าน Server-Sent Events จาก fetch (EventSource ใช้กับ POST ไม่ได้) — event คั่นด้วยบรรทัดว่าง */
async function readEvents(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf('\n\n')) >= 0) {
      const data = buf.slice(0, cut).split('\n')
        .filter(l => l.startsWith('data:')).map(l => l.slice(5).trim()).join('\n');
      buf = buf.slice(cut + 2);
      if (data) onEvent(JSON.parse(data));
    }
  }
}

/* ให้เซิร์ฟเวอร์จำคำถาม-คำตอบนี้เป็นบริบทของคำถามถัดไป (ดู /chat/commit ใน app.py)
   ต้องยิงแยก เพราะ cookie ของ session ถูกส่งไปตั้งแต่สตรีมเริ่ม ก่อนจะรู้คำตอบ */
async function commitTurn(token) {
  if (!token) return;
  try {
    await fetch('/chat/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    });
  } catch { /* คุยต่อได้ แค่ AI อาจจำคำถามนี้ไม่ได้ */ }
}

function hideEmptyState() { emptyState.style.display = 'none'; }

function showEmptyState() {
  messagesEl.querySelectorAll('.msg').forEach(el => el.remove());
  closeDebug();   // คำตอบที่แผง debug โชว์อยู่หายไปจากจอแล้ว
  emptyState.style.display = '';
}

/* ───────────── ประวัติแชทใน localStorage ───────────── */
function loadChats() {
  try { chats = JSON.parse(localStorage.getItem(STORE_KEY)) || []; }
  catch { chats = []; }
}

function saveChats() {
  // เก็บแค่ 20 บทสนทนาล่าสุด กัน localStorage บวม
  chats = chats.slice(0, 20);
  localStorage.setItem(STORE_KEY, JSON.stringify(chats));
}

const TRASH_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
  stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/></svg>`;

function renderHistory() {
  historyList.innerHTML = '';
  historyEmpty.style.display = chats.length ? 'none' : '';
  clearAllBtn.hidden = chats.length === 0;
  resetClearConfirm();   // รายการเปลี่ยน -> ยกเลิกสถานะรอยืนยันที่ค้างอยู่

  for (const chat of chats) {
    const item = document.createElement('div');
    item.className = 'history-item' + (chat.id === currentId ? ' active' : '');
    item.title = chat.title;

    const label = document.createElement('span');
    label.className = 'history-title';
    label.textContent = chat.title;     // textContent -> ชื่อแชทไม่ถูกตีความเป็น HTML
    // เปิดด้วยคีย์บอร์ดได้ (Tab แล้ว Enter/Space) — ไม่ใช้ <button> ครอบทั้งแถว เพราะข้างในมีปุ่มลบอยู่แล้ว
    label.tabIndex = 0;
    label.setAttribute('role', 'button');
    if (chat.id === currentId) label.setAttribute('aria-current', 'true');
    label.addEventListener('keydown', e => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      e.preventDefault();
      openChat(chat.id, { refocus: true });
    });

    const del = document.createElement('button');
    del.className = 'history-del';
    del.type = 'button';
    del.title = 'ลบบทสนทนานี้';
    del.setAttribute('aria-label', `ลบบทสนทนา ${chat.title}`);
    del.innerHTML = TRASH_ICON;
    // stopPropagation: กดถังขยะต้องไม่ไปเปิดบทสนทนานั้นด้วย
    del.addEventListener('click', e => { e.stopPropagation(); deleteChat(chat.id); });

    item.append(label, del);
    item.addEventListener('click', () => openChat(chat.id));
    historyList.appendChild(item);
  }
}

/* ลบบทสนทนาเดียว — ถ้าลบตัวที่กำลังเปิดอยู่ ต้องล้างจอและ session ฝั่งเซิร์ฟเวอร์ด้วย
   ไม่งั้น AI จะยังจำบทสนทนาที่ผู้ใช้เพิ่งลบไปแล้ว */
function deleteChat(id) {
  if (isLoading) return;
  chats = chats.filter(c => c.id !== id);
  saveChats();

  if (currentId === id) {
    currentId = null;
    showEmptyState();
    fetch('/new_chat', { method: 'POST' }).catch(() => {});
  }
  renderHistory();
}

/* ล้างทั้งหมด — ใช้การกดยืนยันสองครั้งแทน confirm() เพราะ confirm() บล็อกทั้งหน้า */
let clearConfirmTimer = null;

function resetClearConfirm() {
  clearTimeout(clearConfirmTimer);
  clearAllBtn.classList.remove('confirming');
  clearAllBtn.dataset.confirm = '';
  clearAllLabel.textContent = 'ล้างประวัติทั้งหมด';
}

function clearAllChats() {
  if (isLoading) return;

  if (clearAllBtn.dataset.confirm !== '1') {
    clearAllBtn.dataset.confirm = '1';
    clearAllBtn.classList.add('confirming');
    clearAllLabel.textContent = `กดอีกครั้งเพื่อลบ ${chats.length} บทสนทนา`;
    clearConfirmTimer = setTimeout(resetClearConfirm, 4000);  // ไม่กดต่อใน 4 วิ = ยกเลิกเอง
    return;
  }

  chats = [];
  currentId = null;
  saveChats();
  showEmptyState();
  renderHistory();
  fetch('/new_chat', { method: 'POST' }).catch(() => {});
}

/* เปิดบทสนทนาเก่า: วาดข้อความกลับขึ้นจอ แล้วส่งให้เซิร์ฟเวอร์จำบริบทเดิมด้วย
   (ของเดิมรายการใน sidebar กดแล้วไม่เกิดอะไรขึ้นเลย) */
async function openChat(id, { refocus = false } = {}) {
  const chat = chats.find(c => c.id === id);
  if (!chat || isLoading) return;

  currentId = id;
  showEmptyState();   // ล้างข้อความเก่าออกจากจอ (appendMsg จะซ่อนการ์ดเริ่มต้นให้เอง)
  chat.messages.forEach(m => appendMsg(m.role, m.content, false, m.meta));
  renderHistory();
  // renderHistory สร้างรายการใหม่ทั้งหมด -> โฟกัสหลุด ผู้ใช้คีย์บอร์ดต้องได้กลับมาที่เดิม
  if (refocus) historyList.querySelector('.history-item.active .history-title')?.focus();

  try {
    await fetch('/restore', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history: chat.messages }),
    });
  } catch { /* คุยต่อได้อยู่ แค่ฝั่งเซิร์ฟเวอร์อาจจำบริบทเดิมไม่ได้ */ }
}

function recordMessage(role, content, meta = null) {
  if (currentId === null) {
    currentId = Date.now();
    chats.unshift({
      id: currentId,
      title: content.length > 34 ? content.slice(0, 34) + '…' : content,
      messages: [],
    });
  }
  const chat = chats.find(c => c.id === currentId);
  if (chat) {
    chat.messages.push(meta ? { role, content, meta } : { role, content });
    chat.messages = chat.messages.slice(-20);
  }
  saveChats();
  renderHistory();
}

/* คำถามที่ไม่ได้คำตอบ (error / กดหยุด) ไม่เก็บลงประวัติ — เปิดแชทนี้ทีหลังจะได้ไม่เจอคำถามค้าง ๆ */
function forgetLastQuestion(content) {
  const chat = chats.find(c => c.id === currentId);
  const last = chat?.messages.at(-1);
  if (!last || last.role !== 'user' || last.content !== content) return;
  chat.messages.pop();
  if (!chat.messages.length) {
    chats = chats.filter(c => c !== chat);
    currentId = null;
  }
  saveChats();
  renderHistory();
}

/* ระหว่างรอคำตอบ: ปุ่มส่งกลายเป็นปุ่ม "หยุด" และล็อกเมนูเลือก AI / โหมด
   (เปลี่ยนกลางคันไม่มีผลกับคำถามที่ส่งไปแล้ว ปล่อยให้กดได้จะชวนเข้าใจผิด) */
const SEND_ICON = sendBtn.innerHTML;
const STOP_ICON = `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
  <rect x="6" y="6" width="12" height="12" rx="2.5"/></svg>`;
let activeRequest = null;   // AbortController ของคำถามที่กำลังรอ

function setLoading(on) {
  isLoading = on;
  sendBtn.innerHTML = on ? STOP_ICON : SEND_ICON;
  sendBtn.title = on ? 'หยุดตอบ' : 'ส่งข้อความ';
  sendBtn.setAttribute('aria-label', sendBtn.title);
  sendBtn.classList.toggle('is-stop', on);
  aiSelect.disabled = on;
  if (on) engineToggle.disabled = true;
  else syncEngineToggle();   // คืนสถานะเดิม (และซ่อนปุ่มต่อถ้าเหลือโหมดเดียว)
}

/* ───────────── ส่งข้อความ ───────────── */
async function sendMessage(text) {
  const msg = (text ?? userInput.value).trim();
  if (!msg || isLoading) return;

  userInput.value = '';
  const request = new AbortController();
  activeRequest = request;
  setLoading(true);

  const userEl = appendMsg('user', msg);
  recordMessage('user', msg);
  const live = createLiveReply();
  live.setStatus('กำลังค้นข้อมูลในร้าน…');   // ตรงกับ event แรกของเซิร์ฟเวอร์ (ซ้ำกันจะถูกข้าม)

  let final = null, error = null, streaming = false;
  // โหมด debug: เก็บกระบวนการคิดของคำตอบนี้ไว้กับปุ่มใต้คำตอบ (แผงเปิดเมื่อกดปุ่มเท่านั้น)
  const entry = debugOn ? { question: msg, trace: null } : null;
  try {
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg, provider: aiChoice.provider, engine: aiChoice.engine,
        ...(debugOn && { debug: true }),
      }),
      signal: request.signal,   // กดหยุด -> ตัดสาย -> เซิร์ฟเวอร์สั่ง agent หยุดต่อเอง
    });
    if (!(res.headers.get('content-type') || '').startsWith('text/event-stream')) {
      // error ที่รู้ก่อนเริ่มตอบ (ถามถี่เกิน, ข้อความยาวเกิน ...) มาเป็น JSON แบบเดียวกับ /chat
      const data = await res.json().catch(() => ({}));
      error = friendlyError(data.error || `เซิร์ฟเวอร์ขัดข้อง (HTTP ${res.status}) ลองอีกครั้งนะครับ`);
    } else {
      streaming = true;
      await readEvents(res, ev => {
        if (ev.type === 'status') live.setStatus(ev.text);
        else if (ev.type === 'delta') live.append(ev.text);
        else if (ev.type === 'replace') live.replace(ev.text);
        else if (ev.type === 'done') final = ev;
        else if (ev.type === 'error') error = friendlyError(ev.error);
        if (ev.debug && entry) entry.trace = ev.debug;   // มากับ done / error ตอนเปิดโหมด debug
      });
      if (!final && !error) error = { text: NET_DROPPED };
    }
  } catch (err) {
    if (!request.signal.aborted) error = { text: streaming ? NET_DROPPED : NET_DOWN, detail: err.message };
  }

  if (final) {
    live.finish(final.message || '...', final.meta, entry);
    recordMessage('assistant', final.message || '', final.meta);
    await commitTurn(final.commit);
  } else {
    // ไม่ได้คำตอบ: ไม่เก็บคำถามลงประวัติ และคืนข้อความไว้ในช่องพิมพ์ให้แก้แล้วส่งใหม่ได้
    forgetLastQuestion(msg);
    if (!userInput.value) userInput.value = msg;
    if (request.signal.aborted) {
      live.stop();
    } else {
      const errorEl = live.fail(error, () => {
        if (isLoading) return;
        userEl.remove();
        errorEl.remove();
        if (userInput.value === msg) userInput.value = '';
        sendMessage(msg);
      }, entry);
    }
  }

  activeRequest = null;
  setLoading(false);
  userInput.focus();
}

/* ───────────── เริ่มแชทใหม่ ───────────── */
async function newChat() {
  if (isLoading) return;
  currentId = null;
  showEmptyState();
  renderHistory();
  userInput.focus();
  try { await fetch('/new_chat', { method: 'POST' }); } catch { /* ไม่เป็นไร */ }
}

/* ───────────── เลือก AI + โหมด ─────────────
   จำตัวเลือกไว้ใน localStorage (ห่อ try เพราะบางเบราว์เซอร์/โหมดส่วนตัวเขียนไม่ได้) */
function loadAiChoice() {
  try { return JSON.parse(localStorage.getItem(AI_KEY)) || {}; }
  catch { return {}; }
}

function saveAiChoice() {
  try { localStorage.setItem(AI_KEY, JSON.stringify(aiChoice)); } catch { /* ไม่เป็นไร */ }
}

const currentProvider = () => providers.find(p => p.id === aiChoice.provider);

function syncEngineToggle() {
  const p = currentProvider();
  // โหมดที่ใช้ได้กับ AI ตัวนี้ (เซิร์ฟเวอร์ตัดโหมดที่ผู้ดูแลปิดออกให้แล้ว)
  const engines = (p?.engines || serverEngines).filter(e => ENGINE_NAME[e]);
  if (engines.length && !engines.includes(aiChoice.engine)) aiChoice.engine = engines[0];

  // เหลือโหมดเดียว -> ซ่อนปุ่มไปเลย ลูกค้าจะไม่รู้ว่ามีอีกโหมด
  // (ของเดิมโชว์ปุ่มแบบกดไม่ได้ ซึ่งทำให้สงสัยว่าอีกโหมดคืออะไร)
  engineToggle.hidden = engines.length < 2;
  const on = aiChoice.engine === 'pi';
  engineToggle.setAttribute('aria-checked', String(on));
  engineToggle.disabled = false;
  engineText.textContent = ENGINE_NAME[on ? 'pi' : 'classic'];
  engineToggle.title = on
    ? `${ENGINE_NAME.pi}: ค้นข้อมูลเพิ่มเองหลายรอบก่อนตอบ ครบกว่าแต่ใช้เวลานานกว่า (กดเพื่อสลับเป็น${ENGINE_NAME.classic})`
    : `${ENGINE_NAME.classic}: ค้นรอบเดียวแล้วตอบทันที (กดเพื่อสลับเป็น${ENGINE_NAME.pi})`;
}

function renderAiPicker() {
  aiSelect.innerHTML = '';
  // แสดงเฉพาะ AI ที่ใช้ได้จริง — ตัวที่ยังไม่ตั้ง API key เคยโชว์เป็นสีเทาพร้อมข้อความ ".env"
  // ซึ่งคนซื้อเห็นแล้วนึกว่าระบบพัง (ผู้ดูแลดูรายการเต็มได้ที่ /providers)
  for (const p of providers.filter(p => p.available)) {
    aiSelect.add(new Option(p.label, p.id));
  }
  aiSelect.value = aiChoice.provider ?? '';
  syncEngineToggle();
  aiPicker.hidden = false;
}

async function initAiPicker() {
  try {
    const res = await fetch('/providers');
    const data = await res.json();
    providers = data.providers || [];
    if (Array.isArray(data.engines) && data.engines.length) serverEngines = data.engines;
    debugOn = data.debug === true;   // ผู้ดูแลเปิด DEBUG_PANEL -> ทุกคำตอบมีปุ่ม "กระบวนการคิด"
    if (!providers.length) return;

    const saved = loadAiChoice();
    const usable = id => providers.some(p => p.id === id && p.available);
    aiChoice.provider = usable(saved.provider) ? saved.provider : data.default_provider;
    aiChoice.engine = ENGINE_NAME[saved.engine] ? saved.engine : data.default_engine;
    renderAiPicker();
  } catch { /* ซ่อนเมนูไว้ เซิร์ฟเวอร์จะใช้ AI เริ่มต้นเอง */ }
}

aiSelect.addEventListener('change', () => {
  aiChoice.provider = aiSelect.value;
  syncEngineToggle();
  saveAiChoice();
});

engineToggle.addEventListener('click', () => {
  aiChoice.engine = aiChoice.engine === 'pi' ? 'classic' : 'pi';
  syncEngineToggle();
  saveAiChoice();
});

/* ───────────── ผูก event ───────────── */
// ระหว่างรอคำตอบ ปุ่มนี้คือปุ่ม "หยุด" (ดู setLoading)
sendBtn.addEventListener('click', () => (isLoading ? activeRequest?.abort() : sendMessage()));
newChatBtn.addEventListener('click', newChat);
clearAllBtn.addEventListener('click', clearAllChats);

userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

document.getElementById('chips').addEventListener('click', e => {
  const chip = e.target.closest('.chip');
  if (chip) sendMessage(chip.textContent.trim());
});

loadChats();
renderHistory();
initAiPicker();
userInput.focus();

// หน้าเพิ่งโหลด -> จอเริ่มจากบทสนทนาว่าง แต่ session ฝั่งเซิร์ฟเวอร์อาจยังจำประวัติและ "งบ"
// ของรอบก่อนรีเฟรชอยู่ ต้องล้างให้ตรงกับสิ่งที่ผู้ใช้เห็น ไม่งั้นบอทจะกรองด้วยงบเก่า
// ที่ผู้ใช้ไม่เห็นแล้ว (เช่น ถามลอย ๆ แต่บอทตอบว่า "จากงบประมาณ 20,000 บาท")
fetch('/new_chat', { method: 'POST' }).catch(() => {});
