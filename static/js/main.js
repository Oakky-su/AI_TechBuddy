/* Tech Buddy — ฝั่งหน้าเว็บ
   หน้าที่: ส่งข้อความไป /chat, เรนเดอร์คำตอบ (markdown), จัดการธีม และเก็บประวัติแชทใน localStorage */

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

const STORE_KEY = 'tb-chats';
let chats = [];        // [{ id, title, messages: [{role, content}] }]
let currentId = null;
let isLoading = false;

/* ───────────── ธีมสว่าง / มืด ───────────── */
themeToggle.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('tb-theme', next);
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

function appendMsg(role, content, isError = false) {
  hideEmptyState();

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

  wrap.append(av, bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot-msg typing';
  wrap.id = 'typing-indicator';
  wrap.innerHTML =
    `<div class="avatar bot-avatar">${BOT_ICON}</div>
     <div class="bubble"><span class="dot-anim"></span><span class="dot-anim"></span><span class="dot-anim"></span></div>`;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideTyping() {
  document.getElementById('typing-indicator')?.remove();
}

function hideEmptyState() { emptyState.style.display = 'none'; }

function showEmptyState() {
  messagesEl.querySelectorAll('.msg').forEach(el => el.remove());
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
async function openChat(id) {
  const chat = chats.find(c => c.id === id);
  if (!chat || isLoading) return;

  currentId = id;
  showEmptyState();   // ล้างข้อความเก่าออกจากจอ (appendMsg จะซ่อนการ์ดเริ่มต้นให้เอง)
  chat.messages.forEach(m => appendMsg(m.role, m.content));
  renderHistory();

  try {
    await fetch('/restore', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history: chat.messages }),
    });
  } catch { /* คุยต่อได้อยู่ แค่ฝั่งเซิร์ฟเวอร์อาจจำบริบทเดิมไม่ได้ */ }
}

function recordMessage(role, content) {
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
    chat.messages.push({ role, content });
    chat.messages = chat.messages.slice(-20);
  }
  saveChats();
  renderHistory();
}

/* ───────────── ส่งข้อความ ───────────── */
async function sendMessage(text) {
  const msg = (text ?? userInput.value).trim();
  if (!msg || isLoading) return;

  userInput.value = '';
  isLoading = true;
  sendBtn.disabled = true;

  appendMsg('user', msg);
  recordMessage('user', msg);
  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    });
    const data = await res.json().catch(() => ({ error: `เซิร์ฟเวอร์ตอบกลับผิดรูปแบบ (HTTP ${res.status})` }));
    hideTyping();

    if (data.error) {
      appendMsg('bot', `เกิดข้อผิดพลาด: ${data.error}`, true);
    } else {
      appendMsg('bot', data.message || '...');
      recordMessage('assistant', data.message || '');
    }
  } catch (err) {
    hideTyping();
    appendMsg('bot', `เชื่อมต่อเซิร์ฟเวอร์ไม่ได้: ${err.message}`, true);
  }

  isLoading = false;
  sendBtn.disabled = false;
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

/* ───────────── ผูก event ───────────── */
sendBtn.addEventListener('click', () => sendMessage());
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
userInput.focus();

// หน้าเพิ่งโหลด -> จอเริ่มจากบทสนทนาว่าง แต่ session ฝั่งเซิร์ฟเวอร์อาจยังจำประวัติและ "งบ"
// ของรอบก่อนรีเฟรชอยู่ ต้องล้างให้ตรงกับสิ่งที่ผู้ใช้เห็น ไม่งั้นบอทจะกรองด้วยงบเก่า
// ที่ผู้ใช้ไม่เห็นแล้ว (เช่น ถามลอย ๆ แต่บอทตอบว่า "จากงบประมาณ 20,000 บาท")
fetch('/new_chat', { method: 'POST' }).catch(() => {});
