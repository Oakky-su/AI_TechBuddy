const messages = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const historyList = document.getElementById('historyList');

let chatHistory = [];
let isLoading = false;

function appendMsg(role, content) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role === 'user' ? 'user-msg' : 'bot-msg'}`;

  const av = document.createElement('div');
  av.className = `avatar ${role === 'user' ? 'user-avatar' : 'bot-avatar'}`;
  av.textContent = role === 'user' ? 'U' : '🤖';

  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.innerHTML = content.replace(/\n/g, '<br>');

  wrap.appendChild(av);
  wrap.appendChild(bub);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return wrap;
}

function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot-msg typing';
  wrap.id = 'typing-indicator';

  const av = document.createElement('div');
  av.className = 'avatar bot-avatar';
  av.textContent = '🤖';

  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.innerHTML = '<div class="dot-anim"></div><div class="dot-anim"></div><div class="dot-anim"></div>';

  wrap.appendChild(av);
  wrap.appendChild(bub);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
}

function hideTyping() {
  const t = document.getElementById('typing-indicator');
  if (t) t.remove();
}

function addHistoryItem(text) {
  const item = document.createElement('div');
  item.className = 'history-item';
  item.textContent = text.length > 28 ? text.slice(0, 28) + '…' : text;
  item.title = text;
  historyList.prepend(item);
}

async function sendMessage() {
  const msg = userInput.value.trim();
  if (!msg || isLoading) return;

  userInput.value = '';
  isLoading = true;
  document.querySelector('.send-btn').disabled = true;

  appendMsg('user', msg);
  if (chatHistory.length === 0) addHistoryItem(msg);
  chatHistory.push({ role: 'user', content: msg });

  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    });

    const data = await res.json();
    hideTyping();

    if (data.error) {
      appendMsg('bot', `❌ เกิดข้อผิดพลาด: ${data.error}`);
    } else {
      appendMsg('bot', data.message || '...');
      chatHistory.push({ role: 'assistant', content: data.message });
    }
  } catch (err) {
    hideTyping();
    appendMsg('bot', `❌ ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้: ${err.message}`);
  }

  isLoading = false;
  document.querySelector('.send-btn').disabled = false;
  userInput.focus();
}

async function newChat() {
  await fetch('/new_chat', { method: 'POST' });
  messages.innerHTML = `
    <div class="msg bot-msg">
      <div class="avatar bot-avatar">🤖</div>
      <div class="bubble">
        สวัสดีครับ! ผม Tech Buddy ผู้ช่วยเลือกโน้ตบุ๊กส่วนตัวของคุณ 😊<br>
        บอกความต้องการมาได้เลยครับ เช่น <em>"โน้ตบุ๊กเล่นเกม งบ 35,000"</em>
      </div>
    </div>`;
  chatHistory = [];
  userInput.focus();
}

userInput.focus();
