let recapActive = false;
let recapSocket = null;
let mediaRecorder = null;
let useVoice = true; // voice mode by default

const blob = document.getElementById("blob");
const micBtn = document.getElementById("mic-btn");
const textInput = document.getElementById("text-input");
const endBtn = document.getElementById("end-button");
const toggleBtn = document.getElementById("toggle-button");
const chatContainer = document.getElementById("chat-container");
const splashContainer = document.getElementById("splash-container");
const splashText = document.getElementById("splash-text");
const appHeader = document.getElementById("app-header");

const startupSound = document.getElementById("startup-sound");
const powerOffSound = document.getElementById("poweroff-sound");
const pressonSound = document.getElementById("presson-sound");
const pressoffSound = document.getElementById("pressoff-sound");

// Append a chat bubble
function appendBubble(text, sender) {
  const msg = document.createElement("div");
  msg.className = `message ${sender}`;
  msg.textContent = text;
  chatContainer.appendChild(msg);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  if (sender === 'user') {
    msg.classList.add('arrow-up');
    setTimeout(() => msg.classList.remove('arrow-up'), 600);
  }
}

// Restore UI from saved state
function restoreUI() {
  appHeader.style.display = 'flex';
  splashContainer.style.display = 'none';
  chatContainer.style.display = 'flex';
  blob.style.display = 'block';
  if (useVoice) {
    micBtn.style.display = 'flex';
    textInput.style.display = 'none';
  } else {
    micBtn.style.display = 'none';
    textInput.style.display = 'block';
  }
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Power off
function powerOff() {
  if (!recapActive) return;
  powerOffSound.currentTime = 0;
  powerOffSound.play().catch(console.error);
  if (mediaRecorder && mediaRecorder.state === "recording") mediaRecorder.stop();
  if (recapSocket) { recapSocket.close(); recapSocket = null; }
  recapActive = false;
  // Clear UI and state
  chatContainer.innerHTML = '';
  localStorage.clear();
  initUI();
}
endBtn.onclick = powerOff;

// Toggle input mode
toggleBtn.onclick = () => {
  useVoice = !useVoice;
  localStorage.setItem('useVoice', useVoice);
  if (useVoice) {
    micBtn.style.display = 'flex';
    textInput.style.display = 'none';
  } else {
    micBtn.style.display = 'none';
    textInput.style.display = 'block';
    textInput.focus();
  }
};

// Initialize UI state
function initUI() {
  appHeader.style.display = 'flex';
  splashContainer.style.display = 'flex';
  blob.style.display = 'none';
  chatContainer.style.display = 'none';
  micBtn.style.display = 'none';
  textInput.style.display = 'none';
}

// Start RECAP session
function startRecap() {
  if (recapActive) return;
  recapActive = true;
  localStorage.setItem('recapActive', true);
  // Show UI
  restoreUI();

  startupSound.currentTime = 0;
  startupSound.play().catch(console.error);
  appendBubble("Hello! I'm RECAP, your AI assistant. How can I help you today?", 'assistant');

  const wsUrl = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/chat`;
  recapSocket = new WebSocket(wsUrl);
  recapSocket.onmessage = evt => {
    appendBubble(evt.data, 'assistant');
    localStorage.setItem('chatState', chatContainer.innerHTML);
  };
  recapSocket.onclose = () => console.log("RECAP socket closed");
  recapSocket.onerror = err => console.error("RECAP socket error", err);

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = ev => { if (ev.data.size) recapSocket.send(ev.data); };
    })
    .catch(err => console.error("Mic access error:", err));
}

// Mic button handler
micBtn.onclick = () => {
  if (!recapActive) { startRecap(); return; }
  pressonSound.currentTime = 0;
  pressonSound.play().catch(console.error);
  appendBubble("...", 'user');
  mediaRecorder.start();
  setTimeout(() => {
    mediaRecorder.stop();
    pressoffSound.currentTime = 0;
    pressoffSound.play().catch(console.error);
    localStorage.setItem('chatState', chatContainer.innerHTML);
  }, 4000);
};

// Text input handler
textInput.addEventListener('keypress', e => {
  if (e.key === 'Enter' && e.target.value.trim()) {
    const msg = e.target.value.trim();
    appendBubble(msg, 'user');
    // send JSON-encoded text frame to server
    recapSocket.send(JSON.stringify({
      type: 'text',
      content: msg
    }));
    localStorage.setItem('chatState', chatContainer.innerHTML);
    e.target.value = '';
  }
});

// Blob click: start or toggle size
blob.addEventListener('click', () => {
  if (!recapActive) startRecap();
  else blob.classList.toggle('blob-pressed');
});

// Save state before unload
window.addEventListener('beforeunload', () => {
  localStorage.setItem('recapActive', recapActive);
  localStorage.setItem('useVoice', useVoice);
  localStorage.setItem('chatState', chatContainer.innerHTML);
});

// Hook splash click and init on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initUI();
  splashText.addEventListener('click', startRecap);
  // Restore if active
  if (localStorage.getItem('recapActive') === 'true') {
    recapActive = true;
    useVoice = localStorage.getItem('useVoice') === 'true';
    chatContainer.innerHTML = localStorage.getItem('chatState') || '';
    restoreUI();
  }
});