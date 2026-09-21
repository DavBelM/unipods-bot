const chat = document.getElementById("chat");
const composer = document.getElementById("composer");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send");

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function addMessage(text, role) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function addPendingMessage() {
  const el = document.createElement("div");
  el.className = "message bot pending";
  el.innerHTML =
    'Thinking <span class="typing-dots"><span></span><span></span><span></span></span>';
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function renderAnswer(el, answer, sources) {
  el.classList.remove("pending");
  el.innerHTML = escapeHtml(answer);

  if (sources && sources.length > 0) {
    const sourcesEl = document.createElement("div");
    sourcesEl.className = "sources";
    sourcesEl.innerHTML = "<strong>Sources</strong>";
    sources.forEach((s) => {
      const item = document.createElement("div");
      item.className = "source-item";
      item.textContent = `${s.session} @ ${s.timestamp}`;
      sourcesEl.appendChild(item);
    });
    el.appendChild(sourcesEl);
  }
  chat.scrollTop = chat.scrollHeight;
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage(question, "user");
  questionInput.value = "";
  sendButton.disabled = true;

  const pending = addPendingMessage();

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    renderAnswer(pending, data.answer, data.sources);
  } catch (err) {
    pending.classList.remove("pending");
    pending.textContent = `Something went wrong: ${err.message}`;
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});
