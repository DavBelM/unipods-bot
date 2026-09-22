function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString();
}

function renderStats(rows) {
  const total = rows.length;
  const answered = rows.filter((r) => r.relevant).length;
  const rejected = total - answered;
  const sessionCounts = {};
  rows.forEach((r) => {
    (r.sources || []).forEach((s) => {
      sessionCounts[s.session] = (sessionCounts[s.session] || 0) + 1;
    });
  });
  const topSession = Object.entries(sessionCounts).sort((a, b) => b[1] - a[1])[0];

  const tiles = [
    { value: total, label: "Total questions" },
    { value: answered, label: "Answered from sessions" },
    { value: rejected, label: "Not covered" },
    { value: topSession ? topSession[0] : "—", label: "Most-cited session" },
  ];

  const stats = document.getElementById("stats");
  stats.innerHTML = "";
  tiles.forEach((t) => {
    const el = document.createElement("div");
    el.className = "stat-tile";
    el.innerHTML = `<div class="value">${escapeHtml(String(t.value))}</div><div class="label">${escapeHtml(t.label)}</div>`;
    stats.appendChild(el);
  });
}

function renderLog(rows) {
  const log = document.getElementById("log");
  log.innerHTML = "";

  if (rows.length === 0) {
    log.innerHTML = '<div class="empty-state">No questions logged yet.</div>';
    return;
  }

  rows.forEach((r) => {
    const el = document.createElement("div");
    el.className = "log-entry";

    const badge = r.relevant
      ? '<span class="badge answered">Answered</span>'
      : '<span class="badge rejected">Not covered</span>';

    const sources = (r.sources || [])
      .map((s) => `${s.session} @ ${s.timestamp}`)
      .join(" · ");

    el.innerHTML = `
      <div class="question">${escapeHtml(r.question)}</div>
      <div class="answer">${escapeHtml(r.answer)}</div>
      <div class="meta">
        ${badge}
        <span>${escapeHtml(formatTime(r.created_at))}</span>
        ${sources ? `<span>${escapeHtml(sources)}</span>` : ""}
      </div>
    `;
    log.appendChild(el);
  });
}

async function load() {
  try {
    const res = await fetch("/queries?limit=200");
    const rows = await res.json();
    renderStats(rows);
    renderLog(rows);
  } catch (err) {
    document.getElementById("log").innerHTML =
      `<div class="empty-state">Failed to load: ${escapeHtml(err.message)}</div>`;
  }
}

load();
