const stateEl = document.getElementById("state");
const timerEl = document.getElementById("timer");
const omenEl = document.getElementById("omen-pill");
const hwEl = document.getElementById("hw");
const logEl = document.getElementById("log");
const camsEl = document.getElementById("cams");
const camMeta = document.getElementById("cam-meta");
const toastEl = document.getElementById("toast");
const jointsEl = document.getElementById("joints");
const jointsMeta = document.getElementById("joints-meta");
const jointsPanel = document.querySelector(".panel.joints");

let toastTimer = 0;
let jointRows = null;
let streamKey = "";

function toast(msg) {
  toastEl.hidden = false;
  toastEl.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.hidden = true;
  }, 2800);
}

function fmtTime(s) {
  const t = Math.max(0, Number(s) || 0);
  const m = Math.floor(t / 60);
  const r = t - m * 60;
  return `${String(m).padStart(2, "0")}:${r.toFixed(1).padStart(4, "0")}`;
}

function clsFor(state) {
  const x = (state || "IDLE").toLowerCase();
  if (x === "recording") return "recording";
  if (x === "teleop") return "teleop";
  if (x === "disconnected") return "disconnected";
  return "idle";
}

function row(k, v, ok) {
  const li = document.createElement("li");
  const vk = ok === true ? "ok" : ok === false ? "bad" : "";
  li.innerHTML = `<span class="k">${k}</span><span class="v ${vk}">${v}</span>`;
  return li;
}

function normPct(v) {
  const x = Number(v) || 0;
  const pct = ((x + 100) / 200) * 100;
  return Math.max(0, Math.min(100, pct));
}

function ensureJointRows(names) {
  if (jointRows && jointRows.length === names.length) return;
  jointsEl.replaceChildren();
  jointRows = names.map((name) => {
    const rowEl = document.createElement("div");
    rowEl.className = "joint-row";
    rowEl.innerHTML = `
      <span class="name">${name}</span>
      <div class="joint-bars">
        <div class="bar-track"><div class="bar-fill state"></div></div>
        <div class="bar-track"><div class="bar-fill action"></div></div>
      </div>
      <span class="vals">—</span>`;
    jointsEl.appendChild(rowEl);
    return {
      stateFill: rowEl.querySelector(".bar-fill.state"),
      actionFill: rowEl.querySelector(".bar-fill.action"),
      vals: rowEl.querySelector(".vals"),
    };
  });
}

function renderStreams(streams) {
  const key = (streams || []).map((s) => s.id).join("|");
  if (key === streamKey) {
    (streams || []).forEach((s) => {
      const slot = camsEl.querySelector(`[data-stream="${CSS.escape(s.id)}"]`);
      if (!slot) return;
      slot.classList.toggle("live", !!s.live);
      slot.classList.toggle("waiting", !s.live);
      const badge = slot.querySelector(".live-badge");
      if (badge) badge.textContent = s.live ? "live" : "offline";
    });
    return;
  }
  streamKey = key;
  camsEl.replaceChildren();
  (streams || []).forEach((s) => {
    const slot = document.createElement("div");
    slot.className = `cam-slot ${s.kind || "rgb"} ${s.live ? "live" : "waiting"}`;
    slot.dataset.stream = s.id;
    slot.innerHTML = `
      <div class="panel-h">
        <h1>${s.label || s.id}</h1>
        <span class="live-badge muted">${s.live ? "live" : "offline"}</span>
      </div>
      <div class="stage">
        <img src="${s.url}?t=${Date.now()}" alt="${s.label || s.id}"/>
      </div>`;
    camsEl.appendChild(slot);
  });
  if (!(streams || []).length) {
    const empty = document.createElement("div");
    empty.className = "cam-slot waiting";
    empty.innerHTML = `
      <div class="panel-h"><h1>no streams</h1></div>
      <div class="stage empty-hint">Set CAMERA_NAMES in camera.env</div>`;
    camsEl.appendChild(empty);
  }
}

async function refreshJoints() {
  const r = await fetch("/api/joints");
  const j = await r.json();
  const names = j.names || [];
  if (!names.length) {
    jointsMeta.textContent = "waiting for record/eval…";
    jointsPanel.classList.add("stale");
    return;
  }
  ensureJointRows(names);
  const state = j.state || [];
  const action = j.action || [];
  jointRows.forEach((rowEl, i) => {
    const s = state[i];
    const a = action[i];
    rowEl.stateFill.style.width = `${normPct(s)}%`;
    rowEl.actionFill.style.width = `${normPct(a)}%`;
    rowEl.vals.textContent = `${Number(s).toFixed(1)} / ${Number(a).toFixed(1)}`;
  });
  if (j.fresh) {
    jointsMeta.textContent = "live";
    jointsPanel.classList.remove("stale");
  } else {
    jointsMeta.textContent = j.age_s != null ? `stale ${j.age_s}s` : "stale";
    jointsPanel.classList.add("stale");
  }
}

async function refresh() {
  const r = await fetch("/api/status");
  const s = await r.json();
  const st = s.state || "IDLE";
  stateEl.textContent = st;
  stateEl.className = `pill ${clsFor(st)}`;
  timerEl.textContent = fmtTime(s.elapsed_s);
  const cam = s.camera || {};
  const streams = s.streams || [];
  renderStreams(streams);

  if (camMeta) {
    const nLive = (cam.live_ids || []).length;
    camMeta.textContent = cam.backend === "zmq"
      ? `${cam.host}:${cam.port} · ${nLive}/${streams.length || "?"} live · ${cam.fps ?? "—"} fps`
      : `opencv · ${cam.fps ?? "—"} fps`;
  }
  omenEl.textContent = cam.reachable ? "OMEN UP" : "OMEN DOWN";
  omenEl.className = `pill ${cam.reachable ? "ok" : "bad"}`;

  const arms = s.arms || {};
  const age = cam.last_frame_age_s == null ? "—" : `${cam.last_frame_age_s}s`;
  const depthNames = (cam.depth_names || []).join(", ") || "—";
  const depthLive = (cam.depth_live || []).map((id) => id.replace(/_depth$/, "")).join(", ");
  const rows = [
    row("Omen", cam.reachable ? `reachable :${cam.port || 5555}` : "not reachable", cam.reachable),
    row("Streams", `${(cam.live_ids || []).length}/${streams.length || 0} live`, (cam.live_ids || []).length > 0),
    row("Frame age", age, cam.last_frame_age_s != null && cam.last_frame_age_s < 2),
    row("FPS", cam.fps ?? "—", cam.fps > 1),
    row("Depth", cam.depth_ok ? (depthLive || depthNames) : "no depth yet", cam.depth_ok),
    row(
      "Follower",
      arms.follower_ok ? arms.follower_port : (arms.follower_port || "missing"),
      !!arms.follower_ok,
    ),
    row(
      "Leader",
      arms.leader_ok ? arms.leader_port : (arms.leader_port || "missing"),
      !!arms.leader_ok,
    ),
    row("Last session", s.last_banner || "—"),
  ];
  hwEl.replaceChildren(...rows);
}

function bindLogs() {
  const src = new EventSource("/api/log/stream");
  src.onmessage = (ev) => {
    let payload;
    try {
      payload = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (payload.reset) {
      logEl.textContent = payload.chunk || "";
    } else {
      logEl.textContent += payload.chunk || "";
      if (logEl.textContent.length > 120000) {
        logEl.textContent = logEl.textContent.slice(-80000);
      }
    }
    logEl.scrollTop = logEl.scrollHeight;
  };
  src.onerror = () => {
    /* EventSource retries */
  };
}

document.addEventListener("keydown", (e) => {
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  if (k === "d") {
    e.preventDefault();
    Promise.all([refresh(), refreshJoints()])
      .then(() => toast("Status refreshed (D)"))
      .catch(() => toast("Status check failed"));
  } else if (k === "a") {
    e.preventDefault();
    toast("Start recording in a terminal — bash scripts/08_record.sh");
  }
});

bindLogs();
refresh();
refreshJoints().catch(() => {});
setInterval(() => {
  refresh().catch(() => {});
}, 1000);
setInterval(() => {
  refreshJoints().catch(() => {});
}, 100);
