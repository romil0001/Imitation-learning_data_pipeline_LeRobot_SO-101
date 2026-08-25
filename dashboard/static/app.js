const stateEl = document.getElementById("state");
const timerEl = document.getElementById("timer");
const omenEl = document.getElementById("omen-pill");
const hwEl = document.getElementById("hw");
const logEl = document.getElementById("log");
const camName = document.getElementById("cam-name-overhead");
const camMeta = document.getElementById("cam-meta");
const toastEl = document.getElementById("toast");
const jointsEl = document.getElementById("joints");
const jointsMeta = document.getElementById("joints-meta");
const jointsPanel = document.querySelector(".panel.joints");

let toastTimer = 0;
let jointRows = null;

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
  // Feetech body joints ~[-100,100] or degrees; gripper ~[0,100]. Map to bar width.
  const x = Number(v) || 0;
  const pct = ((x + 100) / 200) * 100;
  return Math.max(0, Math.min(100, pct));
}

function ensureJointRows(names) {
  if (jointRows && jointRows.length === names.length) return;
  jointsEl.replaceChildren();
  jointRows = names.map((name) => {
    const row = document.createElement("div");
    row.className = "joint-row";
    row.innerHTML = `
      <span class="name">${name}</span>
      <div class="joint-bars">
        <div class="bar-track"><div class="bar-fill state"></div></div>
        <div class="bar-track"><div class="bar-fill action"></div></div>
      </div>
      <span class="vals">—</span>`;
    jointsEl.appendChild(row);
    return {
      stateFill: row.querySelector(".bar-fill.state"),
      actionFill: row.querySelector(".bar-fill.action"),
      vals: row.querySelector(".vals"),
    };
  });
}

async function refreshJoints() {
  const r = await fetch("/api/joints");
  const j = await r.json();
  const names = j.names || [];
  if (!names.length) {
    jointsMeta.textContent = "waiting for record…";
    jointsPanel.classList.add("stale");
    return;
  }
  ensureJointRows(names);
  const state = j.state || [];
  const action = j.action || [];
  jointRows.forEach((row, i) => {
    const s = state[i];
    const a = action[i];
    row.stateFill.style.width = `${normPct(s)}%`;
    row.actionFill.style.width = `${normPct(a)}%`;
    row.vals.textContent = `${Number(s).toFixed(1)} / ${Number(a).toFixed(1)}`;
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
  const names = cam.names || [cam.name || "wrist"];
  if (camName) camName.textContent = names.join(" + ");
  if (camMeta) {
    camMeta.textContent = cam.backend === "zmq"
      ? `${cam.host}:${cam.port}`
      : `opencv`;
  }
  omenEl.textContent = cam.reachable ? "OMEN UP" : "OMEN DOWN";
  omenEl.className = `pill ${cam.reachable ? "ok" : "bad"}`;

  const arms = s.arms || {};
  const age = cam.last_frame_age_s == null ? "—" : `${cam.last_frame_age_s}s`;
  hwEl.replaceChildren(
    row("Omen", cam.reachable ? "reachable :5555" : "not reachable", cam.reachable),
    row("Frame age", age, cam.last_frame_age_s != null && cam.last_frame_age_s < 2),
    row("FPS", cam.fps ?? "—", cam.fps > 1),
    row("Wrist depth", cam.depth_ok ? "streaming" : "no depth yet", cam.depth_ok),
    row("Follower", arms.follower_ok ? arms.follower_port : (arms.follower_port || "missing"), arms.follower_ok),
    row("Leader", arms.leader_ok ? arms.leader_port : (arms.leader_port || "missing"), arms.leader_ok),
    row("Last session", s.last_banner || "—"),
  );
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
