/**
 * LeRobot SO-101 Operator Console - Client Script (RUI / ReachUI HUD Theme)
 * Manages multi-camera feeds (2x D405 + 1x RGB), joint telemetry, hotkeys, and live logs.
 */

// DOM Elements
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

let toastTimer = null;
let jointRows = null;
let streamKey = "";

// ---------------------- UTILITY FUNCTIONS ---------------------- //

/** Display an ImGui-styled HUD toast message */
function toast(msg, durationMs = 2800) {
  if (!toastEl) return;
  toastEl.textContent = msg;
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.hidden = true;
  }, durationMs);
}

/** Format seconds into MM:SS.S */
function fmtTime(s) {
  const t = Math.max(0, Number(s) || 0);
  const m = Math.floor(t / 60);
  const r = t - m * 60;
  return `${String(m).padStart(2, "0")}:${r.toFixed(1).padStart(4, "0")}`;
}

/** Map session state string to CSS class */
function clsFor(state) {
  const x = (state || "IDLE").toLowerCase();
  if (x === "recording") return "recording";
  if (x === "teleop") return "teleop";
  if (x === "disconnected") return "bad";
  return "idle";
}

/** Generate a key-value row for the hardware telemetry list */
function row(k, v, ok) {
  const li = document.createElement("li");
  const vk = ok === true ? "ok" : ok === false ? "bad" : "";
  li.innerHTML = `<span class="k">${k}</span><span class="v ${vk}">${v}</span>`;
  return li;
}

/** Normalize joint angles / servo positions (-180..180 or -100..100) to 0..100% */
function normPct(v) {
  const x = Number(v) || 0;
  // Standard normalize for +/- 180 deg joint span
  const pct = ((x + 180) / 360) * 100;
  return Math.max(0, Math.min(100, pct));
}

// ---------------------- JOINT TELEMETRY ---------------------- //

/** Ensure UI rows exist for all 6 DOF / gripper joints */
function ensureJointRows(names) {
  if (jointRows && jointRows.length === names.length) return;

  jointsEl.replaceChildren();
  jointRows = names.map((name) => {
    const rowEl = document.createElement("div");
    rowEl.className = "joint-row";
    rowEl.innerHTML = `
      <span class="name" title="${name}">${name}</span>
      <div class="joint-bars">
        <div class="bar-track" title="Measured Physical Joint State">
          <div class="bar-fill state"></div>
        </div>
        <div class="bar-track" title="Commanded Target Action">
          <div class="bar-fill action"></div>
        </div>
      </div>
      <span class="vals">—</span>`;
    jointsEl.appendChild(rowEl);

    return {
      name,
      stateFill: rowEl.querySelector(".bar-fill.state"),
      actionFill: rowEl.querySelector(".bar-fill.action"),
      vals: rowEl.querySelector(".vals"),
    };
  });
}

/** Fetch and render real-time joint positions and commanded actions */
async function refreshJoints() {
  try {
    const r = await fetch("/api/joints");
    if (!r.ok) return;
    const j = await r.json();
    const names = j.names || [];

    if (!names.length) {
      if (jointsMeta) jointsMeta.textContent = "STANDBY";
      if (jointsPanel) jointsPanel.classList.add("stale");
      return;
    }

    ensureJointRows(names);
    const state = j.state || [];
    const action = j.action || [];

    jointRows.forEach((item, i) => {
      const s = state[i] != null ? state[i] : 0;
      const a = action[i] != null ? action[i] : s;

      item.stateFill.style.width = `${normPct(s)}%`;
      item.actionFill.style.width = `${normPct(a)}%`;
      item.vals.textContent = `${Number(s).toFixed(1)}° / ${Number(a).toFixed(1)}°`;
    });

    if (jointsMeta) {
      if (j.fresh) {
        jointsMeta.textContent = "LIVE (30Hz)";
        jointsMeta.className = "muted ok";
        if (jointsPanel) jointsPanel.classList.remove("stale");
      } else {
        jointsMeta.textContent = j.age_s != null ? `STALE (${j.age_s}s)` : "STALE";
        jointsMeta.className = "muted bad";
        if (jointsPanel) jointsPanel.classList.add("stale");
      }
    }
  } catch {
    if (jointsMeta) jointsMeta.textContent = "OFFLINE";
  }
}

// ---------------------- CAMERA STREAMS ---------------------- //

/** Dynamically render MJPEG feeds for D405 Left, D405 Right, and RGB Overview */
function renderStreams(streams) {
  const key = (streams || []).map((s) => `${s.id}:${s.live}`).join("|");
  if (key === streamKey) return;
  streamKey = key;

  camsEl.replaceChildren();

  if (!streams || streams.length === 0) {
    const empty = document.createElement("div");
    empty.className = "cam-slot waiting";
    empty.innerHTML = `
      <div class="panel-h"><h1>No Cameras Active</h1></div>
      <div class="stage empty-hint">Check /dev/video* devices or config/camera.env</div>`;
    camsEl.appendChild(empty);
    return;
  }

  streams.forEach((s) => {
    const slot = document.createElement("div");
    slot.className = `cam-slot ${s.kind || "rgb"} ${s.live ? "live" : "waiting"}`;
    slot.dataset.stream = s.id;

    // RUI HUD overlay tags
    const typeTag = s.id.includes("d405")
      ? "D405"
      : s.id.includes("depth")
      ? "Z16 DEPTH"
      : "RGB OVERVIEW";

    slot.innerHTML = `
      <div class="live-badge">
        <span>${s.live ? "LIVE" : "OFFLINE"}</span>
        <span style="opacity:0.6">| ${typeTag}</span>
      </div>
      <div class="stage">
        <img 
          src="${s.url}?t=${Date.now()}" 
          alt="${s.label || s.id}" 
          loading="eager"
          onerror="this.style.opacity='0.3'"
          onload="this.style.opacity='1'"
        />
      </div>
      <div class="panel-h" style="border-top:1px solid var(--border-line); border-bottom:none;">
        <h1>${s.label || s.id}</h1>
        <span class="muted" style="font-family:var(--font-mono); font-size:10px;">${s.id}</span>
      </div>`;

    camsEl.appendChild(slot);
  });
}

// ---------------------- STATUS & HARDWARE REFRESH ---------------------- //

async function refresh() {
  try {
    const r = await fetch("/api/status");
    if (!r.ok) throw new Error("API Offline");
    const s = await r.json();

    // 1. Session State & Timer
    const st = s.state || "IDLE";
    stateEl.textContent = st;
    stateEl.className = `pill ${clsFor(st)}`;
    timerEl.textContent = fmtTime(s.elapsed_s);

    // 2. Camera Feeds
    const cam = s.camera || {};
    const streams = s.streams || [];
    renderStreams(streams);

    if (camMeta) {
      const nLive = (cam.live_ids || []).length;
      camMeta.textContent = `${cam.backend.toUpperCase()} · ${nLive}/${streams.length || 0} LIVE · ${cam.fps ?? 0} FPS`;
    }

    // 3. Controller / Bridge Pill
    if (omenEl) {
      omenEl.textContent = cam.reachable ? "FEED ONLINE" : "FEED DISCONNECTED";
      omenEl.className = `pill ${cam.reachable ? "ok" : "bad"}`;
    }

    // 4. Detailed Telemetry Rows
    const arms = s.arms || {};
    const ageStr = cam.last_frame_age_s != null ? `${cam.last_frame_age_s}s` : "—";
    const depthActive = (cam.depth_live || []).length > 0;

    const rows = [
      row("Camera Engine", `${cam.backend.toUpperCase()} (Port ${cam.port || 8080})`, cam.reachable),
      row("Active Feeds", `${(cam.live_ids || []).length} / ${streams.length || 0} online`, (cam.live_ids || []).length > 0),
      row("Frame Latency", ageStr, cam.last_frame_age_s != null && cam.last_frame_age_s < 2.0),
      row("Capture FPS", `${cam.fps ?? "—"} FPS`, (cam.fps || 0) > 5),
      row("D405 Depth", depthActive ? "ACTIVE (Stereo Z16)" : "STANDBY / RAW RGB", depthActive),
      row(
        "SO-101 Follower",
        arms.follower_ok ? arms.follower_port : (arms.follower_port || "NOT FOUND"),
        !!arms.follower_ok
      ),
      row(
        "Leader Arm",
        arms.leader_ok ? arms.leader_port : (arms.leader_port || "STANDBY"),
        !!arms.leader_ok
      ),
      row("Active Routine", s.last_banner || "Idle Standby"),
    ];

    hwEl.replaceChildren(...rows);
  } catch {
    if (omenEl) {
      omenEl.textContent = "SERVER DOWN";
      omenEl.className = "pill bad";
    }
  }
}

// ---------------------- LIVE LOG STREAM (SSE) ---------------------- //

function bindLogs() {
  if (!logEl) return;
  const src = new EventSource("/api/log/stream");

  src.onmessage = (ev) => {
    try {
      const payload = JSON.parse(ev.data);
      if (payload.reset) {
        logEl.textContent = payload.chunk || "";
      } else {
        logEl.textContent += payload.chunk || "";
        // Limit log buffer to prevent browser slowdown
        if (logEl.textContent.length > 150000) {
          logEl.textContent = logEl.textContent.slice(-100000);
        }
      }
      logEl.scrollTop = logEl.scrollHeight;
    } catch {
      // Ignore parse glitches
    }
  };

  src.onerror = () => {
    // EventSource automatically retries connection
  };
}

// ---------------------- HOTKEYS & CONTROLS ---------------------- //

const OP_KEYS = new Set(["n", "r", "q", "y", "p", "c", "s", "f", " "]);

async function sendOpKey(raw) {
  const key = raw === " " || raw === "Space" || raw === "space" ? " " : String(raw).toLowerCase();
  if (!OP_KEYS.has(key)) return;

  try {
    const r = await fetch("/api/key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    const j = await r.json().catch(() => ({}));

    if (!r.ok) {
      toast(j.detail || `Key [${key.toUpperCase()}] rejected`);
      return;
    }

    const label = key === " " ? "SPACE (Record Toggle)" : key.toUpperCase();
    if (j.listening) {
      toast(`Sent command: ${label}`);
    } else {
      toast(`Queued command: ${label} (Waiting for active process)`);
    }
  } catch {
    toast("Network error sending key command");
  }
}

function handleDashboardKey(raw) {
  const k = raw === " " ? " " : String(raw).toLowerCase();

  // Highlight matching keyboard button visually
  const kbdEl = document.querySelector(`kbd[data-key="${k === ' ' ? 'space' : k}"]`);
  if (kbdEl) {
    kbdEl.style.transform = "scale(0.94)";
    setTimeout(() => { kbdEl.style.transform = ""; }, 120);
  }

  if (k === "d") {
    Promise.all([refresh(), refreshJoints()])
      .then(() => toast("Telemetry Refreshed [D]"))
      .catch(() => toast("Refresh Failed"));
    return;
  }

  if (k === "a") {
    toast("Run record script: bash scripts/08_record.sh");
    return;
  }

  if (OP_KEYS.has(k)) {
    sendOpKey(k);
  }
}

// Physical Keyboard Listener
document.addEventListener("keydown", (e) => {
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;

  const raw = e.key === " " ? " " : e.key;
  const k = raw === " " ? " " : raw.toLowerCase();

  if (k === "d" || k === "a" || OP_KEYS.has(k)) {
    e.preventDefault();
    handleDashboardKey(raw);
  }
});

// Click / Touch Keypad Listeners
document.querySelectorAll("kbd[data-key]").forEach((el) => {
  const triggerKey = () => {
    let keyVal = el.getAttribute("data-key") || "";
    if (keyVal === "space") keyVal = " ";
    handleDashboardKey(keyVal);
  };

  el.addEventListener("click", triggerKey);
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      triggerKey();
    }
  });
});

// ---------------------- INITIALIZATION ---------------------- //

bindLogs();
refresh();
refreshJoints().catch(() => {});

// Poll hardware status every 1 second
setInterval(() => {
  refresh().catch(() => {});
}, 1000);

// Poll joint telemetry at 10 Hz (100ms) for smooth bar animations
setInterval(() => {
  refreshJoints().catch(() => {});
}, 100);
