const stateEl = document.getElementById("state");
const timerEl = document.getElementById("timer");
const omenEl = document.getElementById("omen-pill");
const hwEl = document.getElementById("hw");
const logEl = document.getElementById("log");
const camName = document.getElementById("cam-name");
const camMeta = document.getElementById("cam-meta");
const toastEl = document.getElementById("toast");

let toastTimer = 0;

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

async function refresh() {
  const r = await fetch("/api/status");
  const s = await r.json();
  const st = s.state || "IDLE";
  stateEl.textContent = st;
  stateEl.className = `pill ${clsFor(st)}`;
  timerEl.textContent = fmtTime(s.elapsed_s);
  const cam = s.camera || {};
  camName.textContent = cam.name || "front";
  camMeta.textContent = cam.backend === "zmq"
    ? `${cam.host}:${cam.port}`
    : `opencv`;
  omenEl.textContent = cam.reachable ? "OMEN UP" : "OMEN DOWN";
  omenEl.className = `pill ${cam.reachable ? "ok" : "bad"}`;

  const arms = s.arms || {};
  const age = cam.last_frame_age_s == null ? "—" : `${cam.last_frame_age_s}s`;
  hwEl.replaceChildren(
    row("Omen", cam.reachable ? "reachable :5555" : "not reachable", cam.reachable),
    row("Frame age", age, cam.last_frame_age_s != null && cam.last_frame_age_s < 2),
    row("FPS", cam.fps ?? "—", cam.fps > 1),
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
    refresh().then(() => toast("Status refreshed (D)")).catch(() => toast("Status check failed"));
  } else if (k === "a") {
    e.preventDefault();
    toast("v1: start recording in a terminal — bash scripts/08_record.sh");
  } else if (k === "b") {
    e.preventDefault();
    toast("v1: mark failure in the LeRobot record window");
  } else if (k === "c") {
    e.preventDefault();
    toast("v1: mark success in the LeRobot record window");
  }
});

bindLogs();
refresh();
setInterval(() => {
  refresh().catch(() => {});
}, 1000);
