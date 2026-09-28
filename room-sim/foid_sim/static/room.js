const badge = document.getElementById("live-badge");
const I18N = window.FOID_I18N || { en: {}, de: {}, default: "en" };
const LOCALE_KEY = "foid.locale";

let socket = null;
let reconnectTimer = null;
let backoffMs = 1000;

function byId(devices) {
  return Object.fromEntries(devices.map((device) => [device.id, device]));
}

function currentLocale() {
  const stored = localStorage.getItem(LOCALE_KEY);
  if (stored && I18N[stored]) return stored;
  return I18N.default || "en";
}

function t(key, locale = currentLocale()) {
  const bundle = I18N[locale] || I18N.en || {};
  return bundle[key] || (I18N.en && I18N.en[key]) || key;
}

function applyChrome(locale = currentLocale()) {
  document.documentElement.lang = locale;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    if (el.id === "live-badge") return;
    el.textContent = t(el.dataset.i18n, locale);
  });
  const heading = t("room.heading", locale);
  document.title = t("room.title", locale);
  const svg = document.querySelector("svg[role='img']");
  if (svg) svg.setAttribute("aria-label", t("room.aria", locale) || heading);
  document.querySelectorAll(".lang-switch [data-locale]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.locale === locale);
  });
  if (!badge.classList.contains("is-off")) badge.textContent = t("room.live", locale);
  else badge.textContent = t("room.offline", locale);
}

function setLocale(locale) {
  localStorage.setItem(LOCALE_KEY, locale);
  applyChrome(locale);
}

function apply(payload) {
  const map = byId(payload.devices);
  document.getElementById("window").classList.toggle("is-open", Boolean(map.window?.state.open));
  document.getElementById("tv").classList.toggle("is-on", Boolean(map.tv?.state.on));
  document.getElementById("lamp").classList.toggle("is-on", Boolean(map.lamp?.state.on));
  const headrest = Number(map.bed?.state.headrest ?? 0);
  const rest = document.querySelector("#bed .headrest");
  rest.style.transform = `rotate(${headrest * 58}deg)`;
  document.getElementById("bed").classList.toggle("is-up", headrest >= 0.5);
}

async function send(device, action) {
  await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device, action }),
  });
}

document.getElementById("window").addEventListener("click", () => {
  const open = document.getElementById("window").classList.contains("is-open");
  send("window", open ? "close" : "open");
});
document.getElementById("tv").addEventListener("click", () => {
  const on = document.getElementById("tv").classList.contains("is-on");
  send("tv", on ? "off" : "on");
});
document.getElementById("lamp").addEventListener("click", () => {
  const on = document.getElementById("lamp").classList.contains("is-on");
  send("lamp", on ? "off" : "on");
});
document.getElementById("bed").addEventListener("click", () => {
  const raised = document.getElementById("bed").classList.contains("is-up");
  send("bed", raised ? "down" : "up");
});

document.querySelectorAll(".lang-switch [data-locale]").forEach((button) => {
  button.addEventListener("click", () => setLocale(button.dataset.locale));
});

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, backoffMs);
  backoffMs = Math.min(backoffMs * 2, 15000);
}

function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws`);
  socket.onopen = () => {
    backoffMs = 1000;
    badge.classList.remove("is-off");
    badge.textContent = t("room.live");
  };
  socket.onclose = () => {
    socket = null;
    badge.classList.add("is-off");
    badge.textContent = t("room.offline");
    scheduleReconnect();
  };
  socket.onmessage = (event) => apply(JSON.parse(event.data));
}

applyChrome();
fetch("/api/state")
  .then((res) => res.json())
  .then(apply)
  .finally(connect);
