import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = new URL("../", import.meta.url);
const port = 8766;
const debugPort = 9333;
const profile = await mkdtemp(join(tmpdir(), "casocad-chrome-"));
const chrome = process.env.CHROME_BIN || "google-chrome";
const server = spawn(
  "python3",
  ["-m", "http.server", String(port), "--bind", "127.0.0.1"],
  { cwd: root, stdio: "ignore" },
);
const browser = spawn(
  chrome,
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    `--remote-debugging-port=${debugPort}`,
    "--remote-allow-origins=*",
    `--user-data-dir=${profile}`,
    `http://127.0.0.1:${port}/tests/mesh_worker_smoke.html`,
  ],
  { stdio: "ignore" },
);

const sleep = millis => new Promise(resolve => setTimeout(resolve, millis));
const deadline = Date.now() + 130_000;
const startupDeadline = Date.now() + 15_000;
let socket;

async function cleanup() {
  socket?.close();
  const stop = child =>
    new Promise(resolve => {
      if (child.exitCode !== null) {
        resolve();
        return;
      }
      child.once("exit", resolve);
      child.kill("SIGTERM");
      setTimeout(resolve, 2_000);
    });
  await Promise.all([stop(browser), stop(server)]);
  await rm(profile, {
    recursive: true,
    force: true,
    maxRetries: 5,
    retryDelay: 100,
  }).catch(() => {});
}

try {
  let target;
  while (Date.now() < startupDeadline) {
    if (server.exitCode !== null || browser.exitCode !== null) {
      throw new Error(
        `smoke-test process exited during startup (server=${server.exitCode}, chrome=${browser.exitCode})`,
      );
    }
    try {
      const targets = await fetch(`http://127.0.0.1:${debugPort}/json/list`).then(
        response => response.json(),
      );
      target = targets.find(
        item => item.type === "page" && item.url.includes("mesh_worker_smoke"),
      );
      if (target) break;
    } catch {
      // Chrome or the local server is still starting.
    }
    await sleep(100);
  }
  if (!target) throw new Error("Chrome DevTools endpoint did not start");

  const WebSocketImpl =
    globalThis.WebSocket ?? createRequire(import.meta.url)("undici").WebSocket;
  socket = new WebSocketImpl(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });

  let nextId = 1;
  const replies = new Map();
  socket.addEventListener("message", event => {
    const message = JSON.parse(String(event.data));
    replies.get(message.id)?.(message);
    replies.delete(message.id);
  });
  const evaluate = expression =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      replies.set(id, message => {
        if (message.error || message.result?.exceptionDetails) {
          reject(new Error(JSON.stringify(message.error || message.result.exceptionDetails)));
        } else {
          resolve(message.result.result.value);
        }
      });
      socket.send(
        JSON.stringify({
          id,
          method: "Runtime.evaluate",
          params: { expression, returnByValue: true },
        }),
      );
    });

  let result = "pending";
  while (Date.now() < deadline && result === "pending") {
    result = await evaluate("document.body?.dataset.result || 'pending'");
    if (result === "pending") await sleep(250);
  }
  const detail = await evaluate("document.body?.textContent || ''");
  if (result !== "passed") throw new Error(detail || `smoke test ended as ${result}`);
  console.log(detail);
} finally {
  await cleanup();
}
