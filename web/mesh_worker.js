import init from "./pkg/caso_app.js";

try {
  await init();
  self.postMessage({ kind: "ready" });
} catch (error) {
  self.postMessage({ kind: "error", error: String(error) });
}
