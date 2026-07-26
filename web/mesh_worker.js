import init from "./pkg/caso_app.js";

try {
  await init();
} catch (error) {
  self.postMessage(JSON.stringify({
    kind: "error",
    session_id: 0,
    request_id: null,
    operation: "startup",
    error: String(error),
  }));
}
