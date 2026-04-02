/**
 * SearchBarbara Frontend Telemetry
 *
 * Lightweight client-side observability:
 * - fetch() wrapper: injects X-Trace-ID, records API call durations
 * - SSE event interval tracking
 * - Page render timing helper
 * - Buffered batch sender → POST /api/telemetry
 * - sendBeacon fallback on page unload
 */
(function (global) {
  "use strict";

  // ---- config ----
  const FLUSH_INTERVAL_MS = 5000;
  const FLUSH_BATCH_SIZE = 50;
  const TELEMETRY_ENDPOINT = "/api/telemetry";

  // ---- state ----
  let _sessionId = "";
  let _buffer = [];
  let _flushTimer = null;

  // ---- trace ID ----
  function generateTraceId() {
    const arr = new Uint8Array(8);
    crypto.getRandomValues(arr);
    return Array.from(arr, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  // ---- buffered send ----
  function _enqueue(event) {
    event.client_ts = new Date().toISOString();
    _buffer.push(event);
    if (_buffer.length >= FLUSH_BATCH_SIZE) {
      _flush();
    }
  }

  function _flush() {
    if (_buffer.length === 0) return;
    const batch = _buffer.splice(0);
    const body = JSON.stringify({
      session_id: _sessionId,
      events: batch,
    });
    try {
      // Use the raw fetch to avoid recursion through our wrapper
      _rawFetch(TELEMETRY_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
        keepalive: true,
      }).catch(function () {
        /* best-effort */
      });
    } catch (_e) {
      /* best-effort */
    }
  }

  function _startFlushTimer() {
    if (_flushTimer) return;
    _flushTimer = setInterval(_flush, FLUSH_INTERVAL_MS);
  }

  // ---- fetch wrapper ----
  const _rawFetch = global.fetch.bind(global);

  function tracedFetch(input, init) {
    // Don't trace telemetry calls to avoid infinite loop
    const url =
      typeof input === "string"
        ? input
        : input instanceof Request
          ? input.url
          : String(input);
    if (url.indexOf(TELEMETRY_ENDPOINT) !== -1) {
      return _rawFetch(input, init);
    }

    const traceId = generateTraceId();
    init = init || {};
    init.headers = init.headers || {};
    // Support Headers object or plain object
    if (init.headers instanceof Headers) {
      if (!init.headers.has("X-Trace-ID")) {
        init.headers.set("X-Trace-ID", traceId);
      }
    } else {
      if (!init.headers["X-Trace-ID"]) {
        init.headers["X-Trace-ID"] = traceId;
      }
    }

    const method = (init.method || "GET").toUpperCase();
    const t0 = performance.now();

    return _rawFetch(input, init).then(
      function (response) {
        const durationMs = Math.round(performance.now() - t0);
        _enqueue({
          type: "api_call",
          trace_id: traceId,
          method: method,
          url: url,
          status: response.status,
          duration_ms: durationMs,
        });
        return response;
      },
      function (err) {
        const durationMs = Math.round(performance.now() - t0);
        _enqueue({
          type: "api_call",
          trace_id: traceId,
          method: method,
          url: url,
          status: 0,
          duration_ms: durationMs,
          error: String(err),
        });
        throw err;
      },
    );
  }

  // Replace global fetch
  global.fetch = tracedFetch;

  // ---- SSE tracking ----
  function trackSSE(eventSource, label) {
    if (!eventSource) return;
    let lastEventTs = performance.now();
    let eventCount = 0;

    // Use addEventListener so we don't interfere with the app's onmessage handler
    eventSource.addEventListener("message", function () {
      const now = performance.now();
      const gapMs = Math.round(now - lastEventTs);
      lastEventTs = now;
      eventCount++;

      // Log every 5th event or if gap > 2s to avoid flooding
      if (eventCount % 5 === 0 || gapMs > 2000) {
        _enqueue({
          type: "sse_event",
          label: label,
          event_count: eventCount,
          gap_ms: gapMs,
        });
      }
    });
  }

  // ---- render timing ----
  function trackRender(label) {
    const t0 = performance.now();
    return {
      end: function () {
        const durationMs = Math.round(performance.now() - t0);
        _enqueue({
          type: "render",
          label: label,
          duration_ms: durationMs,
        });
      },
    };
  }

  // ---- session setter ----
  function setSession(sessionId) {
    _sessionId = String(sessionId || "");
  }

  function track(type, fields) {
    if (!type) return;
    const evt = Object.assign({}, fields || {});
    evt.type = String(type);
    _enqueue(evt);
  }

  // ---- page unload: flush via sendBeacon ----
  global.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden" && _buffer.length > 0) {
      var body = JSON.stringify({
        session_id: _sessionId,
        events: _buffer.splice(0),
      });
      try {
        navigator.sendBeacon(TELEMETRY_ENDPOINT, body);
      } catch (_e) {
        /* best-effort */
      }
    }
  });

  // ---- public API ----
  global.SBTelemetry = {
    generateTraceId: generateTraceId,
    trackSSE: trackSSE,
    trackRender: trackRender,
    track: track,
    setSession: setSession,
    flush: _flush,
  };

  // Start periodic flush
  _startFlushTimer();
})(window);
