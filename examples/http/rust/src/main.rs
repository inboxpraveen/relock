//! Use relock from Rust through its HTTP API.
//!
//!     relock serve &            # or: docker run -p 8765:8765 -e RELOCK_TOKEN=... ghcr.io/inboxpraveen/relock
//!     cargo run
//!
//! The orders API moved from MM/DD/YYYY to ISO dates. Two trusted failures later, relock has
//! rewritten the rule in place, and the third attempt succeeds.

use serde_json::{json, Value};
use std::env;

struct Relock {
    base: String,
    token: Option<String>,
}

impl Relock {
    fn new() -> Self {
        let base = env::var("RELOCK_URL").unwrap_or_else(|_| "http://127.0.0.1:8765".into());
        Relock { base: format!("{base}/v1"), token: env::var("RELOCK_TOKEN").ok() }
    }

    fn call(&self, method: &str, path: &str, body: Option<Value>) -> Result<Value, String> {
        let mut req = ureq::request(method, &format!("{}{}", self.base, path));
        if let Some(t) = &self.token {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        let res = match body {
            Some(b) => req.send_json(b),
            None => req.call(),
        };
        match res {
            Ok(r) => r.into_json().map_err(|e| e.to_string()),
            Err(ureq::Error::Status(code, r)) => {
                let v: Value = r.into_json().unwrap_or(Value::Null);
                Err(format!("HTTP {code}: {}", v["error"]["message"]))
            }
            Err(e) => Err(e.to_string()),
        }
    }

    fn post(&self, path: &str, body: Value) -> Result<Value, String> {
        self.call("POST", path, Some(body))
    }
}

// Stand-ins for your model and your real tool.
fn agent_picks_date(context: &str) -> &'static str {
    if context.contains("YYYY-MM-DD") { "2026-03-14" } else { "03/14/2026" }
}

fn get_orders(date: &str) -> bool {
    date.len() == 10 && date.as_bytes()[4] == b'-' && date.as_bytes()[7] == b'-'
}

fn main() -> Result<(), String> {
    let relock = Relock::new();
    relock.post("/items", json!({
        "title": "Fetch orders by date",
        "key": "orders.fetch",
        "tools": ["get_orders"],
        "exist_ok": true,
        "clauses": [
            "Authenticate with the orders API key before calling get_orders.",
            "Pass the date to get_orders as MM/DD/YYYY."
        ]
    }))?;

    for attempt in 1..=3 {
        let ep = relock.post("/episodes", json!({"task": "orders for March 14"}))?;
        let id = ep["episode_id"].as_str().ok_or("no episode id")?.to_string();
        let recall = relock.post(&format!("/episodes/{id}/recall"),
                                 json!({"query": "fetch the orders placed on a date"}))?;
        let date = agent_picks_date(recall["context"].as_str().unwrap_or(""));
        let ok = get_orders(date);
        let mut outcome = json!({"tool": "get_orders", "ok": ok});
        if !ok {
            outcome["error"] = json!({
                "message": format!("invalid date '{date}'"),
                "status": 422,
                "correction": "Pass the date to get_orders as YYYY-MM-DD."
            });
        }
        let out = relock.post(&format!("/episodes/{id}/tool-result"), outcome)?;
        relock.post(&format!("/episodes/{id}/close"), json!({}))?;
        println!("attempt {attempt}: get_orders({date}) {} -> {}",
                 if ok { "ok" } else { "failed" }, out["observations"][0]["verdict"].as_str().unwrap_or("?"));
    }

    let history = relock.call("GET", "/items/orders.fetch/history", None)?;
    for v in history["versions"].as_array().ok_or("no versions")? {
        let clauses: Vec<&str> = v["clauses"].as_array().unwrap().iter()
            .filter_map(|c| c["text"].as_str()).collect();
        println!("v{} [{}] {}", v["version"], v["status"].as_str().unwrap_or("?"), clauses.join(" | "));
    }
    Ok(())
}
