// The orders API moved from MM/DD/YYYY to ISO dates. Watch relock fix the rule in place.
//   relock serve &            (or: docker run -p 8765:8765 -e RELOCK_TOKEN=... ghcr.io/inboxpraveen/relock)
//   node demo.ts              (Node 22.18+ runs TypeScript directly)
import { Relock } from "./relock.ts";

const relock = new Relock(process.env.RELOCK_URL ?? "http://127.0.0.1:8765");

await relock.addItem("Fetch orders by date", [
  "Authenticate with the orders API key before calling get_orders.",
  "Pass the date to get_orders as MM/DD/YYYY.",
], { key: "orders.fetch", tools: ["get_orders"], exist_ok: true });

// Stand-ins for your model and your real tool.
const agentPicksDate = (context: string) => (context.includes("YYYY-MM-DD") ? "2026-03-14" : "03/14/2026");
const getOrders = (date: string) => /^\d{4}-\d{2}-\d{2}$/.test(date);

for (let attempt = 1; attempt <= 3; attempt++) {
  const ep = await relock.episode("orders for March 14");
  const { context } = await ep.recall("fetch the orders placed on a date");
  const date = agentPicksDate(context);
  const ok = getOrders(date);
  const [obs] = await ep.toolResult("get_orders", ok, ok ? undefined : {
    message: `invalid date '${date}'`, status: 422, correction: "Pass the date to get_orders as YYYY-MM-DD.",
  });
  await ep.close();
  console.log(`attempt ${attempt}: get_orders(${date}) ${ok ? "ok" : "failed"} -> ${obs.verdict}`);
}

const { versions } = await relock.history("orders.fetch");
for (const v of versions) console.log(`v${v.version} [${v.status}] ${v.clauses.map((c) => c.text).join(" | ")}`);
