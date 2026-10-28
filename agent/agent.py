"""
The spend agent: a local LLM (Qwen 2.5 in Ollama) that answers procurement questions
by calling read-only tools, then can draft a savings memo for a human to approve.

Try it from the terminal:
    python agent/agent.py "Where can we save money on safety equipment?"
"""
from pathlib import Path
import json
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import SpendTools, TOOL_SPECS, DB_PATH  # noqa: E402

MODEL = "qwen2.5:7b"
MAX_STEPS = 6

SYSTEM = """You are a procurement spend analyst for the City of Austin.
Data: city purchase orders Oct 2009 - Sep 2026 (321,625 lines). Savings tables cover the last 3 years.

Rules:
- Get every number from a tool. Never invent or estimate numbers yourself.
- Start with the most specific tool (search_categories, category_details, price_variance,
  vendor_lookup, top_savings). Use run_sql only when they are not enough.
- Call more than one tool if the question needs it.
- Never add, subtract or calculate numbers yourself. If you need a total, use one a tool returned.
- Say what type each opportunity is (price variance, vendor consolidation, off-contract spend).
- Call all savings "potential savings"; they still need validation by the procurement team.
- Answer in under 200 words: a one-line answer first, then 3-6 bullets with $ figures,
  then one recommended next step.
- If the data cannot answer the question, say so plainly."""


def _chat(**kwargs):
    import ollama
    return ollama.chat(**kwargs)


def _msg(resp):
    return resp["message"] if isinstance(resp, dict) else resp.message


def run_agent(question: str, tools: SpendTools, history: list | None = None,
              model: str = MODEL, chat_fn=_chat) -> dict:
    """Returns {"answer": str, "steps": [ {tool, args, result, seconds} ], "seconds": float}."""
    start = time.time()
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in (history or [])[-4:]:                      # short memory of the conversation
        messages += [{"role": "user", "content": turn["question"]},
                     {"role": "assistant", "content": turn["answer"]}]
    messages.append({"role": "user", "content": question})

    functions = tools.functions()
    steps = []
    for _ in range(MAX_STEPS):
        resp = chat_fn(model=model, messages=messages, tools=TOOL_SPECS, options={"temperature": 0})
        msg = _msg(resp)
        calls = (msg.get("tool_calls") if isinstance(msg, dict) else msg.tool_calls) or []
        if not calls:
            content = msg.get("content") if isinstance(msg, dict) else msg.content
            return {"answer": content or "(no answer)", "steps": steps, "seconds": time.time() - start}
        messages.append(msg)
        for call in calls:
            fn = call["function"] if isinstance(call, dict) else call.function
            name = fn["name"] if isinstance(fn, dict) else fn.name
            args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            t0 = time.time()
            if name in functions:
                try:
                    result = functions[name](**(args or {}))
                except TypeError as exc:
                    result = f"ERROR: bad arguments for {name}: {exc}"
            else:
                result = f"ERROR: unknown tool {name}"
            steps.append({"tool": name, "args": args, "result": result[:4000],
                          "seconds": round(time.time() - t0, 2)})
            messages.append({"role": "tool", "content": result[:6000], "tool_name": name})

    # Too many steps: ask for a final answer with what it has
    messages.append({"role": "user", "content": "Answer now using only the tool results above."})
    msg = _msg(chat_fn(model=model, messages=messages, options={"temperature": 0}))
    content = msg.get("content") if isinstance(msg, dict) else msg.content
    return {"answer": content, "steps": steps, "seconds": time.time() - start}


MEMO_PROMPT = """Write a short savings memo (under 300 words) to the City of Austin Procurement Director.
Use ONLY the facts and numbers in the analysis below. Copy numbers exactly.
Do NOT add up, round or calculate any number. If a total appears in the tool results you may use it.

Format:
Subject: ...
Summary: 2 sentences.
Findings: 3-5 bullets with $ figures.
Recommended actions: 2-4 numbered steps.
Caveats: 1-2 bullets (figures are potential savings and need validation).

Question asked: {question}

Analysis answer:
{answer}

Tool results used:
{evidence}"""


_MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*([KkMmBb](?:illion)?)?")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def check_numbers(text: str, steps: list) -> dict:
    """Find every $ figure in `text` and check it against the numbers the tools returned.

    "$3,813,259" must match a tool number within $1; "$3.8M" must match within half
    of its last digit (so $3.8M matches 3,813,259). Returns verified and unverified figures.
    """
    evidence = [float(n) for s in steps for n in _NUM.findall(s["result"].replace(",", ""))]
    verified, unverified = [], []
    for m in _MONEY.finditer(text):
        raw, suffix = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try:
            value = float(raw)
        except ValueError:
            continue
        scale = {"k": 1e3, "m": 1e6, "b": 1e9}.get(suffix[:1], 1)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol = max(1.0, 0.5 * scale / (10 ** decimals)) if scale > 1 else 1.0
        target = value * scale
        (verified if any(abs(target - e) <= tol for e in evidence) else unverified).append(m.group(0).strip())
    return {"verified": verified, "unverified": unverified}


def draft_memo(question: str, result: dict, model: str = MODEL, chat_fn=_chat) -> str:
    evidence = "\n".join(f"- {s['tool']}({s['args']}): {s['result'][:1200]}" for s in result["steps"])
    prompt = MEMO_PROMPT.format(question=question, answer=result["answer"], evidence=evidence or "(none)")
    msg = _msg(chat_fn(model=model, messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0.2}))
    return msg.get("content") if isinstance(msg, dict) else msg.content


if __name__ == "__main__":
    import duckdb
    q = " ".join(sys.argv[1:]) or "What are the top 5 savings opportunities?"
    con = duckdb.connect(str(DB_PATH), read_only=True)
    res = run_agent(q, SpendTools(con))
    for s in res["steps"]:
        print(f"[tool] {s['tool']}({s['args']})  -> {s['result'][:150]}...")
    print(f"\n{res['answer']}\n\n({res['seconds']:.1f} s, {len(res['steps'])} tool calls)")
    check = check_numbers(res["answer"], res["steps"])
    print(f"Number check: {len(check['verified'])} $ figures verified against the data"
          + (f"; NOT FOUND in the data: {', '.join(check['unverified'])}" if check["unverified"] else ""))
