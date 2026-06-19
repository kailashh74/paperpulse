import streamlit as st
from retriever import Retriever
from groq import Groq
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os, json, threading, time
import numpy as np

load_dotenv()

st.set_page_config(page_title="PaperPulse", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Cabinet+Grotesk:wght@400;500;700;800&display=swap');
:root {
    --bg:#0a0a0a; --bg1:#111111; --bg2:#161616; --border:#222222; --border2:#2a2a2a;
    --text:#e8e8e8; --muted:#555555; --cyan:#00d4ff;
    --cyan-dim:rgba(0,212,255,0.08); --mono:'DM Mono',monospace; --display:'Cabinet Grotesk',sans-serif;
}
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:var(--mono);background:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
.pp-nav{position:sticky;top:0;z-index:100;background:var(--bg);border-bottom:1px solid var(--border);
    padding:0 40px;display:flex;align-items:center;height:56px;}
.pp-logo{font-family:var(--display);font-size:1.1rem;font-weight:800;color:var(--text);
    letter-spacing:-0.03em;margin-right:48px;display:flex;align-items:center;gap:8px;}
.pp-logo-dot{width:7px;height:7px;background:var(--cyan);border-radius:50%;animation:pulse-dot 2s ease-in-out infinite;}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.4;transform:scale(0.7);}}
.pp-nav-right{display:flex;align-items:center;gap:20px;font-size:0.7rem;color:var(--muted);}
.pp-status{display:flex;align-items:center;gap:6px;}
.pp-status-dot{width:5px;height:5px;background:#22c55e;border-radius:50%;}
.pp-stat{display:flex;justify-content:space-between;align-items:baseline;padding:8px 0;border-bottom:1px solid var(--border);}
.pp-stat-label{font-size:0.68rem;color:var(--muted);}
.pp-stat-value{font-size:0.9rem;font-weight:500;color:var(--cyan);letter-spacing:-0.02em;}
.pp-section{font-size:0.62rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--muted);margin:24px 0 10px;}
.pp-page-title{font-family:var(--display);font-size:2rem;font-weight:800;color:var(--text);
    letter-spacing:-0.04em;line-height:1.1;margin-bottom:6px;}
.pp-page-sub{font-size:0.75rem;color:var(--muted);margin-bottom:32px;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
    background:var(--bg1)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:14px 18px!important;margin-bottom:6px!important;}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){
    background:var(--bg)!important;border:1px solid var(--border)!important;border-left:2px solid var(--cyan)!important;
    border-radius:8px!important;padding:14px 18px!important;margin-bottom:12px!important;}
[data-testid="stChatInput"] textarea{background:var(--bg1)!important;border:1px solid var(--border2)!important;
    border-radius:8px!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:0.85rem!important;padding:14px 18px!important;}
[data-testid="stChatInput"] textarea:focus{border-color:rgba(0,212,255,0.35)!important;box-shadow:0 0 0 3px rgba(0,212,255,0.06)!important;}
.conf-wrap{margin:10px 0 4px;display:flex;align-items:center;gap:12px;}
.conf-track{flex:1;height:2px;background:var(--border2);border-radius:2px;overflow:hidden;}
.conf-fill{height:100%;border-radius:2px;}
.conf-label{font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;min-width:110px;}
.src-tag{display:inline-flex;align-items:center;gap:6px;background:var(--bg2);border:1px solid var(--border2);
    border-radius:4px;padding:3px 10px;font-size:0.68rem;color:var(--muted);margin:3px 3px 3px 0;
    text-decoration:none;transition:all 0.15s;}
.src-tag:hover{border-color:var(--cyan);color:var(--cyan);}
.digest-card{border:1px solid var(--border);border-radius:8px;padding:20px 24px;margin-bottom:16px;
    background:var(--bg1);position:relative;overflow:hidden;transition:border-color 0.2s;}
.digest-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;
    background:var(--cyan);opacity:0.4;transition:opacity 0.2s;}
.digest-card:hover::before{opacity:1;}
.digest-title{font-family:var(--display);font-size:0.95rem;font-weight:700;color:var(--text);
    letter-spacing:-0.02em;margin-bottom:6px;line-height:1.3;}
.digest-meta{font-size:0.65rem;color:var(--muted);margin-bottom:10px;display:flex;gap:16px;flex-wrap:wrap;}
.digest-body{font-size:0.78rem;color:#aaaaaa;line-height:1.7;}
.digest-body strong{color:var(--text);}
.digest-tag{display:inline-block;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;
    padding:2px 8px;border-radius:3px;margin-right:6px;}
.tag-b{background:rgba(34,197,94,0.1);color:#22c55e;border:1px solid rgba(34,197,94,0.2);}
.tag-i{background:rgba(234,179,8,0.1);color:#eab308;border:1px solid rgba(234,179,8,0.2);}
.tag-a{background:rgba(239,68,68,0.1);color:#ef4444;border:1px solid rgba(239,68,68,0.2);}
.pulse-row{display:flex;align-items:center;gap:14px;padding:10px 0;border-bottom:1px solid var(--border);}
.pulse-topic{font-size:0.72rem;color:var(--muted);width:210px;flex-shrink:0;}
.pulse-track{flex:1;height:3px;background:var(--border);border-radius:3px;overflow:hidden;}
.pulse-fill{height:100%;background:var(--cyan);border-radius:3px;opacity:0.6;}
.pulse-count{font-size:0.72rem;color:var(--cyan);width:40px;text-align:right;}
[data-testid="stMetric"]{background:var(--bg1);border:1px solid var(--border);border-radius:8px;padding:16px!important;}
[data-testid="stMetricLabel"]{font-family:var(--mono)!important;font-size:0.65rem!important;
    letter-spacing:0.1em!important;text-transform:uppercase!important;color:var(--muted)!important;}
[data-testid="stMetricValue"]{font-family:var(--mono)!important;font-size:1.8rem!important;
    font-weight:500!important;color:var(--cyan)!important;letter-spacing:-0.03em!important;}
/* FIX: style st.button tabs to look like nav tabs */
.tab-row {display:flex;gap:2px;margin-bottom:0;}
.stButton button{background:var(--bg2)!important;border:1px solid var(--border2)!important;
    border-radius:6px!important;color:var(--muted)!important;font-family:var(--mono)!important;
    font-size:0.7rem!important;letter-spacing:0.05em!important;padding:8px 16px!important;
    text-transform:uppercase!important;transition:all 0.15s!important;width:auto!important;}
.stButton button:hover{border-color:var(--cyan)!important;color:var(--cyan)!important;background:var(--cyan-dim)!important;}
section[data-testid="stSidebar"]{display:none!important;}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
.fade-1{animation:fadeUp 0.4s ease both;}
.fade-2{animation:fadeUp 0.4s ease 0.08s both;}
</style>
""", unsafe_allow_html=True)

# ── Scheduler ─────────────────────────────────────────────────
_scheduler_lock = threading.Lock()
_scheduler_running = False

def _run_scheduler():
    import schedule as sch
    def _job():
        try:
            from live_ingest import run_update
            run_update()
            retriever.reload()
        except Exception as e:
            print(f"[Scheduler] Error: {e}")
    sch.every().week.do(_job)
    while True:
        sch.run_pending()
        time.sleep(3600)

def start_scheduler():
    global _scheduler_running
    with _scheduler_lock:
        if not _scheduler_running:
            threading.Thread(target=_run_scheduler, daemon=True).start()
            _scheduler_running = True

start_scheduler()

# ── Core ──────────────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

# FIX: include digest_html in session state defaults
for key, val in [("messages",[]),("active_tab","ask"),("answer_mode","technical"),
                  ("reading_list",[]),("last_update",datetime.now()),
                  ("digest_html",""),("digest_count",0),("pending_query",None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Metadata ──────────────────────────────────────────────────
BASE = Path(os.path.dirname(os.path.abspath(__file__)))
meta = []
meta_path = BASE / "metadata.json"
if meta_path.exists():
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

research = [m for m in meta if m.get("type") == "research"]
paper_count = len(research)
total_chunks = len(meta)

latest_ts = max((r.get("timestamp","") for r in research), default="") if research else ""
try:
    latest_fmt = datetime.strptime(latest_ts[:10], "%Y-%m-%d").strftime("%b %d, %Y")
except:
    latest_fmt = "—"

tab = st.session_state.active_tab

# ── Nav ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="pp-nav fade-1">
  <div class="pp-logo"><div class="pp-logo-dot"></div>PaperPulse</div>
  <div class="pp-nav-right">
    <div class="pp-status"><div class="pp-status-dot"></div><span>{paper_count:,} papers</span></div>
    <span style="color:var(--border2)">|</span>
    <span>Updated {latest_fmt}</span>
  </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 3.2], gap="large")

# ── Sidebar ───────────────────────────────────────────────────
with left:
    # FIX: tabs as st.button so session_state persists across clicks
    st.markdown('<div class="pp-section">Navigation</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    with t1:
        if st.button("◈ Digest"):
            st.session_state.active_tab = "digest"
            st.rerun()
    with t2:
        if st.button("◎ Ask"):
            st.session_state.active_tab = "ask"
            st.rerun()
    with t3:
        if st.button("▸ Pulse"):
            st.session_state.active_tab = "pulse"
            st.rerun()

    st.markdown(f'<div style="font-size:0.6rem;color:var(--cyan);margin-bottom:8px">▸ {tab.upper()}</div>', unsafe_allow_html=True)

    st.markdown('<div class="pp-section">Index</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.metric("Papers", f"{paper_count:,}")
    with c2: st.metric("Chunks", f"{total_chunks:,}")

    last_upd = st.session_state.get("last_update", datetime.now())
    st.markdown(f"""
    <div style="margin-top:12px">
      <div class="pp-stat"><span class="pp-stat-label">Latest paper</span><span class="pp-stat-value">{latest_fmt}</span></div>
      <div class="pp-stat"><span class="pp-stat-label">Last fetched</span><span class="pp-stat-value">{last_upd.strftime('%b %d %H:%M')}</span></div>
      <div class="pp-stat"><span class="pp-stat-label">Schedule</span><span class="pp-stat-value" style="color:#22c55e">weekly ●</span></div>
      <div class="pp-stat"><span class="pp-stat-label">Source</span><span class="pp-stat-value">arXiv</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pp-section">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Results", 2, 10, 5, label_visibility="collapsed")
    show_sources = st.checkbox("Show sources", value=True)

    st.markdown('<div class="pp-section">Actions</div>', unsafe_allow_html=True)

    if st.button("⟳  Fetch latest"):
        status = st.empty()
        status.info("Fetching...")
        def _fetch():
            try:
                from live_ingest import run_update
                run_update()
                retriever.reload()
            except Exception as e:
                print(f"[Fetch] Error: {e}")
        t = threading.Thread(target=_fetch, daemon=True)
        t.start(); t.join(timeout=3)
        st.session_state.last_update = datetime.now()
        st.cache_resource.clear()
        status.success("Fetch started.")

    if st.button("✕  Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.reading_list:
        st.markdown('<div class="pp-section">Reading list</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="pp-stat-label">{len(st.session_state.reading_list)} saved</div>', unsafe_allow_html=True)
        if st.button("Export"):
            st.code("\n".join([f"- {p['title']}: {p['url']}" for p in st.session_state.reading_list]))

# ── Main ──────────────────────────────────────────────────────
with right:

    # ════════════════════════════════════════
    # DIGEST
    # ════════════════════════════════════════
    if tab == "digest":
        st.markdown("""
        <div class="fade-1">
          <div class="pp-page-title">This week in AI</div>
          <div class="pp-page-sub">Curated digest from your arXiv index — grouped by subfield</div>
        </div>""", unsafe_allow_html=True)

        col_gen, col_clr = st.columns([2,1])
        with col_gen:
            gen = st.button("◈  Generate digest", key="gen_digest")
        with col_clr:
            if st.button("↺  Refresh", key="clr_digest"):
                st.session_state.digest_html = ""
                st.session_state.digest_count = 0
                st.rerun()

        if gen:
            if not research:
                st.warning("No papers indexed.")
            else:
                recent = sorted(research, key=lambda x: x.get("timestamp",""), reverse=True)[:40]
                context = "\n\n".join([
                    f"[{r.get('timestamp','')[:10]}] [Fields: {', '.join(r.get('fields',[])[:2])}] "
                    f"[Authors: {r.get('authors','')}]\n"
                    f"Title: {r.get('title','')}\n"
                    f"Abstract: {r['text'][len(r.get('title',''))+2:500]}"
                    for r in recent
                ])
                prompt = f"""You are PaperPulse — an AI research digest for CS/ML students and engineers.

From these recent arXiv papers, select 6 covering DIFFERENT subfields (LLM, Vision, RL, Safety, Efficiency, Agents, RAG, Multimodal, Robotics).

Output EXACTLY this format for each, separated by ---:
TITLE: <exact paper title from the list>
DATE: <YYYY-MM-DD>
AUTHORS: <authors>
SUBFIELD: <LLM|Vision|RL|Safety|Efficiency|Agents|RAG|Multimodal|Robotics|Other>
DIFFICULTY: <Beginner|Intermediate|Advanced>
WHAT: <one sentence — what this paper does>
WHY: <one sentence — why it matters>
PREREQ: <one sentence — background needed>
---

Papers:
{context}

Generate 6 digests:"""

                with st.spinner("Generating digest..."):
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=2500,
                    )
                    digest_text = resp.choices[0].message.content

                all_cards = ""
                rendered = 0
                for entry in digest_text.strip().split("---"):
                    entry = entry.strip()
                    if not entry:
                        continue
                    lines = {}
                    for line in entry.split("\n"):
                        if ":" in line:
                            k, v = line.split(":", 1)
                            lines[k.strip()] = v.strip()
                    title = lines.get("TITLE","").strip()
                    if not title:
                        continue
                    date     = lines.get("DATE","")
                    authors  = lines.get("AUTHORS","")[:60]
                    subfield = lines.get("SUBFIELD","")
                    diff     = lines.get("DIFFICULTY","Intermediate")
                    what     = lines.get("WHAT","")
                    why      = lines.get("WHY","")
                    prereq   = lines.get("PREREQ","")
                    diff_cls = {"Beginner":"tag-b","Intermediate":"tag-i","Advanced":"tag-a"}.get(diff,"tag-i")
                    url = next((r.get("url","") for r in recent if title.lower()[:35] in r.get("title","").lower()), "")
                    url_html = f'<a href="{url}" target="_blank" style="color:var(--cyan);font-size:0.65rem;margin-left:8px">↗ arXiv</a>' if url else ""
                    all_cards += f"""
<div class="digest-card">
  <div class="digest-title">{title}{url_html}</div>
  <div class="digest-meta"><span>{date}</span><span>{authors}</span><span>{subfield}</span></div>
  <div style="margin-bottom:10px"><span class="digest-tag {diff_cls}">{diff}</span></div>
  <div class="digest-body">
    <strong>What:</strong> {what}<br><br>
    <strong>Why it matters:</strong> {why}<br><br>
    <strong>You should know:</strong> {prereq}
  </div>
</div>"""
                    rendered += 1

                if rendered > 0:
                    # FIX: store in session_state — survives reruns
                    st.session_state.digest_html = all_cards
                    st.session_state.digest_count = rendered
                else:
                    st.warning("Parsing failed. Try again.")

        # FIX: always render from session_state
        if st.session_state.digest_html:
            st.markdown(st.session_state.digest_html, unsafe_allow_html=True)
            st.caption(f"{st.session_state.digest_count} papers · arXiv index")
        elif not gen:
            st.markdown("""
            <div style="border:1px dashed var(--border2);border-radius:8px;padding:64px;text-align:center;margin-top:24px">
              <div style="font-size:0.7rem;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px">Ready</div>
              <div style="font-family:var(--display);font-size:1.6rem;font-weight:800;color:var(--text);letter-spacing:-0.03em">Generate this week's digest</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:10px">6 papers · multiple subfields · from your arXiv index</div>
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # ASK
    # ════════════════════════════════════════
    elif tab == "ask":
        st.markdown("""
        <div class="fade-1">
          <div class="pp-page-title">Ask anything</div>
          <div class="pp-page-sub">Hybrid BM25 + semantic retrieval over arXiv papers — live search fallback</div>
        </div>""", unsafe_allow_html=True)

        col_t, _ = st.columns([2,5])
        with col_t:
            mode = st.radio("Mode", ["Technical","Simple"], horizontal=True, label_visibility="collapsed",
                           index=0 if st.session_state.answer_mode=="technical" else 1)
            st.session_state.answer_mode = "technical" if mode=="Technical" else "simple"

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    conf = msg.get("confidence", 0.5)
                    tier = msg.get("tier","index")
                    raw_sem = msg.get("raw_sem", None)

                    # FIX: honest confidence — use raw semantic similarity floor
                    # if best semantic score is below 0.3, cap at Low regardless of RRF
                    if raw_sem is not None and raw_sem < 0.3:
                        conf = min(conf, 0.3)

                    if conf >= 0.6:    bar_color="#22c55e"; conf_label="High confidence"
                    elif conf >= 0.35: bar_color="#eab308"; conf_label="Medium confidence"
                    else:              bar_color="#ef4444"; conf_label="Low confidence"

                    tier_label = {"index":"hybrid index","scholar":"live search","none":"no match"}.get(tier,tier)
                    st.markdown(f"""
                    <div class="conf-wrap">
                      <div class="conf-track"><div class="conf-fill" style="width:{min(conf*100,100):.0f}%;background:{bar_color}"></div></div>
                      <span class="conf-label" style="color:{bar_color}">{conf_label}</span>
                      <span style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em">via {tier_label}</span>
                    </div>""", unsafe_allow_html=True)

                    if msg.get("sources") and show_sources:
                        with st.expander(f"◎ {len(msg['sources'])} sources"):
                            for c in msg["sources"]:
                                url = c.get("url","")
                                score = c.get("score",0)
                                sem = c.get("sem_score",0)
                                ts = c.get("timestamp","")[:10]
                                authors = c.get("authors","")
                                try: date_str = datetime.strptime(ts,"%Y-%m-%d").strftime("%b %d, %Y")
                                except: date_str = ts or "—"
                                if url:
                                    st.markdown(f'<a class="src-tag" href="{url}" target="_blank">↗ arXiv · {date_str} · {authors[:25]} · rrf:{score:.2f} · sem:{sem:.2f}</a>', unsafe_allow_html=True)

                    if msg.get("sources"):
                        srcs = [s for s in msg["sources"] if s.get("url")]
                        if srcs and st.button("+ Save to reading list", key=f"rl_{id(msg)}"):
                            for s in srcs[:2]:
                                e = {"title":s.get("title",""),"url":s.get("url","")}
                                if e not in st.session_state.reading_list:
                                    st.session_state.reading_list.append(e)
                            st.toast("Saved to reading list")

        # FIX: pull a pending query from Pulse tab handoff, same as typed input
        pending = st.session_state.pending_query
        st.session_state.pending_query = None
        typed = st.chat_input("Ask about AI research...")
        query = pending or typed

        if query:
            st.session_state.messages.append({"role":"user","content":query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                # FIX: detect follow-up intent — short queries with no topical
                # content of their own ("explain simpler", "tell me more", etc)
                # should reuse the previous turn's sources, not re-retrieve blind.
                FOLLOWUP_PHRASES = [
                    "explain", "simpler", "simple terms", "elaborate", "more detail",
                    "tell me more", "go on", "continue", "what do you mean",
                    "can you clarify", "rephrase", "in other words", "summarize that",
                    "summarise that", "break it down", "dumb it down", "eli5"
                ]
                q_lower = query.lower().strip()
                is_followup = (
                    len(q_lower.split()) <= 8
                    and any(p in q_lower for p in FOLLOWUP_PHRASES)
                    and len(st.session_state.messages) >= 2
                )

                prev_chunks = None
                if is_followup:
                    # find the most recent assistant message with sources
                    for m in reversed(st.session_state.messages[:-1]):
                        if m["role"] == "assistant" and m.get("sources"):
                            prev_chunks = m["sources"]
                            break

                if is_followup and prev_chunks:
                    # FIX: reuse previous sources — don't re-retrieve on vague follow-ups
                    chunks = prev_chunks
                    top_score = chunks[0]["score"] if chunks else 0.0
                    raw_sem_score = chunks[0].get("sem_score", 0.0) if chunks else 0.0
                    tier_used = "index"
                else:
                    chunks = retriever.retrieve(query, top_k=top_k)
                    top_score = chunks[0]["score"] if chunks else 0.0
                    raw_sem_score = chunks[0].get("sem_score", 0.0) if chunks else 0.0
                    tier_used = "index"

                    if top_score < 0.40 or raw_sem_score < 0.25:
                        try:
                            from fetchers import fetch_semantic_scholar
                            live = fetch_semantic_scholar(query, max_results=top_k)
                            if live:
                                max_cit = max(d.get("citations",0) for d in live) or 1
                                for d in live:
                                    d["score"] = 0.35 + 0.15 * (d.get("citations",0) / max_cit)
                                    d["sem_score"] = d["score"]
                                live.sort(key=lambda x: x["score"], reverse=True)
                                chunks = live[:top_k]
                                top_score = chunks[0]["score"]
                                raw_sem_score = top_score
                                tier_used = "scholar"
                        except Exception as e:
                            print(f"[Tier2] {e}")

                if not chunks:
                    tier_used = "none"

                # FIX: honest confidence — cap if semantic similarity is weak
                display_conf = top_score
                if raw_sem_score < 0.3:
                    display_conf = min(display_conf, 0.3)

                context = "\n\n".join([
                    f"[{c.get('source','')} | {c.get('timestamp','')[:10]} | {c.get('authors','')}]\n{c['text']}"
                    for c in chunks
                ])[:3500]

                mode_inst = (
                    "Use simple analogies. No jargon without explanation. Be conversational."
                    if st.session_state.answer_mode == "simple"
                    else "Be technically precise. Use ML terminology. Cite paper titles and authors."
                )
                tier_inst = {
                    "index":  "Answer from the retrieved arXiv papers. Cite titles and authors.",
                    "scholar":"Answer from these live Semantic Scholar results. Cite titles and authors.",
                    "none":   "No papers were found. Tell the user clearly. Do NOT hallucinate citations."
                }[tier_used]

                history = [{"role":m["role"],"content":m["content"]} for m in st.session_state.messages[-4:] if m["role"] in ["user","assistant"]]
                SYSTEM_PROMPT = """You are PaperPulse — an expert AI/ML research assistant grounded in arXiv papers.

HARD RULE: If the question is about product recommendations, hardware purchasing advice (laptops, GPUs to buy, phones, etc), general consumer tech, or anything that is not an AI/ML research topic, you MUST respond with exactly: "I'm built to answer AI/ML research questions, not [topic]. Try asking me about a specific paper, method, or research area instead." Do NOT use the retrieved context to answer off-topic questions even if it loosely mentions hardware or products. Do NOT invent product specifications or recommendations under any circumstances.

For genuine AI/ML research questions: answer from the retrieved context below. Cite paper titles and authors. If the retrieved context is weak or irrelevant to the question, say so honestly rather than stretching unrelated papers to fit."""

                history.append({"role":"user","content":f"{SYSTEM_PROMPT}\n\n{tier_inst}\n{mode_inst}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"})

                placeholder = st.empty()
                full_response = ""
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=history,
                    max_tokens=1024,
                    stream=True,
                )
                for chunk_data in stream:
                    delta = chunk_data.choices[0].delta.content or ""
                    full_response += delta
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

                conf = float(display_conf)
                if conf >= 0.6:    bar_color="#22c55e"; conf_label="High confidence"
                elif conf >= 0.35: bar_color="#eab308"; conf_label="Medium confidence"
                else:              bar_color="#ef4444"; conf_label="Low confidence"
                tier_label = {"index":"hybrid index","scholar":"live search","none":"no match"}.get(tier_used,tier_used)

                st.markdown(f"""
                <div class="conf-wrap">
                  <div class="conf-track"><div class="conf-fill" style="width:{min(conf*100,100):.0f}%;background:{bar_color}"></div></div>
                  <span class="conf-label" style="color:{bar_color}">{conf_label}</span>
                  <span style="font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em">via {tier_label}</span>
                </div>""", unsafe_allow_html=True)

                if show_sources and chunks:
                    with st.expander(f"◎ {len(chunks)} sources"):
                        for c in chunks:
                            url = c.get("url","")
                            ts = c.get("timestamp","")[:10]
                            authors = c.get("authors","")
                            score = c.get("score",0)
                            sem = c.get("sem_score",0)
                            try: date_str = datetime.strptime(ts,"%Y-%m-%d").strftime("%b %d, %Y")
                            except: date_str = ts or "—"
                            if url:
                                st.markdown(f'<a class="src-tag" href="{url}" target="_blank">↗ arXiv · {date_str} · {authors[:25]} · rrf:{score:.2f} · sem:{sem:.2f}</a>', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role":"assistant","content":full_response,
                "sources":chunks,"confidence":float(display_conf),
                "raw_sem":float(raw_sem_score),"tier":tier_used
            })

    # ════════════════════════════════════════
    # PULSE
    # ════════════════════════════════════════
    elif tab == "pulse":
        st.markdown("""
        <div class="fade-1">
          <div class="pp-page-title">Topic pulse</div>
          <div class="pp-page-sub">Paper volume by AI subfield — live from your index</div>
        </div>""", unsafe_allow_html=True)

        TOPICS = {
            "Large language models": ["large language","llm","gpt","language model","instruction tuning"],
            "RAG & retrieval":       ["retrieval augmented","rag","vector search","dense retrieval","embedding"],
            "LLM agents":            ["agent","tool use","autonomous","agentic","function calling"],
            "Diffusion models":      ["diffusion","stable diffusion","image generation","denoising","flow matching"],
            "Vision & multimodal":   ["vision language","multimodal","visual","vqa","image text","clip"],
            "Reinforcement learning":["reinforcement learning","rlhf","reward model","policy","ppo","dpo"],
            "Safety & alignment":    ["safety","hallucination","alignment","bias","red teaming","jailbreak"],
            "Efficient ML":          ["quantization","pruning","distillation","compression","sparse","efficient"],
            "Graph neural nets":     ["graph neural","gnn","knowledge graph","graph transformer"],
            "Fine-tuning":           ["fine-tun","lora","peft","adapter","parameter efficient"],
            "Reasoning":             ["chain of thought","reasoning","cot","step-by-step","math","theorem"],
            "Robotics":              ["robot","embodied","manipulation","locomotion","sim-to-real"],
        }

        counts = {}
        for topic, kws in TOPICS.items():
            counts[topic] = sum(1 for r in research if any(kw in r.get("text","").lower() for kw in kws))

        max_c = max(counts.values()) if counts and max(counts.values()) > 0 else 1
        sorted_topics = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # FIX: one HTML block
        all_rows = ""
        for topic, count in sorted_topics:
            pct = count / max_c * 100
            all_rows += f"""
<div class="pulse-row">
  <span class="pulse-topic">{topic}</span>
  <div class="pulse-track"><div class="pulse-fill" style="width:{pct:.1f}%"></div></div>
  <span class="pulse-count">{count}</span>
</div>"""
        st.markdown(all_rows, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:32px'>", unsafe_allow_html=True)
        topic_choice = st.selectbox("Explore topic", [t for t,_ in sorted_topics], label_visibility="collapsed")
        if st.button("▸  Ask about this topic"):
            st.session_state.active_tab = "ask"
            # FIX: was appending a user message directly with no answer pipeline
            # ever running on it. Now sets a pending flag that the Ask tab
            # picks up and runs through the normal retrieve+answer flow.
            st.session_state.pending_query = f"What are the latest developments in {topic_choice}? Summarize key recent papers."
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)