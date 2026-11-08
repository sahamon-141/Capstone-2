"""
IoT Sentinel: Streamlit frontend for the three-module IDS.

    streamlit run app.py

Pages
    Overview          architecture and headline results
    Live Attack Lab   one traffic window through Module 1 -> 2 -> 3, step by step
    Network Monitor   a stream of windows (normal -> attacks -> zero-day) with an event log
    Model Performance accuracy, confusion matrices, quantisation, zero-day results
    How It Works      the method, formulas and design decisions, for presenting
"""

import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ids.config import (FAMILY_DESCRIPTION, FAMILY_OF, HELD_OUT_CLASSES, METRICS_DIR, NORMAL_LABEL,
                        RISK_ATTACK, RISK_SUSPICIOUS, WINDOW_SIZE)
from ids.module1_detection import RISK_UNIVERSE, RULES

st.set_page_config(page_title="IoT Sentinel IDS", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
.step-head {display:flex; align-items:center; gap:.7rem; margin:.2rem 0 .6rem 0;}
.step-num {background:#2563eb; color:#fff; border-radius:999px; width:2.1rem; height:2.1rem;
           display:flex; align-items:center; justify-content:center; font-weight:700; flex:none;}
.step-title {font-size:1.25rem; font-weight:700;}
.step-sub {color:#64748b; font-size:.92rem;}
.banner {padding:.9rem 1.1rem; border-radius:.6rem; font-weight:600; font-size:1.05rem; margin:.4rem 0;}
.b-green {background:#dcfce7; color:#14532d; border-left:6px solid #16a34a;}
.b-amber {background:#fef3c7; color:#78350f; border-left:6px solid #d97706;}
.b-red   {background:#fee2e2; color:#7f1d1d; border-left:6px solid #dc2626;}
.b-blue  {background:#dbeafe; color:#1e3a8a; border-left:6px solid #2563eb;}
.b-gray  {background:#f1f5f9; color:#334155; border-left:6px solid #94a3b8;}
.pill {display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.8rem; font-weight:700; color:#fff;}
.card {border:1px solid #e2e8f0; border-radius:.7rem; padding:1rem 1.1rem; height:100%;}
.card h4 {margin:.1rem 0 .4rem 0;}
.big {font-size:1.9rem; font-weight:800; line-height:1.1;}
.muted {color:#64748b; font-size:.88rem;}
.gauge {height:18px; border-radius:9px; position:relative;
        background:linear-gradient(90deg,#22c55e 0%,#22c55e 35%,#f59e0b 35%,#f59e0b 65%,#ef4444 65%,#ef4444 100%);}
.gauge .mark {position:absolute; top:-6px; width:4px; height:30px; background:#0f172a; border-radius:2px;}
@media (prefers-color-scheme: dark) {
  .card {border-color:#334155;} .muted,.step-sub {color:#94a3b8;}
}
</style>""", unsafe_allow_html=True)

SEV_COLOR = {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "MEDIUM": "#d97706", "LOW": "#16a34a"}
VERDICT_STYLE = {"NORMAL": "b-green", "SUSPICIOUS": "b-amber", "ATTACK": "b-red"}


def step_header(num, title, sub=""):
    st.markdown(f"<div class='step-head'><div class='step-num'>{num}</div><div>"
                f"<div class='step-title'>{title}</div><div class='step-sub'>{sub}</div></div></div>",
                unsafe_allow_html=True)


def banner(text, style):
    st.markdown(f"<div class='banner {style}'>{text}</div>", unsafe_allow_html=True)


def gauge(value):
    st.markdown(f"<div class='gauge'><div class='mark' style='left:calc({value * 100:.1f}% - 2px)'></div></div>"
                "<div style='display:flex;justify-content:space-between' class='muted'>"
                "<span>0 · NORMAL</span><span>0.35</span><span>SUSPICIOUS</span><span>0.65</span><span>ATTACK · 1</span></div>",
                unsafe_allow_html=True)


def pct(x):
    return f"{100 * x:.1f}%"


@st.cache_resource(show_spinner="Loading models (Module 1 baseline, Module 2 classifiers)...")
def load_pipeline():
    from ids.pipeline import IDSPipeline
    return IDSPipeline()


try:
    P = load_pipeline()
except FileNotFoundError:
    st.error("Models not found. Run `python -m ids.prepare_data` and then `python -m ids.train_module2` first.")
    st.stop()
M = P.metrics

# ---------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ IoT Sentinel")
    st.caption("Detect → Classify → Mitigate\n\nCICIoT2023 · 3-module IDS")
    page = st.radio("Navigate", ["🏠 Overview", "🧪 Live Attack Lab", "📡 Network Monitor",
                                 "📊 Model Performance", "📘 How It Works"], label_visibility="collapsed")
    st.divider()
    engine = st.radio("Module 2 engine", ["Random Forest (accurate)", "INT8 MLP (edge device)"],
                      help="Random Forest is the main classifier. The INT8 MLP is the 8-bit quantised "
                           "neural network meant for a resource-constrained IoT gateway.")
    engine_key = "int8" if engine.startswith("INT8") else "rf"
    m1_mode_label = st.radio("Module 1 inputs", ["IAT timing + behaviour (final)", "IAT timing only (paper v1)"],
                             help="Timing only = CV, rate and IAT deviation (original design). The final version adds "
                                  "a behaviour-deviation score learned from benign traffic only, which catches "
                                  "attacks whose timing looks normal (recon, spoofing, web).")
    m1_mode = "timing" if "only" in m1_mode_label else "full"
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1️⃣ **Module 1** · Is it an attack?  \n2️⃣ **Module 2** · Which attack?  \n3️⃣ **Module 3** · Mitigate & verify")

# =======================================================================================
# PAGE: OVERVIEW
# =======================================================================================
if page.startswith("🏠"):
    st.title("IoT Sentinel: Lightweight Intrusion Detection & Response for IoT")
    st.markdown("A three-stage pipeline that **detects** anomalous IoT traffic from packet timing, "
                "**classifies** the attack type, and **mitigates** it with an attack-specific response "
                "that is verified by re-checking the traffic.")

    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
  edge [fontname="Helvetica", fontsize=10];
  T  [label="IoT traffic\\n(flows)", fillcolor="#e2e8f0"];
  W  [label="Sliding window\\n20 flows", fillcolor="#e2e8f0"];
  M1 [label="MODULE 1\\nIAT statistics + Fuzzy logic\\n→ attack risk 0..1", fillcolor="#bfdbfe"];
  OK [label="NORMAL\\nforward traffic\\n(Module 2 idle)", fillcolor="#bbf7d0"];
  M2 [label="MODULE 2\\nRandom Forest / INT8 MLP\\n→ attack class + zero-day check", fillcolor="#fed7aa"];
  FA [label="Benign\\n(false alarm)", fillcolor="#bbf7d0"];
  M3 [label="MODULE 3\\nPlaybook → action\\n(auto / analyst approval)", fillcolor="#fecaca"];
  V  [label="Verify: re-run Module 1\\nrisk < 0.35 ?", fillcolor="#e9d5ff"];
  T -> W -> M1;
  M1 -> OK [label="risk < 0.35"];
  M1 -> M2 [label="risk ≥ 0.35"];
  M2 -> FA [label="benign"];
  M2 -> M3 [label="attack / unknown"];
  M3 -> V;
  V -> M3 [label="no: escalate", style=dashed];
  V -> OK [label="yes: resolved"];
}""", use_container_width=True)

    m1, m2 = M["module1"], M["module2"]
    rf, mlp, we = m2["random_forest"], m2["mlp"], m2["window_eval"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class='card'><h4>1️⃣ Module 1 · Detection</h4>
        <div class='big'>{pct(m1['recall'])}</div><div class='muted'>of attack windows detected (recall)</div><br>
        Uses only packet <b>inter-arrival time</b> + rate. No labels needed, so it also catches
        <b>unseen</b> attacks: <b>{pct(m1['mirai_zero_day_detection'])}</b> of zero-day Mirai windows flagged.</div>""",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='card'><h4>2️⃣ Module 2 · Classification</h4>
        <div class='big'>{pct(rf['accuracy'])}</div><div class='muted'>accuracy on 31 known classes (per flow)</div><br>
        Attack-family accuracy <b>{pct(rf['family_accuracy'])}</b>. Edge INT8 model:
        <b>{pct(mlp['int8']['accuracy'])}</b> accuracy at <b>{mlp['int8']['size_kb']:.0f} KB</b>
        ({mlp['size_reduction_pct']:.0f}% smaller than FP32).</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='card'><h4>3️⃣ Module 3 · Mitigation</h4>
        <div class='big'>8</div><div class='muted'>attack-family playbooks + 33 class-specific tips</div><br>
        Chooses an action, decides <b>auto vs. analyst approval</b>, applies it in a simulated
        controller, then <b>re-runs Module 1</b> to verify the traffic is normal again.</div>""",
                    unsafe_allow_html=True)

    st.subheader("How to demo this project")
    st.markdown("""
1. **Live Attack Lab**: pick an attack (e.g. *DDoS-SYN_Flood*), generate traffic, and walk through the
   four steps: capture, **Module 1** risk, **Module 2** class, **Module 3** mitigation + verification.
2. Try **BenignTraffic**: Module 1 says NORMAL and Modules 2–3 stay idle (this saves compute on the gateway).
3. Try a **Mirai** attack: it was never shown to Module 2. Module 1 still flags it, and Module 2's
   zero-day check reports *Unknown*, so Module 3 uses the containment playbook.
4. **Network Monitor**: run a whole stream to see the system working continuously.
5. **Model Performance**: the measured accuracy of every model.
""")
    st.info(f"Dataset: CICIoT2023, {M['dataset']['classes']} classes. Trained on "
            f"{M['dataset']['train_rows']:,} sampled flows, evaluated on {M['dataset']['test_rows']:,} "
            f"flows from the separate test file. Mirai family held out as a zero-day proxy.")

# =======================================================================================
# PAGE: LIVE ATTACK LAB
# =======================================================================================
elif page.startswith("🧪"):
    st.title("🧪 Live Attack Lab")
    st.caption("Generate one window of 20 real CICIoT2023 test flows and follow it through the pipeline.")

    fam_order = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
    options = []
    for f in fam_order:
        for c in sorted(c for c in P.classes if FAMILY_OF[c] == f):
            tag = "  ⚠️ zero-day (never trained)" if c in HELD_OUT_CLASSES else ""
            options.append(f"{f} · {c}{tag}")

    with st.container(border=True):
        a, b, c = st.columns([3, 2, 1.2])
        choice = a.selectbox("Traffic to inject", options, index=options.index("DDoS · DDoS-SYN_Flood"))
        share = b.slider("Attack share of the window", 0.5, 1.0, 0.8, 0.05,
                         help="The rest of the window is normal background traffic.")
        c.write(""); c.write("")
        go = c.button("▶ Generate & analyse", type="primary", use_container_width=True)
    cls = choice.split(" · ")[1].split("  ")[0]

    if go or "lab" not in st.session_state or st.session_state.lab["cls"] != cls:
        win = P.make_window(cls, share, seed=int(time.time() * 1000) % 10**9)
        st.session_state.lab = {"cls": cls, "res": P.run(win, model=engine_key, mitigate=False, m1_mode=m1_mode),
                                "done": False, "cfg": (engine_key, m1_mode)}
    lab = st.session_state.lab
    if lab["cfg"] != (engine_key, m1_mode):      # sidebar setting changed: re-analyse the same window
        lab["res"] = P.run(lab["res"].window, model=engine_key, mitigate=False, m1_mode=m1_mode)
        lab["done"], lab["cfg"] = False, (engine_key, m1_mode)
    res = lab["res"]

    # ---- STEP 1: capture ---------------------------------------------------------------
    with st.container(border=True):
        step_header(1, "Traffic capture: sliding window",
                    f"{WINDOW_SIZE} consecutive flows observed at the IoT gateway")
        w = res.window
        l, r = st.columns([1.1, 1])
        with l:
            show = w[["IAT", "Rate", "Protocol Type", "Tot size", "syn_flag_number", "label"]].copy()
            show.insert(0, "flow", range(1, len(w) + 1))
            show = show.rename(columns={"label": "ground truth (hidden from models)"})
            st.dataframe(show, hide_index=True, height=300, use_container_width=True)
        with r:
            dfc = pd.DataFrame({"flow": range(1, len(w) + 1), "IAT": w["IAT"].values,
                                "type": np.where(w["label"] == NORMAL_LABEL, "benign", "attack")})
            ch = alt.Chart(dfc).mark_bar().encode(
                x=alt.X("flow:O", title="flow #"), y=alt.Y("IAT:Q", title="Inter-arrival time", scale=alt.Scale(zero=False)),
                color=alt.Color("type:N", scale=alt.Scale(domain=["benign", "attack"], range=["#22c55e", "#ef4444"])),
                tooltip=["flow", "IAT", "type"]).properties(height=270, title="Inter-arrival time of each flow")
            st.altair_chart(ch, use_container_width=True)
        n_att = int((w["label"] != NORMAL_LABEL).sum())
        st.caption(f"Ground truth: {n_att} attack flows ({res.true_label}) + {len(w) - n_att} benign flows. "
                   "The models only see the numeric features, never the label.")

    # ---- STEP 2: Module 1 --------------------------------------------------------------
    m1r = res.m1
    with st.container(border=True):
        step_header(2, "MODULE 1: Anomaly detection (IAT + fuzzy logic)",
                    "Is this window anomalous? Uses only timing and rate; no attack labels needed.")
        f = m1r.features
        bl = P.m1.baseline
        k0, k1, k2, k3, k4 = st.columns(5)
        k0.metric("Std IAT  σ", f"{f['std_iat']:,.0f}")
        k1.metric("Mean IAT  μ", f"{f['mean_iat']:,.0f}")
        k2.metric("CV = σ / μ", f"{f['cv']:.3f}", help=f"Normal traffic CV ≈ {bl['cv_median']:.2f}. "
                  "Floods → CV near 0 (machine-regular timing). Scans → CV high (irregular).")
        k3.metric("Packet rate (median)", f"{f['rate']:,.0f} pkt/s", help=f"Normal median ≈ {bl['rate_median']:.0f} pkt/s")
        k4.metric("IAT deviation D", f"{f['dev']:.3f}", help="|μ − μ_normal| / μ_normal")
        if f.get("beh") is not None:
            st.markdown(f"**Behaviour deviation B = {f['beh']:.2f}** "
                        + ("(used)" if m1_mode == "full" else "(not used in timing-only mode)")
                        + ": distance of these flows to the nearest *normal* training flows. B > 1 means only 1% of "
                          "normal windows are this unusual. Learned from benign traffic only, so no attack labels are needed.")

        l, r = st.columns([1, 1.15])
        with l:
            st.markdown("**a) Fuzzification**: how strongly each input is LOW / MEDIUM / HIGH")
            mem = pd.DataFrame(m1r.memberships).T.rename(index={"cv": "CV", "rate": "Rate", "dev": "IAT dev", "beh": "Behaviour B"})
            mem = mem[["low", "medium", "high"]].round(3)
            st.dataframe(mem.style.background_gradient(cmap="Blues", vmin=0, vmax=1).format("{:.2f}"),
                         use_container_width=True)
            st.markdown("**b) Rules that fired** (strength = min of the IF-parts)")
            if m1r.fired_rules:
                fr = pd.DataFrame(m1r.fired_rules)[["id", "if", "output", "strength", "meaning"]]
                fr["output"] = "RISK " + fr["output"].str.upper()
                st.dataframe(fr.style.format({"strength": "{:.2f}"}), hide_index=True, use_container_width=True)
            else:
                st.write("No rule fired.")
        with r:
            st.markdown("**c) Aggregation + centroid defuzzification → attack risk**")
            agg = pd.DataFrame({"risk": RISK_UNIVERSE, "membership": m1r.aggregated})
            area = alt.Chart(agg).mark_area(opacity=.55, color="#6366f1").encode(
                x=alt.X("risk:Q", title="attack risk"), y=alt.Y("membership:Q", scale=alt.Scale(domain=[0, 1])))
            rule = alt.Chart(pd.DataFrame({"x": [m1r.risk]})).mark_rule(color="#0f172a", size=3).encode(x="x:Q")
            bands = alt.Chart(pd.DataFrame({"a": [0, .35, .65], "b": [.35, .65, 1], "z": ["normal", "suspicious", "attack"]})) \
                .mark_rect(opacity=.12).encode(x="a:Q", x2="b:Q", color=alt.Color("z:N", legend=None,
                scale=alt.Scale(domain=["normal", "suspicious", "attack"], range=["#22c55e", "#f59e0b", "#ef4444"])))
            st.altair_chart((bands + area + rule).properties(height=230), use_container_width=True)
            st.markdown(f"**Attack risk = {m1r.risk:.3f}**")
            gauge(m1r.risk)
        if m1r.verdict == "NORMAL":
            banner(f"✅ NORMAL (risk {m1r.risk:.2f} < {RISK_SUSPICIOUS}). Traffic is forwarded. "
                   "Module 2 and Module 3 stay idle, which saves CPU and energy on the IoT gateway.", "b-green")
        else:
            banner(f"{'🚨' if m1r.verdict == 'ATTACK' else '⚠️'} {m1r.verdict} (risk {m1r.risk:.2f} ≥ {RISK_SUSPICIOUS}). "
                   "Window handed to Module 2 for classification.", VERDICT_STYLE[m1r.verdict])

    # ---- STEP 3: Module 2 --------------------------------------------------------------
    if res.m2 is not None:
        m2r = res.m2
        with st.container(border=True):
            step_header(3, "MODULE 2: Attack classification",
                        f"Which attack is it? Engine: {m2r.model}. Each flow is classified, then the window is decided by majority vote.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Predicted attack", m2r.attack_class)
            k2.metric("Attack family", m2r.family)
            k3.metric("Confidence", pct(m2r.confidence))
            k4.metric("Inference time (20 flows)", f"{m2r.latency_ms:.2f} ms")
            l, r = st.columns([1.2, 1])
            with l:
                t5 = pd.DataFrame(m2r.top5, columns=["class", "probability"])
                st.altair_chart(alt.Chart(t5).mark_bar(color="#f97316").encode(
                    x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("class:N", sort="-x", title=None),
                    tooltip=["class", alt.Tooltip("probability:Q", format=".3f")]).properties(
                    height=200, title="Top-5 class probabilities"), use_container_width=True)
                votes = ", ".join(f"{k}: {v}" for k, v in sorted(m2r.flow_votes.items(), key=lambda x: -x[1]))
                st.caption(f"Per-flow votes → {votes}")
            with r:
                st.markdown("**Zero-day check** (open-set detection)")
                st.markdown("Distance of the attack flows to the nearest known training traffic, "
                            "divided by the calibrated threshold. **> 1.0 means unlike anything seen in training.**")
                zd = m2r.zero_day_score
                st.progress(min(zd / 3, 1.0), text=f"zero-day score = {zd:.2f}  (threshold 1.00)")
                if m2r.is_zero_day:
                    banner("🧬 UNKNOWN: does not match any known attack. Treated as a possible zero-day.", "b-red")
                else:
                    banner("✔ Matches known attack traffic.", "b-blue")
            ok_family = (m2r.family == FAMILY_OF.get(res.true_label)) or (m2r.is_zero_day and res.true_label in HELD_OUT_CLASSES)
            st.caption(f"Ground truth: **{res.true_label}** → prediction "
                       f"{'✅ correct' if (m2r.attack_class == res.true_label or ok_family) else '❌ wrong'}"
                       f"{' (correct family)' if ok_family and m2r.attack_class != res.true_label and not m2r.is_zero_day else ''}"
                       f"{' (correctly flagged as unseen)' if m2r.is_zero_day and res.true_label in HELD_OUT_CLASSES else ''}")
            if m2r.family == "Benign":
                banner("✅ Module 2 finds only benign traffic, so Module 1's alert was a false alarm. "
                       "No mitigation is needed.", "b-green")

    # ---- STEP 4: Module 3 --------------------------------------------------------------
    if res.m3 is not None:
        plan = res.m3
        with st.container(border=True):
            step_header(4, "MODULE 3: Mitigation & suggestions",
                        "What should we do? Pick a playbook, apply it (simulated controller), and verify by re-running Module 1.")
            l, r = st.columns([1.3, 1])
            with l:
                st.markdown(f"<span class='pill' style='background:{SEV_COLOR[plan.severity]}'>{plan.severity}</span> "
                            f"&nbsp;<b>{plan.family}</b>: {FAMILY_DESCRIPTION.get(plan.family, '')}",
                            unsafe_allow_html=True)
                st.markdown(f"**Why it matters:** {plan.why}")
                if plan.class_tip:
                    st.markdown(f"**Specific to {plan.attack_class}:** {plan.class_tip}")
            with r:
                if plan.auto_execute:
                    banner(f"⚡ Automatic response allowed: {plan.approval_reason}.", "b-blue")
                else:
                    banner(f"👤 Needs analyst approval: {plan.approval_reason}.", "b-amber")

            st.markdown("**Response plan (escalation ladder)**")
            for i, a in enumerate(plan.actions, 1):
                with st.expander(f"Stage {i}: {a.name}  ·  expected to stop ~{a.efficacy:.0%} of attack flows"
                                 f"{'  ·  ⚠ disruptive' if a.disruptive else ''}", expanded=(i == 1)):
                    st.write(a.description)
                    st.code(a.command, language="bash")

            label = "▶ Apply mitigation automatically" if plan.auto_execute else "👤 Approve & apply mitigation"
            if not lab["done"]:
                if st.button(label, type="primary"):
                    lab["res"].m3 = P.m3.execute_and_verify(plan, res.window, res.m2.attack_flow_mask)
                    lab["done"] = True
                    st.rerun()
            if lab["done"]:
                st.markdown("**Verification: Module 1 is re-run on the traffic after each action**")
                srows = [{"stage": s.stage, "action": s.action.name,
                          "attack flows": f"{s.attack_flows_before} → {s.attack_flows_after}",
                          "risk": f"{s.risk_before:.2f} → {s.risk_after:.2f}",
                          "CV": f"{s.cv_before:.2f} → {s.cv_after:.2f}",
                          "packet rate": f"{s.rate_before:,.0f} → {s.rate_after:,.0f}",
                          "result": "✅ resolved" if s.resolved else "↗ escalate"} for s in plan.steps]
                st.dataframe(pd.DataFrame(srows), hide_index=True, use_container_width=True)
                pts = [{"point": "before", "risk": plan.steps[0].risk_before}] + \
                      [{"point": f"after stage {s.stage}", "risk": s.risk_after} for s in plan.steps]
                pdf = pd.DataFrame(pts)
                line = alt.Chart(pdf).mark_line(point=alt.OverlayMarkDef(size=90), color="#6366f1").encode(
                    x=alt.X("point:N", sort=None, title=None), y=alt.Y("risk:Q", title="attack risk", scale=alt.Scale(domain=[0, 1])))
                thr = alt.Chart(pd.DataFrame({"y": [RISK_SUSPICIOUS]})).mark_rule(strokeDash=[6, 4], color="#16a34a").encode(y="y:Q")
                st.altair_chart((line + thr).properties(height=220, title="Attack risk before / after mitigation"),
                                use_container_width=True)
                last = plan.steps[-1]
                if plan.resolved:
                    banner(f"✅ Mitigation verified: risk fell from {plan.steps[0].risk_before:.2f} to "
                           f"{last.risk_after:.2f} and CV returned to {last.cv_after:.2f} (normal ≈ {P.m1.baseline['cv_median']:.2f}). "
                           "Traffic is back to normal.", "b-green")
                else:
                    banner(f"⚠️ Risk is still {last.risk_after:.2f} after all automated stages. "
                           "Escalate to the security team.", "b-amber")
                if st.button("↺ Reset mitigation"):
                    lab["res"] = P.run(res.window, model=engine_key, mitigate=False, m1_mode=m1_mode)
                    lab["done"] = False
                    st.rerun()

            st.markdown("**Long-term recommendations**")
            for rec in plan.recommendations:
                st.markdown(f"- {rec}")
            st.caption("The network controller is simulated: actions are applied to the captured window, "
                       "and the commands show what would run on a Linux IoT gateway.")

# =======================================================================================
# PAGE: NETWORK MONITOR
# =======================================================================================
elif page.startswith("📡"):
    st.title("📡 Network Monitor")
    st.caption("A continuous stream of traffic windows. Every window passes Module 1; only flagged "
               "windows reach Modules 2 and 3.")
    presets = {
        "Mixed day (recommended)": [(NORMAL_LABEL, 8), ("DDoS-SYN_Flood", 4), (NORMAL_LABEL, 4),
                                    ("Recon-PortScan", 4), (NORMAL_LABEL, 4), ("Mirai-udpplain", 4),
                                    (NORMAL_LABEL, 4), ("DoS-UDP_Flood", 4), (NORMAL_LABEL, 3)],
        "Flood attacks": [(NORMAL_LABEL, 5), ("DDoS-ICMP_Flood", 5), ("DoS-TCP_Flood", 5),
                          ("DDoS-UDP_Fragmentation", 5), (NORMAL_LABEL, 5)],
        "Zero-day Mirai": [(NORMAL_LABEL, 6), ("Mirai-greeth_flood", 6), ("Mirai-greip_flood", 6), (NORMAL_LABEL, 4)],
        "Stealthy & spoofing": [(NORMAL_LABEL, 5), ("MITM-ArpSpoofing", 5), ("DNS_Spoofing", 5),
                                ("Recon-OSScan", 5), (NORMAL_LABEL, 5)],
    }
    a, b, c = st.columns([2, 1.5, 1])
    preset = a.selectbox("Scenario", list(presets))
    share = b.slider("Attack share per window", 0.5, 1.0, 0.8, 0.05)
    c.write(""); c.write("")
    run = c.button("▶ Run stream", type="primary", use_container_width=True)
    st.caption("Phases: " + "  →  ".join(f"{cls} ×{n}" for cls, n in presets[preset]))

    if run:
        wins = P.stream(presets[preset], share, seed=int(time.time()) % 10**6)
        rows, bar = [], st.progress(0.0, text="Processing windows...")
        for i, w in enumerate(wins):
            r = P.run(w, model=engine_key, mitigate=True, m1_mode=m1_mode)
            rows.append({
                "window": i + 1, "ground truth": r.true_label, "M1 risk": r.m1.risk, "M1 verdict": r.m1.verdict,
                "M2 invoked": r.m2 is not None,
                "M2 class": r.m2.attack_class if r.m2 else "—",
                "M2 family": r.m2.family if r.m2 else "—",
                "M3 action": r.m3.steps[-1].action.name if (r.m3 and r.m3.steps) else "—",
                "risk after M3": r.m3.steps[-1].risk_after if (r.m3 and r.m3.steps) else r.m1.risk,
                "resolved": (r.m3.resolved if r.m3 else None),
                "approval": ("auto" if r.m3.auto_execute else "analyst") if r.m3 else "—",
            })
            bar.progress((i + 1) / len(wins), text=f"Window {i + 1}/{len(wins)}")
        bar.empty()
        st.session_state.monitor = pd.DataFrame(rows)

    if "monitor" in st.session_state:
        df = st.session_state.monitor
        att = df["ground truth"] != NORMAL_LABEL
        flagged = df["M1 verdict"] != "NORMAL"
        k = st.columns(5)
        k[0].metric("Windows", len(df))
        k[1].metric("Attack windows detected (M1)", f"{(flagged & att).sum()} / {att.sum()}")
        k[2].metric("False alarms (M1)", f"{(flagged & ~att).sum()} / {(~att).sum()}")
        k[3].metric("Module 2 skipped", pct(1 - df["M2 invoked"].mean()), help="Windows Module 2 never had to process")
        mit = df[df["resolved"].notna()]
        k[4].metric("Mitigations verified", f"{int(mit['resolved'].sum())} / {len(mit)}")

        long = pd.concat([
            df[["window", "M1 risk"]].rename(columns={"M1 risk": "risk"}).assign(series="risk (Module 1)"),
            df[["window", "risk after M3"]].rename(columns={"risk after M3": "risk"}).assign(series="risk after mitigation"),
        ])
        bg = alt.Chart(df).mark_rect(opacity=.12).encode(
            x=alt.X("window:Q"), x2=alt.X2("w2:Q"),
            color=alt.Color("phase:N", title="ground truth", scale=alt.Scale(scheme="tableau10"))
        ).transform_calculate(w2="datum.window + 1", phase="datum['ground truth']")
        lines = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("window:Q", title="window #"), y=alt.Y("risk:Q", title="attack risk", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("series:N", scale=alt.Scale(range=["#ef4444", "#16a34a"]), title=None),
            tooltip=["window", "series", alt.Tooltip("risk:Q", format=".2f")])
        thr = alt.Chart(pd.DataFrame({"y": [RISK_SUSPICIOUS, RISK_ATTACK]})).mark_rule(strokeDash=[5, 4], color="#64748b").encode(y="y:Q")
        st.altair_chart((bg + lines + thr).properties(height=320, title="Attack risk over time"),
                        use_container_width=True)

        st.markdown("**Event log**")
        st.dataframe(df.style.format({"M1 risk": "{:.2f}", "risk after M3": "{:.2f}"}),
                     hide_index=True, use_container_width=True, height=380)

# =======================================================================================
# PAGE: MODEL PERFORMANCE
# =======================================================================================
elif page.startswith("📊"):
    st.title("📊 Model Performance")
    st.caption("Measured on the separate CICIoT2023 test file. The Mirai family was never used for training.")
    m1, m2 = M["module1"], M["module2"]
    rf, mlp, we = m2["random_forest"], m2["mlp"], m2["window_eval"]
    t1, t2, t3, t4 = st.tabs(["Module 1 · Detection", "Module 2 · Classification",
                              "Zero-day (Mirai)", "Earlier experiments"])

    with t1:
        st.markdown(f"Evaluated on **{m1['windows']} windows** (40 per class, 20 flows each, from the test file).")
        k = st.columns(5)
        for col, (n, v) in zip(k, [("Recall (attacks caught)", m1["recall"]), ("Precision", m1["precision"]),
                                   ("F1-score", m1["f1"]), ("Balanced accuracy", m1["balanced_accuracy"]),
                                   ("False-positive rate", m1["fpr"])]):
            col.metric(n, pct(v))
        tm = m1["timing_only"]
        cmp_df = pd.DataFrame([
            {"Module 1 version": "IAT timing only (CV, rate, D)", "recall": tm["recall"], "precision": tm["precision"],
             "F1": tm["f1"], "balanced acc.": tm["balanced_accuracy"], "FPR": tm["fpr"],
             "Mirai (zero-day) caught": tm["mirai_zero_day_detection"]},
            {"Module 1 version": "IAT timing + behaviour B (final)", "recall": m1["recall"], "precision": m1["precision"],
             "F1": m1["f1"], "balanced acc.": m1["balanced_accuracy"], "FPR": m1["fpr"],
             "Mirai (zero-day) caught": m1["mirai_zero_day_detection"]},
        ])
        st.markdown("**Improvement from adding the behaviour input**")
        st.dataframe(cmp_df.style.format({c: "{:.1%}" for c in cmp_df.columns[1:]}), hide_index=True,
                     use_container_width=True)
        l, r = st.columns(2)
        with l:
            cm = m1["confusion"]
            st.markdown("**Confusion matrix (windows)**")
            st.table(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                                  index=["actual normal", "actual attack"], columns=["predicted normal", "predicted attack"]))
            st.markdown("**Detection rate per attack family**")
            st.dataframe(pd.Series(m1["detection_rate_by_family"], name="flagged").sort_values().map(pct),
                         use_container_width=True)
        with r:
            st.image(str(METRICS_DIR / "m1_flag_rate_by_family.png"))
            st.info("Module 1 is unsupervised: it only knows what *normal* timing looks like. Benign windows that "
                    "get flagged are not dropped; Module 2 double-checks them, which is why the two-stage design "
                    "tolerates Module 1's false alarms.")
        with st.expander("Detection rate per class"):
            st.dataframe(pd.Series(m1["detection_rate_by_class"], name="flagged").sort_values().map(pct),
                         use_container_width=True)

    with t2:
        st.markdown("**Flow-level results on 31 known classes**")
        tbl = pd.DataFrame([
            {"model": "Random Forest (FP32)", "accuracy": rf["accuracy"], "macro-F1": rf["macro_f1"],
             "family accuracy": rf["family_accuracy"], "size": f"{rf['size_mb']:.1f} MB",
             "latency / flow": f"{rf['latency_ms_per_flow'] * 1000:.1f} µs"},
            {"model": "MLP (FP32)", "accuracy": mlp["fp32"]["accuracy"], "macro-F1": mlp["fp32"]["macro_f1"],
             "family accuracy": mlp["fp32"]["family_accuracy"], "size": f"{mlp['fp32']['size_kb']:.0f} KB",
             "latency / flow": "—"},
            {"model": "MLP (INT8, edge)", "accuracy": mlp["int8"]["accuracy"], "macro-F1": mlp["int8"]["macro_f1"],
             "family accuracy": mlp["int8"]["family_accuracy"], "size": f"{mlp['int8']['size_kb']:.0f} KB",
             "latency / flow": f"{mlp['int8']['latency_ms_per_flow'] * 1000:.1f} µs"},
        ])
        st.dataframe(tbl.style.format({c: "{:.2%}" for c in ["accuracy", "macro-F1", "family accuracy"]}),
                     hide_index=True, use_container_width=True)
        k = st.columns(3)
        k[0].metric("INT8 size reduction", f"{mlp['size_reduction_pct']:.0f}%")
        k[1].metric("FP32 ↔ INT8 agreement", pct(mlp["agreement_fp32_int8"]))
        k[2].metric("Window accuracy (RF, 70% attack + 30% benign)", pct(we["known_class_accuracy"]),
                    help=f"Family accuracy at window level: {pct(we['known_family_accuracy'])}")
        l, r = st.columns(2)
        l.image(str(METRICS_DIR / "m2_family_confusion.png"))
        r.image(str(METRICS_DIR / "m2_per_class_f1.png"))
        st.info("Classes such as DDoS vs. DoS of the same protocol, or Recon-OSScan vs. Recon-PortScan, look "
                "alike at flow level, so most remaining errors fall *inside* the right family. That is why the "
                "family accuracy, which decides the Module 3 playbook, is much higher than the 31-class accuracy.")
        with st.expander("Per-class precision / recall / F1"):
            rep = pd.DataFrame(m2["per_class_report"]).T
            st.dataframe(rep.style.format("{:.3f}"), use_container_width=True)

    with t3:
        st.markdown("The three **Mirai** classes were removed from Module 2's training data and only appear at test time.")
        k = st.columns(4)
        k[0].metric("Module 1 flags Mirai windows", pct(m1["mirai_zero_day_detection"]))
        k[1].metric("Module 2 says 'Unknown'", pct(we["mirai_flagged_unknown"]))
        k[2].metric("Known windows wrongly 'Unknown'", pct(we["known_false_unknown_rate"]))
        k[3].metric("Zero-day threshold", f"{m2['zero_day_threshold']:.2f}")
        if we["mirai_assigned_to"]:
            st.markdown("When Module 2 did **not** flag Mirai as unknown, it assigned it to:")
            st.dataframe(pd.Series(we["mirai_assigned_to"], name="windows"), use_container_width=True)
        st.info("Module 1 does not depend on labels, so it detects the unseen family. Module 2's nearest-neighbour "
                "check catches Mirai variants that look unlike all known traffic. Variants that resemble a known flood "
                "are mapped to that flood's playbook, which is still a sensible response.")

    with t4:
        st.markdown("Results of the earlier version of the project (paper draft), kept for comparison.")
        import json as _json
        from pathlib import Path
        old = Path("results/metrics/summary_metrics.json")
        old2 = Path("results/metrics/module2_metrics.json")
        if old.exists():
            mc = pd.DataFrame(_json.loads(old.read_text())["method_comparison"])
            st.markdown("**Module 1: fuzzy vs. threshold baselines** (phased stream of 4,999 windows)")
            st.dataframe(mc[["method", "accuracy", "precision", "recall_tpr", "f1_score", "false_positive_rate",
                             "zero_day_detection_rate"]], hide_index=True, use_container_width=True)
        if old2.exists():
            st.markdown("**Earlier Module 2: LSTM sequence models (small training budget)**")
            st.dataframe(pd.DataFrame(_json.loads(old2.read_text())["model_comparison"]), hide_index=True,
                         use_container_width=True)
            st.caption("The LSTM was replaced because, with the available sample budget, it reached only ~34% accuracy. "
                       "The per-flow Random Forest and INT8 MLP are both far more accurate and lighter.")
        for img in ["09_confusion_matrices.png", "10_performance_comparison.png"]:
            pth = Path("results/plots") / img
            if pth.exists():
                st.image(str(pth))

# =======================================================================================
# PAGE: HOW IT WORKS
# =======================================================================================
else:
    st.title("📘 How It Works: step by step")
    bl = P.m1.baseline
    st.markdown("### Step 0 · Data")
    st.markdown(f"""
- **CICIoT2023**: real traffic from 105 IoT devices; 34 classes (benign + 33 attacks in 7 families).
- The CSV gives one row per *flow* with 46 numeric features (rate, flags, sizes, protocol, **IAT**, ...).
- `ids/prepare_data.py` samples a balanced subset: **{M['dataset']['train_rows']:,}** training flows from
  `train.csv` and **{M['dataset']['test_rows']:,}** test flows from the separate `test.csv`.
- **Zero-day proxy:** all 3 Mirai classes are removed from Module 2 training.
""")
    st.markdown("### Step 1 · Module 1: anomaly detection")
    st.markdown(f"A sliding window of **W = {WINDOW_SIZE}** flows. From their inter-arrival times (IAT):")
    st.latex(r"\mu=\frac1W\sum IAT_i,\quad \sigma=\sqrt{\frac1W\sum(IAT_i-\mu)^2},\quad CV=\frac{\sigma}{\mu},\quad D=\frac{|\mu-\mu_{normal}|}{\mu_{normal}}")
    st.markdown(f"""
- **Why CV?** Normal IoT traffic has natural jitter (CV ≈ {bl['cv_median']:.2f}). Flood tools send packets
  like a clock, so **CV → 0**. Scanners probe irregularly, so **CV is high**. A plain threshold `CV > T`
  misses floods, which is why the earlier threshold baselines scored ~0% recall.
- **Behaviour deviation B** (added in the final version): the median distance of the window's flows to the
  nearest *benign* training flows, divided by a threshold (B > 1 means only 1% of normal windows are this unusual).
  It needs no attack labels, so Module 1 stays unsupervised. Timing catches floods; B catches attacks whose
  timing looks normal (recon, spoofing, web, brute force). The packet rate is the window **median**, so one
  extreme flow cannot trigger an alarm on its own.
- **Fuzzy logic** combines CV, packet rate, D and B. Each input is given LOW/MEDIUM/HIGH membership degrees,
  **{len(RULES)} IF-THEN rules** fire (AND = min), outputs are clipped and merged (max), and the
  **centroid** gives one risk score in [0, 1].
""")
    st.dataframe(pd.DataFrame([{"rule": r[0], "IF": " AND ".join(f"{v.upper()} is {t.upper()}" for v, t in r[1].items()),
                                "THEN risk": r[2].upper(), "meaning": r[3]} for r in RULES]),
                 hide_index=True, use_container_width=True)
    st.markdown(f"- Decision: risk < {RISK_SUSPICIOUS} **NORMAL** · {RISK_SUSPICIOUS}–{RISK_ATTACK} **SUSPICIOUS** · "
                f"≥ {RISK_ATTACK} **ATTACK**. Suspicious and attack windows go to Module 2.")

    st.markdown("### Step 2 · Module 2: attack classification")
    st.markdown("""
- **Random Forest** (100 trees) on all 46 flow features. It classifies every flow in the window, and the
  window gets the majority attack label. Background benign flows are ignored if attacks are ≥ 30% of the window.
- **INT8 edge model**: an MLP (46→128→64→31). After training, weights and activations are quantised to 8-bit:
""")
    st.latex(r"s=\frac{r_{max}-r_{min}}{255},\qquad z=\mathrm{round}\!\left(-128-\frac{r_{min}}{s}\right),\qquad q=\mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{r}{s}\right)+z,\,-128,\,127\right)")
    st.markdown("""
  Matrix products then run in integer arithmetic (int8 × int8 → int32), as on a microcontroller.
  The model becomes about 4× smaller with almost the same accuracy.
- **Zero-day check**: for the attack flows, the distance to the nearest *known* training flow (standardised
  log features). Its median is divided by a threshold calibrated on validation data (97.5th percentile).
  A score > 1 means "unlike anything known", so the window is labelled **Unknown (zero-day)**.
""")
    st.markdown("### Step 3 · Module 3: mitigation and verification")
    st.markdown("""
1. **Playbook selection** by attack family (DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce, Unknown),
   plus a class-specific tip (e.g. SYN flood → SYN cookies).
2. **Auto vs. approval**: automatic only if confidence ≥ 60%, the attack is known, and the action is not
   disruptive. Otherwise an analyst must approve.
3. **Apply** (simulated controller): flows that Module 2 marked as attacks are filtered with the action's
   efficacy, while legitimate traffic keeps flowing.
4. **Verify**: Module 1 re-computes the risk on the new traffic. If it is < 0.35 the attack is resolved
   (CV is back near normal). If not, escalate to the next, stronger stage.
""")
    st.markdown("### Why two stages?")
    st.markdown("""
- **Efficiency**: on a real network most windows are normal, and only Module 1's cheap statistics run on them.
- **Zero-day coverage**: a supervised classifier always picks one of its known classes. Module 1 does not
  rely on labels, and the zero-day check stops Module 2 from forcing a known label onto new traffic.
- **Explainability**: every decision can be traced: fired fuzzy rules, class probabilities, playbook, verification.
""")
