import py_compile
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / (sys.argv[1] if len(sys.argv) > 1 else "app.py")
failures = []


def sub_once(src, old, new, name):
    if src.count(old) != 1:
        failures.append(f"{name} (found {src.count(old)} matches, expected 1)")
        return src
    return src.replace(old, new)


def rsub_once(src, pattern, repl, name, flags=re.S):
    matches = list(re.finditer(pattern, src, flags))
    if len(matches) != 1:
        failures.append(f"{name} (found {len(matches)} matches, expected 1)")
        return src
    m = matches[0]
    return src[:m.start()] + repl(m) + src[m.end():]


NEW_RUN_ANALYSIS = r'''def make_agent_specs(case: dict, lang: str):
    """Independent specialist agents that run in parallel; the master agent then reconciles them."""
    idx = case["status"]
    patient = (f"Age: {case['age']} years\nStatus profile: {T['English']['statuses'][idx]}\n"
               f"History: {clip(case['history'])}\nRecent operations: {clip(case['ops'])}\n"
               f"Regular medications: {clip(case['meds'])}\nFood, drinks & supplements: {clip(case['diet'])}")
    specs = [
        ("Age specialist", AGE_AGENTS[age_bracket(case["age"])], False),
        ("Status specialist", STATUS_AGENTS[idx], False),
        ("Diet & herbal analyst", "Cross-reference every listed food, drink, herb and supplement "
                                  "against the medicines and conditions. Read any image for medicines.", True),
    ]
    if case.get("images"):
        specs.insert(0, ("Prescription vision analyst", "Read the attached image(s). Extract every medicine, "
                         "strength, form, frequency and duration. Mark anything unclear.", True))
    return specs, up.scrub(patient)


def run_analysis(case: dict) -> bool:
    """Parallel agents -> streamed master synthesis -> structured JSON -> rules -> score -> PDF."""
    s = st.session_state
    lang = s.lang
    key = get_api_key()
    if not key:
        st.error(tr("need_key"))
        return False
    case = {k: v for k, v in case.items() if k != "agent_reports"}

    cache = s.setdefault("result_cache", {})
    fp = case_fingerprint(case, lang)
    cached = cache.get(fp)
    trace = None
    if cached:
        report, struct, *rest = cached
        trace = rest[0] if rest else None
    else:
        try:
            get_client(key)  # create the cached client in the main thread before using worker threads
            specs, patient = make_agent_specs(case, lang)
            with st.spinner("Specialist agents are working in parallel..."):
                reports, trace = up.run_parallel_agents(
                    generate_text, key, MODEL_NAME, specs, patient, LANG_NAMES[lang], case.get("images") or [])
            case = {**case, "agent_reports": reports}
            with st.status(tr("running"), expanded=True) as status:
                report = normalize_tags(str(st.write_stream(stream_gemini(key, case, lang)) or "").strip())
                if not report:
                    raise RuntimeError("The model returned an empty or blocked response.")
                status.update(label="✅ " + tr("s3_title"), state="complete", expanded=False)
        except Exception as exc:
            st.error(tr("err_ai"))
            with st.expander("Technical detail"):
                st.code(f"Model: {MODEL_NAME}\n{type(exc).__name__}: {exc}")
            return False
        struct = None

    if struct is None:  # first run, or an earlier extraction failed: try again
        try:
            with st.spinner("…"):
                struct = extract_structured(key, report, lang)
        except Exception:
            struct = None  # the full report still works; the UI shows a notice
    cache[fp] = (report, struct, trace)

    s.case, s.report, s.report_lang, s.chat, s.care, s.trace = case, report, lang, [], None, trace
    for k in [k for k in s.keys() if str(k).startswith("dose_")]:
        del s[k]  # new report: reset the dose tracker
    rules = run_rule_engine(case, struct)
    score = safety_score(final_counts(report, struct), rules)
    s.struct, s.rules, s.score = struct, rules, score
    try:
        s.pdf_bytes, s.pdf_fonts_ok = build_pdf(case, report, lang, {"score": score, "rules": rules, "struct": struct})
        s.pdf_error = None
    except Exception as exc:
        s.pdf_bytes, s.pdf_error = None, f"{type(exc).__name__}: {exc}"
    save_history(case, report, score)
    up.audit("analysis", (s.user or {}).get("username", "guest"), f"fp={fp[:12]}")
    return True


'''

FINAL_COUNTS = r'''def final_counts(report: str, struct: dict | None) -> dict:
    """Source of truth = the structured JSON. Regex on prose is only a fallback."""
    if struct and struct["interactions"]:
        c = {k: 0 for k in SEV_ORDER}
        for i in struct["interactions"]:
            c[i["severity"]] += 1
        return c
    return severity_counts(report)


'''

DELETE_MY_DATA = r'''def delete_my_data() -> None:
    """Privacy: remove the signed-in user's saved reports (DPDP-style right to erasure)."""
    user = st.session_state.user or {}
    if user.get("guest") or not user.get("username"):
        return
    with _HIST_LOCK:
        db = _load_hist()
        db.pop(user["username"], None)
        up.secure_write(HISTORY_DB, json.dumps(db, ensure_ascii=False))
    up.audit("data_deleted", user["username"])


'''

TAB_AG = r'''    with tab_ag:
        st.markdown("**Multi-agent trace** (agents ran in parallel threads)")
        up.render_trace(st.session_state.get("trace"))
        st.divider()
        st.markdown("**HL7 FHIR R4 export**")
        bundle = up.build_fhir_bundle(case, struct, rules)
        problems = up.fhir_check(bundle)
        st.caption("Structure check: " + ("passed" if not problems else "; ".join(problems[:3])))
        st.download_button("Download FHIR bundle (.json)", json.dumps(bundle, indent=2, ensure_ascii=False),
                           file_name="safescript_fhir_bundle.json", mime="application/fhir+json")
        with st.expander("Preview bundle"):
            st.json(bundle)
        audit_ok, audit_n = up.verify_audit()
        st.caption(f"Audit log: {audit_n} entries, hash chain {'valid ✅' if audit_ok else 'BROKEN ❌'}")
'''


def main():
    if not APP.exists():
        sys.exit(f"Cannot find {APP}. Put this script next to your app.py.")
    if not (HERE / "upgrades.py").exists():
        sys.exit("upgrades.py must be in the same folder.")
    src = APP.read_text(encoding="utf-8")
    if "import upgrades as up" in src:
        sys.exit("This app.py is already upgraded. Nothing to do.")

    src = sub_once(src, "from PIL import Image, ImageOps\n",
                   "from PIL import Image, ImageOps\n\nimport upgrades as up  # parallel agents, FHIR, evaluation, security\n", "import")

    src = sub_once(src, "- Name: {clip(case['name'])}", "- Name: withheld for privacy", "prompt name")
    src = sub_once(src, "Produce the full structured report now.",
                   "{up.agent_reports_block(case)}\nProduce the full structured report now.", "prompt agent block")

    src = sub_once(src, "contents: list = [build_user_prompt(case, lang)]",
                   "contents: list = [up.scrub(build_user_prompt(case, lang))]", "stream scrub")
    src = sub_once(src, "prompt = (f\"PATIENT: {case['name']}, {case['age']} years.",
                   "prompt = (f\"PATIENT: {case['age']} years.", "followup name")
    src = sub_once(src, "generate_text(api_key, STRUCT_MODEL, prompt, system_instruction=system,",
                   "generate_text(api_key, STRUCT_MODEL, up.scrub(prompt), system_instruction=system,", "followup scrub")

    src = sub_once(src, 'SPEECH_LANG = {"English": "en-IN"', FINAL_COUNTS + 'SPEECH_LANG = {"English": "en-IN"', "final_counts")
    src = sub_once(src, 'counts = severity_counts(report)\n    pdf.use("B", 11)',
                   'counts = final_counts(report, (extras or {}).get("struct"))\n    pdf.use("B", 11)', "pdf counts")
    src = rsub_once(
        src,
        r"counts = severity_counts\(report\)\n(\s+)score = st\.session_state\.score if st\.session_state\.score is not None else 100\n\s+struct = st\.session_state\.struct",
        lambda m: ("struct = st.session_state.struct\n" + m.group(1) + "counts = final_counts(report, struct)\n"
                   + m.group(1) + "score = st.session_state.score if st.session_state.score is not None else 100"),
        "step3 counts")

    start = src.find("def run_analysis(case: dict) -> bool:")
    end = src.find("def gauge_html(score: int) -> str:")
    if start == -1 or end == -1 or end < start:
        failures.append("run_analysis block not found")
    else:
        src = src[:start] + NEW_RUN_ANALYSIS + src[end:]
    src = sub_once(src, '    "mode": "full",\n}', '    "mode": "full",\n    "trace": None,\n}', "default trace")

    src = sub_once(src, 'return json.loads(HISTORY_DB.read_text(encoding="utf-8"))',
                   "return json.loads(up.secure_read(HISTORY_DB))", "load history")
    src = rsub_once(
        src, r'tmp = HISTORY_DB\.with_suffix\("\.tmp"\)\n\s*tmp\.write_text\(.*?\)\n\s*os\.replace\(tmp, HISTORY_DB\)',
        lambda m: "up.secure_write(HISTORY_DB, json.dumps(db, ensure_ascii=False, indent=1))", "save history")
    src = sub_once(src, "def avatar(greeting: str", DELETE_MY_DATA + "def avatar(greeting: str", "delete_my_data")
    src = sub_once(src, '        st.button(tr("logout"), on_click=sign_out)',
                   '        st.button(tr("logout"), on_click=sign_out)\n        st.button("🗑️ Delete my saved data", on_click=delete_my_data)',
                   "sidebar delete")

    src = sub_once(src, "st.session_state.user = user\n                        st.session_state.login_fails = 0",
                   'st.session_state.user = user\n                        up.audit("login", user["username"])\n                        st.session_state.login_fails = 0',
                   "audit login")
    src = sub_once(src, "st.session_state.login_fails += 1",
                   'st.session_state.login_fails += 1\n                    up.audit("login_failed", u.strip()[:30])', "audit login fail")
    src = sub_once(src, 'st.session_state.user = {"name": disp.strip() or uname',
                   'up.audit("signup", uname.lower())\n                    st.session_state.user = {"name": disp.strip() or uname', "audit signup")
    src = sub_once(src, "def sign_out() -> None:\n",
                   'def sign_out() -> None:\n    up.audit("logout", (st.session_state.user or {}).get("username", "guest"))\n', "audit logout")

    src = sub_once(src, "    b1, b2, _ = st.columns([1, 2, 5])",
                   '    consent = st.checkbox(\n'
                   '        "I consent to this data being processed by Google Gemini for this analysis (DPDP Act 2023). "\n'
                   '        "My name is not sent, and phone, email and ID numbers are masked.", key="consent")\n'
                   "    b1, b2, _ = st.columns([1, 2, 5])", "consent box")
    src = sub_once(src, 'if b2.button(tr("analyze"), type="primary"):',
                   'if b2.button(tr("analyze"), type="primary", disabled=not consent):', "consent button")

    src = sub_once(src, "tab_r, tab_m, tab_s, tab_k, tab_v, tab_care, tab_c, tab_id, tab_h = st.tabs(",
                   "tab_r, tab_m, tab_s, tab_k, tab_v, tab_care, tab_c, tab_id, tab_h, tab_ag = st.tabs(", "tabs unpack")
    src = rsub_once(src, r'tr\("tab_id"\), "[^"]*History"\]',
                    lambda m: 'tr("tab_id"), "🗂️ History", "🧭 Agents & FHIR"]', "tabs labels")
    src = sub_once(src, "    with tab_h:\n        history_tab()\n",
                   "    with tab_h:\n        history_tab()\n\n" + TAB_AG, "tab_ag body")

    src = sub_once(src, '(expected: {expected})")\n',
                   '(expected: {expected})")\n\n    with st.expander("Evaluation benchmark: precision and recall"):\n        up.render_eval(run_rule_engine)\n',
                   "eval expander")

    if failures:
        print("UPGRADE NOT APPLIED. These anchors were not found in your app.py:")
        for f in failures:
            print("  -", f)
        print("\nYour app.py was not changed. Send me this list and I will fix the script.")
        sys.exit(1)

    tmp = HERE / "_app_upgraded_tmp.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        py_compile.compile(str(tmp), doraise=True)
    except py_compile.PyCompileError as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(f"UPGRADE NOT APPLIED: the result had a syntax error.\n{exc}")
    shutil.copy(APP, HERE / "app_backup.py")
    tmp.replace(APP)
    print("Done. app.py is upgraded (original saved as app_backup.py).")
    print("Next: pip install cryptography   then   streamlit run app.py")


if __name__ == "__main__":
    main()