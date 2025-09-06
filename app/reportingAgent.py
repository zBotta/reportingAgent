
"""
reportAgent.py

web app in streamlit to run the reporting agent

"""
import sys
from conf.logManager import Logger
import logging

logging.setLoggerClass(Logger)
log = logging.getLogger(__name__)


from projectSetup import Setup
import streamlit as st
# PDF
from io import BytesIO
from xml.sax.saxutils import escape
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem,
    Preformatted, HRFlowable
)


env = Setup()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM Report Generator", page_icon="🤖", layout="wide")
st.title("Reporting Agent 🤖")
# st.subheader("Instructions:")
st.markdown("""**Instructions**:\n
1) On the left sidebar. *Select* a model. *Load* it and wait until is ready.  
2) Input the form and click on *generate*, to a report.  
\n**NB**: If you change the **language**, the **inputs** in the form must be in that language.\n""")
st.caption("If the models have been loaded once, they will be cached for faster reloads.")

# ──────────────────────────────────────────────────────────────────────────────
# RUNTIME CHOICE
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Select a model")
st.sidebar.markdown("### Backend mode: \n*Locally loaded (no API used)*")

# ──────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION
# ──────────────────────────────────────────────────────────────────────────────

MODEL_OPTIONS = {
    "Qwen2.5-0.5B-Instruct (1 GB)": "Qwen/Qwen2.5-0.5B-Instruct",
    "SmolLM2-360M-Specialized (722 MB)": "zBotta/smollm2-accident-reporter-360m",
    # "GPT-2 tiny (124M)": "gpt2",
}

model_choice = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()), key="model_choice",
help="*Qwen2.5-0.5B-Instruct* generates long, structured reports in different languages.\n" \
     "\n*SmolLM2-360M-Specialized* generates short, one-paragraph reports ONLY in English")
MODEL_ID = MODEL_OPTIONS[model_choice]
IS_QWEN = True if MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct" else False

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
defaults = {
    "loaded_model_id": None,
    "model_ready": False,
    "tok": None,
    "model": None,
    "load_error": None,
    "language_choice": "ENGLISH",
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

def reset_local_state():
    st.session_state.tok = None
    st.session_state.model = None
    st.session_state.model_ready = False
    st.session_state.loaded_model_id = None
    st.session_state.load_error = None
    st.session_state.language_choice = "ENGLISH"

# ──────────────────────────────────────────────────────────────────────────────
# LOCAL LOADER (synchronous; NO st.* INSIDE)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_local_model(model_id: str):
    from mods.modelLoader import ModelLoader
    
    ml = ModelLoader(model_id=model_id, device=env.device, torch_dtype=env.torch_dtype)
    model, tok = ml.load_model(hf_token=env.config["HF_TOKEN"])

    return tok, model

# ──────────────────────────────────────────────────────────────────────────────
# MANUAL LOAD BUTTON (LOCAL)
# ──────────────────────────────────────────────────────────────────────────────

if st.sidebar.button("⚡ Load selected model"):
    try:
        if st.session_state.loaded_model_id != MODEL_ID:
            reset_local_state()
        with st.spinner(f"Loading {MODEL_ID}…"):
            tok, mdl = load_local_model(MODEL_ID)
        st.session_state.tok = tok
        st.session_state.model = mdl
        st.session_state.loaded_model_id = MODEL_ID
        st.session_state.model_ready = True
        st.session_state.load_error = None
        st.success(f"✅ Loaded: {MODEL_ID}")
    except Exception as e:
        st.session_state.load_error = str(e)
        st.error(f"Model load failed: {e}")

# Status
if st.session_state.load_error:
    st.error(f"Model load failed: {st.session_state.load_error}")
elif st.session_state.model_ready and st.session_state.loaded_model_id == MODEL_ID:
    st.info(f"Ready: {MODEL_ID}")
else:
    st.info("No local model loaded. Choose a model and click **Load selected model**.")

# ──────────────────────────────────────────────────────────────────────────────
# FORM UI  ——  ADDED EVENT FIELDS HERE
# ──────────────────────────────────────────────────────────────────────────────
with st.form("gen_form"):
    st.markdown("## Report generation form")
    # Event details
    st.markdown("### Event details")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        ev_what = st.text_input("What", value="A sideswipe collision between two vehicles")
        ev_when = st.text_input("When", value="Monday at 7:30 am")
        ev_who  = st.text_input("Who", value="Driver A (delivery van) & Driver B (hatchback)")
    with ec2:
        ev_where = st.text_input("Where", value="Intersection of Rue de la République & Rue Victor Hugo, Lyon")
        ev_how   = st.text_area("How", value="Van turned right from middle lane while B overtook on bus lane")
    with ec3:
        ev_why   = st.text_area("Why", value="Driver A misjudged traffic flow, missed blind-spot check")
        ev_cont  = st.text_area("Contingency actions", value="No injuries; police called; insurance exchanged")

    # # Optional extras kept
    # st.markdown("### Extras")
    # topic = st.text_input("Topic (optional)", value="Incident summary & actions")
    # objectives = st.text_area(
    #     "Objectives / key points (optional)",
    #     value="- summarize facts\n- identify cause\n- list next steps"
    # )
    if IS_QWEN:
        LANGUAGE_OPTIONS = {
            "English": "ENGLISH",
            "French": "FRENCH",
            "Spanish": "SPANISH",
        }
        # Default parameters for Qwen2.5-0.5B-Instruct
        D_PARAM = {"max_new_tokens": 600,
                "temperature": 0.5,
                "top_p": 0.9,
                "rep_penalty": 1.1,
        }
    else:
        LANGUAGE_OPTIONS = {
            "English": "ENGLISH"
        }
        # Default parameters for SmolLM2-360M-Specialized
        D_PARAM = {"max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.6,
                "rep_penalty": 1.0,
        }

    st.markdown("**Generation settings**")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        max_new_tokens = st.number_input("Max new tokens", 16, 1024, D_PARAM["max_new_tokens"], step=16)
        language_list = st.selectbox("Language", list(LANGUAGE_OPTIONS.keys()), key="language_list")
        language_ui = LANGUAGE_OPTIONS[language_list]
        print("language_ui:", language_ui, file=sys.stderr)
        st.session_state.language_choice = language_ui
    with g2:
        temperature = st.slider("Temperature", 0.0, 1.5, D_PARAM["temperature"], 0.05)
    with g3:
        top_p = st.slider("Top-p", 0.1, 1.0, D_PARAM["top_p"], 0.05)
    with g4:
        rep_penalty = st.slider("Repetition penalty", 1.0, 2.0, D_PARAM["rep_penalty"], 0.05)

    submitted = st.form_submit_button("Generate")


def build_prompt(ev_what, ev_when, ev_why, ev_who, ev_how, ev_where, ev_cont):
    from mods.promptGenerator import PromptGenerator

    pg = PromptGenerator(what=ev_what, 
                         when=ev_when, 
                         why=ev_why, 
                         who=ev_who, 
                         how=ev_how, 
                         where=ev_where, 
                         contingency_actions=ev_cont
                         )
    # TODO: add in deploy config
    if IS_QWEN:
        prompt = pg.create_prompt(prompt_method="Q", language=language_ui)
    else: 
        prompt = pg.create_prompt(prompt_method='D')
    return prompt
    
# ──────────────────────────────────────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def generate_local(prompt, max_new_tokens, temperature, top_p, rep_penalty):
    from mods.reportGenerator import ReportGenerator
    from mods.dataHandler import DataHandler, Report, StreamlitReport

    if not IS_QWEN:
        dh = DataHandler()
        rg = ReportGenerator(model=st.session_state.model,
                            tokenizer=st.session_state.tok,
                            output_type=Report)
        output, gen_param = rg.generate_report(prompt=prompt,
                                               max_new_tokens=max_new_tokens,
                                               temperature=temperature,
                                               top_p=top_p,
                                               repetition_penalty=rep_penalty)
        title, report = dh.get_title_and_report(model_output = output) 
    else: 
        dh = DataHandler()
        rg = ReportGenerator(model=st.session_state.model,
                            tokenizer=st.session_state.tok,
                            output_type=StreamlitReport)
        output, gen_param = rg.generate_report(prompt=prompt,
                                               do_sample=True,
                                               max_new_tokens=max_new_tokens,
                                               temperature=temperature,
                                               top_p=top_p,
                                               repetition_penalty=rep_penalty)
        print("gen_param:", gen_param, file=sys.stderr)
        print("output:", output, file=sys.stderr)
        title = StreamlitReport.model_validate_json(output).title.strip()
        summary = StreamlitReport.model_validate_json(output).summary.strip()
        event_details = StreamlitReport.model_validate_json(output).event_details.strip()
        immediate_actions = StreamlitReport.model_validate_json(output).immediate_actions.strip()
        next_steps = StreamlitReport.model_validate_json(output).next_steps.strip()
        conclusions = StreamlitReport.model_validate_json(output).conclusions.strip()
        language_ui = st.session_state.language_choice
        summary_md = "### **Summary**:\n" if language_ui == "ENGLISH" else "### **Résumé**:\n" if language_ui == "FRENCH" else "### **Resumen**:\n"
        event_details_md = "### **Event Details**:\n" if language_ui == "ENGLISH" else "### **Détails de l'événement**:\n" if language_ui == "FRENCH" else "### **Detalles del evento**:\n"
        immediate_actions_md = "### **Immediate Actions Taken**:\n" if language_ui == "ENGLISH" else "### **Actions immédiates prises**:\n" if language_ui == "FRENCH" else "### **Acciones inmediatas tomadas**:\n"
        next_steps_md = "### **Next Steps / Recommendations**:\n" if language_ui  == "ENGLISH" else "### **Prochaines étapes / Recommandations**:\n" if language_ui == "FRENCH" else "### **Próximos pasos / Recomendaciones**:\n"    
        conclusions_md = "### **Conclusions**: \n" if language_ui == "ENGLISH" else "### **Conclusions**: \n" if language_ui == "FRENCH" else "### **Conclusiones**: \n"
        report = f"{summary_md}\n{summary}\n\n" \
                 f"{event_details_md}\n{event_details}\n\n" \
                 f"{immediate_actions_md}\n{immediate_actions}\n\n" \
                 f"{next_steps_md}\n{next_steps}\n\n" \
                 f"{conclusions_md}\n{conclusions}\n\n"

    return title, report

def make_pdf(report_text: str, title: str = "Generated Report") -> bytes:
    """Create a nicely wrapped PDF from Markdown-capable text."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm,
        title=title,
    )

    # ---- Styles
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    h1 = ParagraphStyle("H1", parent=base, fontSize=16, leading=20, spaceBefore=6, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=base, fontSize=13.5, leading=18, spaceBefore=6, spaceAfter=6)
    h3 = ParagraphStyle("H3", parent=base, fontSize=12, leading=16, spaceBefore=4, spaceAfter=4)
    quote = ParagraphStyle("Quote", parent=base, leftIndent=10, textColor="#444444", italic=True)
    code = ParagraphStyle(
        "Code",
        parent=base,
        fontName="Courier",
        fontSize=9.5,
        leading=12,
        backColor="#f5f5f5",
        leftIndent=6,
        rightIndent=6,
        spaceBefore=4,
        spaceAfter=6,
    )

    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 8)]

    # ---- Prepare text
    text = (report_text or "").replace("\r\n", "\n").strip()
    # Normalize: ensure a blank line around fenced code to simplify splitting
    text = re.sub(r"```+\s*\n", "\n```\n", text)
    text = re.sub(r"\n```", "\n```\n", text)

    # --- Tokenize into blocks, respecting fenced code and inline headings
    lines = text.split("\n")
    blocks, cur, in_code = [], [], False
    heading_re = re.compile(r"^\s*(#{1,6})(?:\s+|$)(.*)$")  # accepts '###Title' and '### Title'

    def flush_cur():
        nonlocal cur
        if cur and any(s.strip() for s in cur):
            blocks.append("\n".join(cur).strip("\n"))
        cur = []

    for ln in lines:
        if ln.strip().startswith("```"):
            if not in_code:
                flush_cur()
                in_code = True
                cur = [ln]
            else:
                cur.append(ln)
                blocks.append("\n".join(cur).strip("\n"))
                cur, in_code = [], False
            continue

        if in_code:
            cur.append(ln)
            continue

        if not ln.strip():
            flush_cur()
            continue

        # Treat a heading line as its own block even without blank lines
        if heading_re.match(ln):
            flush_cur()
            blocks.append(ln.strip())
            continue

        cur.append(ln)

    flush_cur()

    # ---- Helpers
    def is_ul(block: str) -> bool:
        lines_ = [l for l in block.split("\n") if l.strip()]
        return bool(lines_) and all(re.match(r"^\s*[-*+]\s+.+", l) for l in lines_)

    def is_ol(block: str) -> bool:
        lines_ = [l for l in block.split("\n") if l.strip()]
        return bool(lines_) and all(re.match(r"^\s*\d+[.)]\s+.+", l) for l in lines_)

    def convert_inline_md(s: str) -> str:
        """Convert inline markdown to ReportLab's mini-HTML."""
        s = escape(s)

        # Links: [text](url)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)

        # Bold **text**
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

        # Italic *text* (avoid ** already consumed)
        s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)

        # Inline code `code`
        s = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', s)

        # Convert single newlines to <br/>
        s = s.replace("\n", "<br/>")
        return s

    # ---- Build flowables
    for block in blocks:
        b = block.strip()

        # Fenced code block
        if b.startswith("```"):
            code_lines = b.split("\n")
            inner = "\n".join(code_lines[1:-1])  # drop first/last fence
            story.append(Preformatted(inner, code))
            continue

        # Horizontal rule (---, ***)
        if re.fullmatch(r"\s*(-{3,}|\*{3,})\s*", b):
            story.append(HRFlowable(width="100%", color="#aaaaaa", spaceBefore=6, spaceAfter=6, thickness=0.7))
            continue

        # Headings
        m = heading_re.match(b)
        if m:
            level = len(m.group(1))
            content = convert_inline_md(m.group(2).strip())
            style = h1 if level == 1 else h2 if level == 2 else h3
            story.append(Paragraph(content, style))
            continue

        # Blockquote
        if b.startswith(">"):
            q = "\n".join([re.sub(r"^\s*>\s?", "", ln) for ln in b.split("\n")])
            story.append(Paragraph(convert_inline_md(q), quote))
            continue

        # Lists
        if is_ul(b) or is_ol(b):
            items = []
            ordered = is_ol(b)
            for ln in b.split("\n"):
                if not ln.strip():
                    continue
                if ordered:
                    clean = re.sub(r"^\s*\d+[.)]\s+", "", ln).strip()
                else:
                    clean = re.sub(r"^\s*[-*+]\s+", "", ln).strip()
                items.append(ListItem(Paragraph(convert_inline_md(clean), base)))
            story.append(
                ListFlowable(
                    items,
                    bulletType="1" if ordered else "bullet",
                    start="1",
                    leftIndent=12
                )
            )
            continue

        # Default paragraph
        story.append(Paragraph(convert_inline_md(b), base))

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf



# ──────────────────────────────────────────────────────────────────────────────
# RUN GENERATION
# ──────────────────────────────────────────────────────────────────────────────
if submitted:
    prompt = build_prompt(ev_what, ev_when, ev_why, ev_who, ev_how, ev_where, ev_cont)
    try:
        # if backend_mode == "Local (Transformers)":
        if not (st.session_state.model_ready and st.session_state.loaded_model_id == MODEL_ID):
            st.error("No local model is ready. Choose a model and click **Load selected model** first.")
            st.stop()
        label = MODEL_ID
        with st.spinner(f"Generating with {label}…"):
            report_title, report = generate_local(prompt, max_new_tokens, temperature, top_p, rep_penalty)
       
        st.success("Done!")
        st.header("Generated Report")
        title_md = "## Title: " if language_ui == "ENGLISH" else "## Titre : " if language_ui == "FRENCH" else "## Título: "
        st.markdown(title_md + report_title)
        st.markdown(report)
        pdf_bytes = make_pdf(report, title=report_title)
        st.download_button(
            "⬇️ Download report (PDF)",
            data=pdf_bytes,
            file_name= report_title + ".pdf",
            mime="application/pdf"
        )

        with st.expander("Show prompt (debug)"):
            st.code(prompt)

    except Exception as e:
        st.error(f"Generation failed: {e}")
