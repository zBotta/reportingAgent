
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
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


env = Setup()

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM Report Generator", page_icon="🤖", layout="wide")
st.title(" Reporting Agent 🤖")
st.caption("Select a model, load it and wait until is ready. Then, generate a report from your form inputs.")

# ──────────────────────────────────────────────────────────────────────────────
# RUNTIME CHOICE
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Select a model")
st.sidebar.markdown("### Backend mode: \nLocal locally")
backend_mode = "Local (Transformers)"  # default to local mode
# backend_mode = st.sidebar.radio(
#     "Choose backend",
#     options=["Local (Transformers)", "Hugging Face Inference API"],
#     help="Local runs on your machine. HF Inference API calls a hosted endpoint (needs token).",
# )

# ──────────────────────────────────────────────────────────────────────────────
# MODEL SELECTION
# ──────────────────────────────────────────────────────────────────────────────

MODEL_OPTIONS = {
    "Phi-2 (Microsoft)": "microsoft/phi-2",
    "GPT-2 Large (774M)": "gpt2-xl",
    "GPT-2 tiny (124M)": "gpt2",
}

model_choice = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()), key="model_choice")
MODEL_ID = MODEL_OPTIONS[model_choice]

# # HF API (no preloading)
# if backend_mode == "Hugging Face Inference API":
#     HF_TOKEN = st.sidebar.text_input(
#         "Hugging Face Access Token",
#         type="password",
#         value=os.environ.get("HF_TOKEN", ""),
#         help="Create one at https://huggingface.co/settings/tokens (scope: read).",
#     )
#     API_MODEL_ID = MODEL_ID

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
defaults = {
    "loaded_model_id": None,
    "model_ready": False,
    "tok": None,
    "model": None,
    "load_error": None,
    # "load_meta": {},  # {'repo': str, 'quantized': bool}
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

def reset_local_state():
    st.session_state.tok = None
    st.session_state.model = None
    st.session_state.model_ready = False
    st.session_state.loaded_model_id = None
    st.session_state.load_error = None
    # st.session_state.load_meta = {}

# ──────────────────────────────────────────────────────────────────────────────
# LOCAL LOADER (synchronous; NO st.* INSIDE)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_local_model(model_id: str):
    from mods.modelLoader import ModelLoader
    
    ml = ModelLoader(model_id=model_id, device=env.device, torch_dtype=env.torch_dtype)
    model, tok = ml.load_model(hf_token=env.config["HF_TOKEN"]) # TODO: maybe change the loader without outlines for the APP

    # _ = model.generate(
    #     **tok("Hello", return_tensors="pt").to(model.device),
    #     max_new_tokens=2,
    #     do_sample=False,
    #     pad_token_id=tok.pad_token_id,
    # )
    return tok, model

# ──────────────────────────────────────────────────────────────────────────────
# MANUAL LOAD BUTTON (LOCAL)
# ──────────────────────────────────────────────────────────────────────────────
if backend_mode == "Local (Transformers)":
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
if backend_mode == "Local (Transformers)":
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
        ev_what = st.text_input("What", placeholder="A sideswipe collision between two vehicles")
        ev_when = st.text_input("When", placeholder="Monday at 7:30 am")
        ev_who  = st.text_input("Who", placeholder="Driver A (delivery van) & Driver B (hatchback)")
    with ec2:
        ev_where = st.text_input("Where", placeholder="Intersection of Rue de la République & Rue Victor Hugo, Lyon")
        ev_how   = st.text_area("How", placeholder="Van turned right from middle lane while B overtook on bus lane")
    with ec3:
        ev_why   = st.text_area("Why", placeholder="Driver A misjudged traffic flow, missed blind-spot check")
        ev_cont  = st.text_area("Contingency actions", placeholder="No injuries; police called; insurance exchanged")

    # # Optional extras kept
    # st.markdown("### Extras")
    # topic = st.text_input("Topic (optional)", value="Incident summary & actions")
    # objectives = st.text_area(
    #     "Objectives / key points (optional)",
    #     value="- summarize facts\n- identify cause\n- list next steps"
    # )

    st.markdown("**Generation settings**")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        max_new_tokens = st.number_input("Max new tokens", 16, 1024, 256, step=16)
    with g2:
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    with g3:
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    with g4:
        rep_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1, 0.05)

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
    PROMPT_METHOD = 'B'  # TODO: add in deploy config
    prompt = pg.create_prompt(prompt_method=PROMPT_METHOD)
    return prompt
    
# ──────────────────────────────────────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def generate_local(prompt, max_new_tokens, temperature, top_p, rep_penalty):
    from mods.reportGenerator import ReportGenerator
    from mods.dataHandler import DataHandler, Report

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
    # tok = st.session_state.tok
    # model = st.session_state.model
    # inputs = tok(prompt, return_tensors="pt")
    # inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # # with torch.no_grad():  # TODO: discuss with Samd ?
    #     out_ids = model.generate(
    #         **inputs,
    #         max_new_tokens=int(max_new_tokens),
    #         do_sample=True,
    #         temperature=float(temperature),
    #         top_p=float(top_p),
    #         repetition_penalty=float(rep_penalty),
    #         pad_token_id=tok.pad_token_id,
    #     )
    return title, report

def make_pdf(report_text: str, title: str = "Generated Report") -> bytes:
    """Return PDF bytes for the given text content."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36,
        title=title
    )
    styles = getSampleStyleSheet()
    flow = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
    ]
    mono = ParagraphStyle(
        name="Mono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=10,
        leading=12,
    )
    # Preformatted preserves newlines/spacing (good for markdown-ish text)
    flow.append(Preformatted(report_text, mono))
    doc.build(flow)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# ──────────────────────────────────────────────────────────────────────────────
# RUN GENERATION
# ──────────────────────────────────────────────────────────────────────────────
if submitted:
    prompt = build_prompt(ev_what, ev_when, ev_why, ev_who, ev_how, ev_where, ev_cont)
    try:
        if backend_mode == "Local (Transformers)":
            if not (st.session_state.model_ready and st.session_state.loaded_model_id == MODEL_ID):
                st.error("No local model is ready. Choose a model and click **Load selected model** first.")
                st.stop()
            label = MODEL_ID
            with st.spinner(f"Generating with {label}…"):
                report_title, report = generate_local(prompt, max_new_tokens, temperature, top_p, rep_penalty)
       
        st.success("Done!")
        st.subheader("Generated Report")
        st.markdown("Title: " + report_title)
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
