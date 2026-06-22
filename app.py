
import streamlit as st
import pandas as pd
import re
import io
import os
import tempfile
import subprocess
from docx import Document
from pypdf import PdfReader
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)

st.set_page_config(
    page_title="Bloom's-Level Balance Analyzer",
    layout="wide"
)

# ============================================================
# BLOOM SETTINGS
# ============================================================

BLOOM_LEVELS = [
    "Remember & Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

BLOOM_ORDER = {
    "Remember & Understand": 1,
    "Apply": 2,
    "Analyze": 3,
    "Evaluate": 4,
    "Create": 5
}

NUMERIC_BLOOM_MAP = {
    "1": "Remember & Understand",
    "2": "Apply",
    "3": "Analyze",
    "4": "Evaluate",
    "5": "Create"
}

TARGET_DISTRIBUTION = {
    "Remember & Understand": 30,
    "Apply": 25,
    "Analyze": 25,
    "Evaluate": 10,
    "Create": 10
}

BLOOM_VERBS = {
    "Remember & Understand": [
        "define", "list", "name", "identify", "recall", "state",
        "what", "which", "who", "when", "where", "explain",
        "describe", "summarize", "outline", "interpret",
        "illustrate", "discuss", "classify", "categorize"
    ],
    "Apply": [
        "apply", "solve", "calculate", "compute", "demonstrate",
        "implement", "use", "show", "prepare", "execute"
    ],
    "Analyze": [
        "analyze", "analyse", "compare", "contrast", "differentiate",
        "distinguish", "examine", "determine", "infer", "inspect"
    ],
    "Evaluate": [
        "evaluate", "justify", "assess", "criticize", "critique",
        "validate", "defend", "recommend", "review"
    ],
    "Create": [
        "design", "develop", "construct", "create", "formulate",
        "propose", "build", "generate", "compose"
    ]
}


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(x):
    x = "" if x is None else str(x)
    x = x.replace("\xa0", " ")
    x = x.replace("\n", " ")
    x = x.replace("\t", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_bloom(value):
    value = clean_text(value)

    if value in NUMERIC_BLOOM_MAP:
        return NUMERIC_BLOOM_MAP[value]

    v = value.lower()

    if "remember" in v or "understand" in v or "k1" in v or "l1" in v:
        return "Remember & Understand"
    if "apply" in v or "k2" in v or "l2" in v:
        return "Apply"
    if "analyze" in v or "analyse" in v or "k3" in v or "l3" in v:
        return "Analyze"
    if "evaluate" in v or "k4" in v or "l4" in v:
        return "Evaluate"
    if "create" in v or "k5" in v or "l5" in v:
        return "Create"

    return "Not Provided"


def is_number(x):
    return bool(re.fullmatch(r"\d+", clean_text(x)))


def remove_mcq_options(question):
    question = clean_text(question)
    question = re.split(r"\bA\.\s+", question)[0]
    question = re.split(r"\ba\.\s+", question)[0]
    return clean_text(question)


def is_or_row(row):
    joined = " ".join([clean_text(x).lower().replace("|", "") for x in row])
    parts = [clean_text(x).lower().replace("|", "") for x in row if clean_text(x)]
    if not parts:
        return True
    if all(x == "or" for x in parts):
        return True
    if joined.strip() == "or":
        return True
    return False


def is_valid_sno(sno):
    sno = clean_text(sno)
    return bool(
        re.fullmatch(
            r"\d+|\d+\([a-zA-Z]\)|\([a-zA-Z]\)|[a-zA-Z]\)|[a-zA-Z]",
            sno
        )
    )


def normalize_sno_value(sno, last_main_no):
    sno = clean_text(sno)

    if re.fullmatch(r"\d+", sno):
        return sno

    if re.fullmatch(r"\d+\([a-zA-Z]\)", sno):
        return sno

    if re.fullmatch(r"\([a-zA-Z]\)", sno) and last_main_no:
        return f"{last_main_no}{sno}"

    if re.fullmatch(r"[a-zA-Z]\)", sno) and last_main_no:
        return f"{last_main_no}({sno[0]})"

    if re.fullmatch(r"[a-zA-Z]", sno) and last_main_no:
        return f"{last_main_no}({sno})"

    return sno


# ============================================================
# DOC / DOCX EXTRACTION
# ============================================================

def convert_doc_to_docx(file_bytes):
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_path = os.path.join(tmpdir, "input.doc")

        with open(doc_path, "wb") as f:
            f.write(file_bytes)

        try:
            subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    tmpdir,
                    doc_path
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            docx_path = os.path.join(tmpdir, "input.docx")

            if os.path.exists(docx_path):
                with open(docx_path, "rb") as f:
                    return f.read()

        except Exception:
            return None

    return None


def extract_docx_rows(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    rows = []

    for table in doc.tables:
        for row in table.rows:
            cells = [clean_text(cell.text) for cell in row.cells]
            rows.append(cells)

    return rows


def is_header_row(row):
    joined = " ".join([clean_text(x).lower() for x in row])
    return (
        ("s.no" in joined or "s no" in joined or "sno" in joined)
        and "question" in joined
        and ("course outcome" in joined or "co" in joined)
        and "bloom" in joined
        and "marks" in joined
    )


def extract_questions_from_docx_rows(rows):
    extracted = []
    inside_question_table = False
    last_main_no = None

    for row in rows:
        row = [clean_text(x) for x in row]

        if is_header_row(row):
            inside_question_table = True
            continue

        if not inside_question_table:
            continue

        if is_or_row(row):
            continue

        if len(row) < 2:
            continue

        sno = clean_text(row[0])
        question = clean_text(row[1])

        if not is_valid_sno(sno):
            continue

        if not question or question.lower() in ["question", "or"]:
            continue

        if re.match(r"^\d+", sno):
            last_main_no = re.match(r"^\d+", sno).group()

        sno = normalize_sno_value(sno, last_main_no)

        numeric_cells = [clean_text(x) for x in row if is_number(x)]

        if len(numeric_cells) >= 3:
            co = numeric_cells[-3]
            bloom = numeric_cells[-2]
            marks = numeric_cells[-1]
        else:
            continue

        extracted.append({
            "S.No.": sno,
            "Question": remove_mcq_options(question),
            "Full Question with Options": question,
            "Course Outcome": co,
            "Bloom’s Taxonomy Level": bloom,
            "Bloom Level in Paper": normalize_bloom(bloom),
            "Marks": marks
        })

    df = pd.DataFrame(extracted)

    if df.empty:
        return df

    df = df[df["Question"].str.lower() != "or"]
    df = df[df["S.No."].str.lower() != "or"]
    df = df.drop_duplicates(subset=["S.No.", "Question"])
    return df.reset_index(drop=True)


def extract_questions_from_doc_or_docx(file_bytes, file_name):
    if file_name.lower().endswith(".doc"):
        converted = convert_doc_to_docx(file_bytes)

        if converted is None:
            return pd.DataFrame()

        file_bytes = converted

    rows = extract_docx_rows(file_bytes)
    return extract_questions_from_docx_rows(rows)


# ============================================================
# PDF / TEXT EXTRACTION FALLBACK
# ============================================================

def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)

    return "\n".join(text)


def extract_questions_from_plain_text(text):
    lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]
    rows = []
    header_seen = False
    last_main_no = None
    i = 0

    while i < len(lines):
        line = lines[i].lower()

        if (
            ("s.no" in line or "s no" in line or "sno" in line)
            and "question" in line
            and ("course outcome" in line or "co" in line)
            and "bloom" in line
            and "marks" in line
        ):
            header_seen = True
            i += 1
            continue

        if not header_seen:
            i += 1
            continue

        if lines[i].lower() == "or":
            i += 1
            continue

        if is_valid_sno(lines[i]):
            sno_raw = lines[i]

            if re.match(r"^\d+", sno_raw):
                last_main_no = re.match(r"^\d+", sno_raw).group()

            sno = normalize_sno_value(sno_raw, last_main_no)
            question_parts = []
            i += 1

            while i < len(lines):
                current = lines[i]

                if current.lower() == "or":
                    i += 1
                    break

                if is_valid_sno(current) and question_parts:
                    break

                if is_number(current) and i + 2 < len(lines) and is_number(lines[i + 1]) and is_number(lines[i + 2]):
                    co = current
                    bloom = lines[i + 1]
                    marks = lines[i + 2]

                    rows.append({
                        "S.No.": sno,
                        "Question": remove_mcq_options(" ".join(question_parts)),
                        "Full Question with Options": " ".join(question_parts),
                        "Course Outcome": co,
                        "Bloom’s Taxonomy Level": bloom,
                        "Bloom Level in Paper": normalize_bloom(bloom),
                        "Marks": marks
                    })

                    i += 3
                    break

                question_parts.append(current)
                i += 1
        else:
            i += 1

    return pd.DataFrame(rows)


# ============================================================
# BLOOM CLASSIFICATION
# ============================================================

def rule_based_classifier(question):
    q = remove_mcq_options(question).lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    scores = {level: 0 for level in BLOOM_LEVELS}
    matched = []

    for level, verbs in BLOOM_VERBS.items():
        for verb in verbs:
            if re.search(r"\b" + re.escape(verb.lower()) + r"\b", q):
                scores[level] += 1
                matched.append(f"{verb} → {level}")

    max_score = max(scores.values())

    if max_score == 0:
        return "Remember & Understand", "No clear verb found", ""

    best = [level for level, score in scores.items() if score == max_score]
    selected = max(best, key=lambda x: BLOOM_ORDER[x])

    remark = "Keyword matched"

    if len(best) > 1:
        remark = "Mixed verbs found; higher Bloom level selected"

    return selected, remark, ", ".join(matched)


def create_training_data():
    data = [
        ("What is natural language processing?", "Remember & Understand"),
        ("Define word embeddings.", "Remember & Understand"),
        ("Explain rule based POS tagging.", "Remember & Understand"),
        ("Describe word sense disambiguation.", "Remember & Understand"),
        ("Summarize advantages of NLP.", "Remember & Understand"),
        ("Illustrate the concept of syntax and semantics.", "Remember & Understand"),

        ("Apply lexicons in NLP systems.", "Apply"),
        ("Demonstrate NLP components with applications.", "Apply"),
        ("Use NLTK for text preprocessing.", "Apply"),
        ("Solve the given NLP task.", "Apply"),

        ("Analyze ambiguity resolution challenges.", "Analyze"),
        ("Analyse the importance of coherence.", "Analyze"),
        ("Compare NLU and NLG.", "Analyze"),
        ("Differentiate syntax and semantics.", "Analyze"),
        ("Examine prerequisites of discourse processing.", "Analyze"),

        ("Evaluate the limitations of rule based NLP.", "Evaluate"),
        ("Justify the use of transformers.", "Evaluate"),
        ("Assess the performance of an NLP model.", "Evaluate"),

        ("Design an NLP pipeline.", "Create"),
        ("Develop a chatbot.", "Create"),
        ("Propose a semantic analysis system.", "Create"),
        ("Formulate an algorithm for WSD.", "Create")
    ]

    return pd.DataFrame(data, columns=["Question", "Bloom_Level"])


def train_model():
    df = create_training_data()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(df["Question"], df["Bloom_Level"])
    return model, df


model, training_df = train_model()


def hybrid_classifier(question, paper_bloom):
    rule_level, rule_remark, matched_verbs = rule_based_classifier(question)
    nlp_level = model.predict([remove_mcq_options(question)])[0]

    if paper_bloom != "Not Provided":
        final_level = paper_bloom
        status = "Final level taken from question paper table"
    elif rule_level == nlp_level:
        final_level = rule_level
        status = "Rule and NLP agree"
    else:
        final_level = max([rule_level, nlp_level], key=lambda x: BLOOM_ORDER[x])
        status = "Rule and NLP differ; higher cognitive level selected"

    return rule_level, nlp_level, final_level, status, rule_remark, matched_verbs


# ============================================================
# REPORTING
# ============================================================

def generate_balance_report(df):
    total = len(df)

    if total == 0:
        return pd.DataFrame(), [], 0, 0

    counts = df["Final Bloom Level"].value_counts().reindex(BLOOM_LEVELS, fill_value=0)
    actual = (counts / total * 100).round(2)

    report = pd.DataFrame({
        "Bloom Level": BLOOM_LEVELS,
        "Question Count": counts.values,
        "Actual %": actual.values,
        "Recommended %": [TARGET_DISTRIBUTION[x] for x in BLOOM_LEVELS]
    })

    report["Difference %"] = report["Actual %"] - report["Recommended %"]

    lower = report[
        report["Bloom Level"] == "Remember & Understand"
    ]["Actual %"].sum()

    higher = report[
        report["Bloom Level"].isin(["Apply", "Analyze", "Evaluate", "Create"])
    ]["Actual %"].sum()

    suggestions = []

    for _, row in report.iterrows():
        if row["Difference %"] > 10:
            suggestions.append(
                f"Reduce {row['Bloom Level']} questions by about {row['Difference %']:.1f}%."
            )
        elif row["Difference %"] < -10:
            suggestions.append(
                f"Increase {row['Bloom Level']} questions by about {abs(row['Difference %']):.1f}%."
            )

    if higher < 50:
        suggestions.append(
            "Higher-order thinking is low. Add more Apply, Analyze, Evaluate and Create questions."
        )

    if report.loc[report["Bloom Level"] == "Evaluate", "Question Count"].iloc[0] == 0:
        suggestions.append(
            "No Evaluate-level question found. Add justify/evaluate/assess type questions."
        )

    if report.loc[report["Bloom Level"] == "Create", "Question Count"].iloc[0] == 0:
        suggestions.append(
            "No Create-level question found. Add design/develop/propose type questions."
        )

    if not suggestions:
        suggestions.append("The question paper has a balanced Bloom distribution.")

    return report, suggestions, lower, higher


def validate_against_expert(expert_df):
    if not {"Question", "Expert_Level"}.issubset(expert_df.columns):
        return None, "CSV must contain Question and Expert_Level columns."

    expert_df = expert_df.copy()
    expert_df["Expert_Level"] = expert_df["Expert_Level"].apply(normalize_bloom)

    preds = []

    for q in expert_df["Question"]:
        _, _, final, _, _, _ = hybrid_classifier(q, "Not Provided")
        preds.append(final)

    expert_df["Predicted_Level"] = preds

    metrics = {
        "Accuracy": accuracy_score(expert_df["Expert_Level"], expert_df["Predicted_Level"]),
        "Cohen Kappa": cohen_kappa_score(expert_df["Expert_Level"], expert_df["Predicted_Level"])
    }

    report = classification_report(
        expert_df["Expert_Level"],
        expert_df["Predicted_Level"],
        labels=BLOOM_LEVELS,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        expert_df["Expert_Level"],
        expert_df["Predicted_Level"],
        labels=BLOOM_LEVELS
    )

    return {
        "data": expert_df,
        "metrics": metrics,
        "classification_report": pd.DataFrame(report).transpose(),
        "confusion_matrix": pd.DataFrame(cm, index=BLOOM_LEVELS, columns=BLOOM_LEVELS)
    }, None


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Question-Paper Bloom's-Level Balance Analyzer")
st.caption("Correctly extracts S.No., Question, Course Outcome, Bloom’s Taxonomy Level and Marks")

tab1, tab2, tab3 = st.tabs([
    "Analyze Question Paper",
    "Expert Validation",
    "Training Data"
])


with tab1:
    st.subheader("Upload DOC/DOCX/PDF/TXT/CSV/XLSX Question Paper")

    uploaded_file = st.file_uploader(
        "Upload Question Paper",
        type=["doc", "docx", "pdf", "txt", "csv", "xlsx"]
    )

    manual_text = st.text_area(
        "Or paste question paper text",
        height=250
    )

    question_df = pd.DataFrame()

    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".doc") or file_name.endswith(".docx"):
            question_df = extract_questions_from_doc_or_docx(file_bytes, file_name)

            if question_df.empty:
                st.warning("No valid DOC/DOCX question table found.")

        elif file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
            question_df = extract_questions_from_plain_text(text)

        elif file_name.endswith(".txt"):
            text = file_bytes.decode("utf-8", errors="ignore")
            question_df = extract_questions_from_plain_text(text)

        elif file_name.endswith(".csv"):
            question_df = pd.read_csv(io.BytesIO(file_bytes))

        elif file_name.endswith(".xlsx"):
            question_df = pd.read_excel(io.BytesIO(file_bytes))

    elif manual_text.strip():
        question_df = extract_questions_from_plain_text(manual_text)

    if not question_df.empty:
        st.subheader("Extracted Table")
        st.dataframe(question_df, width="stretch")
        st.info(f"Extracted {len(question_df)} valid question rows. OR rows removed.")

    if st.button("Analyze Bloom Balance", type="primary"):

        if question_df.empty:
            st.error(
                "No table extracted. Ensure the paper has columns: S.No., Question, Course Outcome, Bloom’s Taxonomy Level, Marks."
            )
        else:
            results = []

            for _, row in question_df.iterrows():
                sno = clean_text(row.get("S.No.", ""))
                question = clean_text(row.get("Question", ""))
                full_question = clean_text(row.get("Full Question with Options", question))
                co = clean_text(row.get("Course Outcome", ""))
                bloom_raw = clean_text(row.get("Bloom’s Taxonomy Level", ""))
                marks = clean_text(row.get("Marks", ""))

                if not question or question.lower() == "or":
                    continue

                paper_bloom = normalize_bloom(bloom_raw)

                rule, nlp, final, status, remark, verbs = hybrid_classifier(question, paper_bloom)

                results.append({
                    "S.No.": sno,
                    "Question": question,
                    "Full Question with Options": full_question,
                    "Course Outcome": co,
                    "Bloom’s Taxonomy Level": bloom_raw,
                    "Bloom Level in Paper": paper_bloom,
                    "Marks": marks,
                    "Rule-Based Baseline": rule,
                    "NLP Classifier": nlp,
                    "Final Bloom Level": final,
                    "Robustness Status": status,
                    "Rule Remark": remark,
                    "Matched Verbs": verbs
                })

            result_df = pd.DataFrame(results)

            st.success(f"{len(result_df)} questions analyzed successfully.")

            st.subheader("Question-wise Bloom Analysis")
            st.dataframe(result_df, width="stretch")

            report, suggestions, lower, higher = generate_balance_report(result_df)

            st.subheader("Bloom Distribution")

            c1, c2 = st.columns(2)

            with c1:
                st.dataframe(report, width="stretch")

            with c2:
                fig = px.bar(
                    report,
                    x="Bloom Level",
                    y="Actual %",
                    text="Actual %",
                    title="Bloom-Level Distribution"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig, width="stretch")

            st.subheader("Decision Output")

            m1, m2, m3 = st.columns(3)
            m1.metric("Remember & Understand", f"{lower:.1f}%")
            m2.metric("Higher Order Thinking", f"{higher:.1f}%")
            m3.metric("Total Questions", len(result_df))

            for s in suggestions:
                st.info(s)

            st.download_button(
                "Download Question-wise Report CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                "bloom_questionwise_report.csv",
                "text/csv"
            )

            st.download_button(
                "Download Balance Report CSV",
                report.to_csv(index=False).encode("utf-8"),
                "bloom_balance_report.csv",
                "text/csv"
            )


with tab2:
    st.subheader("Expert Validation")

    expert_file = st.file_uploader(
        "Upload CSV with Question and Expert_Level columns",
        type=["csv"],
        key="expert"
    )

    if expert_file:
        expert_df = pd.read_csv(expert_file)
        validation, error = validate_against_expert(expert_df)

        if error:
            st.error(error)
        else:
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{validation['metrics']['Accuracy'] * 100:.2f}%")
            col2.metric("Cohen Kappa", f"{validation['metrics']['Cohen Kappa']:.2f}")

            st.subheader("Validation Table")
            st.dataframe(validation["data"], width="stretch")

            st.subheader("Per-Level Report")
            st.dataframe(validation["classification_report"], width="stretch")

            st.subheader("Confusion Matrix")
            st.dataframe(validation["confusion_matrix"], width="stretch")


with tab3:
    st.subheader("Training Data")
    st.dataframe(training_df, width="stretch")
