"""
Generate a professional PDF project report for TruthLens using ReportLab.
"""
import subprocess, sys, datetime, os

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "reportlab"])

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor

ACCENT    = HexColor("#2563EB")
DARK      = HexColor("#0F172A")
LIGHT_BG  = HexColor("#EFF6FF")
CODE_BG   = HexColor("#1E293B")
CODE_FG   = HexColor("#E2E8F0")
MID       = HexColor("#64748B")
WHITE     = colors.white
GREEN     = HexColor("#16A34A")
AMBER     = HexColor("#D97706")

W, H = A4
MARGIN = 2 * cm


def make_styles():
    base = getSampleStyleSheet()
    s = {}

    s["body"] = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10, leading=16,
        textColor=DARK, spaceAfter=6, alignment=TA_JUSTIFY
    )
    s["bullet"] = ParagraphStyle(
        "bullet", fontName="Helvetica", fontSize=10, leading=15,
        textColor=DARK, spaceAfter=3, leftIndent=14, bulletIndent=4,
        bulletText="\u2022"
    )
    s["section"] = ParagraphStyle(
        "section", fontName="Helvetica-Bold", fontSize=12, leading=18,
        textColor=ACCENT, spaceBefore=10, spaceAfter=4
    )
    s["chapter"] = ParagraphStyle(
        "chapter", fontName="Helvetica-Bold", fontSize=16, leading=22,
        textColor=WHITE, spaceBefore=0, spaceAfter=0, backColor=ACCENT
    )
    s["toc_num"] = ParagraphStyle(
        "toc_num", fontName="Helvetica-Bold", fontSize=11, leading=18,
        textColor=ACCENT
    )
    s["toc_title"] = ParagraphStyle(
        "toc_title", fontName="Helvetica", fontSize=11, leading=18,
        textColor=DARK
    )
    s["code"] = ParagraphStyle(
        "code", fontName="Courier", fontSize=8, leading=12,
        textColor=CODE_FG, backColor=CODE_BG,
        leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=4
    )
    s["info"] = ParagraphStyle(
        "info", fontName="Helvetica-Oblique", fontSize=9.5, leading=14,
        textColor=HexColor("#1E3A8A"), backColor=LIGHT_BG,
        leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=4
    )
    s["qa_q"] = ParagraphStyle(
        "qa_q", fontName="Helvetica-Bold", fontSize=10.5, leading=16,
        textColor=ACCENT, spaceBefore=8, spaceAfter=2
    )
    s["qa_a"] = ParagraphStyle(
        "qa_a", fontName="Helvetica", fontSize=10, leading=15,
        textColor=DARK, spaceAfter=6, leftIndent=8, alignment=TA_JUSTIFY
    )
    s["caption"] = ParagraphStyle(
        "caption", fontName="Helvetica-Oblique", fontSize=8, leading=12,
        textColor=MID, spaceAfter=6, alignment=TA_CENTER
    )
    s["cover_title"] = ParagraphStyle(
        "cover_title", fontName="Helvetica-Bold", fontSize=42, leading=52,
        textColor=WHITE, alignment=TA_CENTER
    )
    s["cover_sub"] = ParagraphStyle(
        "cover_sub", fontName="Helvetica", fontSize=15, leading=22,
        textColor=HexColor("#BFDBFE"), alignment=TA_CENTER
    )
    s["cover_detail"] = ParagraphStyle(
        "cover_detail", fontName="Helvetica", fontSize=11, leading=16,
        textColor=HexColor("#93C5FD"), alignment=TA_CENTER
    )
    s["cover_author"] = ParagraphStyle(
        "cover_author", fontName="Helvetica-Bold", fontSize=12, leading=18,
        textColor=WHITE, alignment=TA_CENTER
    )
    return s


def chapter_header(title, num=""):
    label = f"  {num}   {title}" if num else f"  {title}"
    t = Table([[Paragraph(label, ParagraphStyle(
        "ch", fontName="Helvetica-Bold", fontSize=14, leading=20, textColor=WHITE
    ))]], colWidths=[W - 2 * MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    return t


def section_header(title, styles):
    return [
        Paragraph(title, styles["section"]),
        HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=4),
    ]


def body(text, styles):
    return Paragraph(text, styles["body"])


def bullets(items, styles):
    return [Paragraph(item, styles["bullet"]) for item in items]


def code_block(code, styles):
    lines = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(f'<font name="Courier" size="8">{lines}</font>', styles["code"])


def info_box(text, styles):
    return Paragraph(text, styles["info"])


def kv_table(rows):
    data = [[Paragraph(f"<b>{k}</b>", ParagraphStyle("kk", fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=DARK)),
             Paragraph(v, ParagraphStyle("kv", fontName="Helvetica", fontSize=10, leading=14, textColor=DARK))]
            for k, v in rows]
    t = Table(data, colWidths=[5 * cm, W - 2 * MARGIN - 5 * cm])
    style = [
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
    ]
    for i in range(len(rows)):
        bg = LIGHT_BG if i % 2 == 0 else WHITE
        style.append(("BACKGROUND", (0, i), (-1, i), bg))
    t.setStyle(TableStyle(style))
    return t


def metrics_table(headers, rows):
    col_w = (W - 2 * MARGIN) / len(headers)
    data = [[Paragraph(f"<b>{h}</b>", ParagraphStyle("mh", fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=WHITE, alignment=TA_CENTER))
             for h in headers]]
    for row in rows:
        data.append([Paragraph(c, ParagraphStyle("mc", fontName="Helvetica", fontSize=10, leading=14, textColor=DARK, alignment=TA_CENTER))
                     for c in row])
    t = Table(data, colWidths=[col_w] * len(headers))
    style = [
        ("BACKGROUND",   (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(rows) + 1):
        bg = LIGHT_BG if i % 2 == 1 else WHITE
        style.append(("BACKGROUND", (0, i), (-1, i), bg))
    t.setStyle(TableStyle(style))
    return t


def cover_page(styles):
    cover_bg_table = Table(
        [[Paragraph("TruthLens", styles["cover_title"]),
          Spacer(1, 12),
          Paragraph("Misinformation Intelligence Platform", styles["cover_sub"]),
          Spacer(1, 6),
          Paragraph("Full Project Report &mdash; Architecture, ML Pipeline &amp; API", styles["cover_detail"]),
          Spacer(1, 40),
          Paragraph("Meet Modi", styles["cover_author"]),
          Paragraph(datetime.date.today().strftime("%B %d, %Y"), styles["cover_detail"]),
          Paragraph("github.com/meetcodz/fake-news-detection-system", styles["cover_detail"]),
          ]],
        colWidths=[W - 2 * MARGIN]
    )
    cover_bg_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), ACCENT),
        ("TOPPADDING",   (0, 0), (-1, -1), 100),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 80),
        ("LEFTPADDING",  (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return [cover_bg_table, PageBreak()]


def toc_page(styles):
    items = []
    items.append(chapter_header("Table of Contents"))
    items.append(Spacer(1, 12))
    toc = [
        ("1",  "Project Overview"),
        ("2",  "Problem Statement"),
        ("3",  "System Architecture"),
        ("4",  "ML Pipeline - Stage by Stage"),
        ("5",  "Explainable AI (XAI)"),
        ("6",  "RAG Fact-Check Engine"),
        ("7",  "REST API"),
        ("8",  "TruthLens UI"),
        ("9",  "Model Performance & Benchmarks"),
        ("10", "Tech Stack"),
        ("11", "Project Structure"),
        ("12", "Setup & Running Locally"),
        ("13", "Interview Q&A Guide"),
    ]
    data = [[Paragraph(n + ".", styles["toc_num"]),
             Paragraph(t, styles["toc_title"])] for n, t in toc]
    t = Table(data, colWidths=[1.2 * cm, W - 2 * MARGIN - 1.2 * cm])
    t.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    items.append(t)
    items.append(PageBreak())
    return items


def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="TruthLens - Project Report",
        author="Meet Modi",
    )

    styles = make_styles()
    story = []

    story += cover_page(styles)
    story += toc_page(styles)

    def sec(*elems):
        for e in elems:
            if isinstance(e, list):
                story.extend(e)
            else:
                story.append(e)

    def sp(n=6):
        story.append(Spacer(1, n))

    # ── 1. PROJECT OVERVIEW ─────────────────────────────────────────────────
    story.append(chapter_header("Project Overview", "1"))
    sp(10)
    sec(body(
        "TruthLens is a production-grade misinformation intelligence platform built as a "
        "portfolio project for Devkriti'26. Unlike trivial classifiers, TruthLens is designed "
        "as a full software product: a modular ML pipeline, REST API, explainability layer, "
        "RAG fact-check engine, and a browser-based UI.", styles
    ))
    sec(body(
        "The project demonstrates end-to-end ML engineering - from raw dataset ingestion to "
        "a live, explainable prediction served over HTTP, complete with fact-check evidence "
        "retrieved from a local knowledge base.", styles
    ))
    sec(info_box(
        "Key differentiator: TruthLens does not just output fake/real. It tells the user "
        "WHY - via highlighted flagged phrases and retrieved supporting evidence.", styles
    ))
    sp()
    sec(section_header("Core Capabilities", styles))
    sec(bullets([
        "Multi-tier inference: classical SVM, BiLSTM/GRU deep learning, DistilBERT transformer",
        "Automatic routing: short text (<200 chars) goes to headline-tuned SVM; longer text to full-text GRU",
        "Uncertain-band logic (35-65% probability zone returns 'uncertain' instead of forcing a label)",
        "Gradient saliency XAI for neural models; coefficient extraction for SVM",
        "Local RAG engine retrieves top-3 relevant fact-check articles via TF-IDF cosine similarity",
        "FastAPI REST backend with automatic OpenAPI documentation",
        "React+Babel frontend - no build step, works directly in the browser",
        "Dockerized - one command deploys the full stack",
        "43 passing pytest tests covering every module",
    ], styles))
    story.append(PageBreak())

    # ── 2. PROBLEM STATEMENT ────────────────────────────────────────────────
    story.append(chapter_header("Problem Statement", "2"))
    sp(10)
    sec(body(
        "Misinformation spreads faster than corrections. Existing fact-checking tools are "
        "either manual (too slow) or black-box binary classifiers (untrustworthy). "
        "TruthLens addresses three concrete problems:", styles
    ))
    sec(bullets([
        "Speed: Automated ML inference in milliseconds vs. hours for manual fact-checking.",
        "Explainability: Users need to understand what triggered the classification - not just a score.",
        "Evidence: Classification alone is insufficient; retrieved supporting evidence builds user trust.",
    ], styles))
    sp()
    sec(section_header("Dataset", styles))
    sec(kv_table([
        ("Name",    "WELFake_Dataset.csv"),
        ("Size",    "72,134 labeled articles"),
        ("Classes", "0 = Real, 1 = Fake"),
        ("Columns", "title, text, label"),
        ("Source",  "Verma et al., 2021 (IEEE TITS) - merged from 4 popular datasets"),
    ]))
    story.append(PageBreak())

    # ── 3. ARCHITECTURE ──────────────────────────────────────────────────────
    story.append(chapter_header("System Architecture", "3"))
    sp(10)
    sec(body(
        "TruthLens follows a layered pipeline architecture where each component has a single "
        "responsibility and can be replaced independently.", styles
    ))
    sec(code_block(
        "  Raw Text Input (HTTP POST /predict)\n"
        "        |\n"
        "        v\n"
        "  Preprocessing  (lowercasing, URL removal, dateline stripping)\n"
        "        |\n"
        "        |---> [classical]     TF-IDF Vectorizer --> Calibrated SVM\n"
        "        |                         +--> Coefficient XAI\n"
        "        |\n"
        "        |---> [deep_learning] Vocabulary Tokenizer --> Bidirectional GRU\n"
        "        |                         +--> Gradient Saliency XAI\n"
        "        |\n"
        "        +---> [transformer]   HuggingFace Tokenizer --> DistilBERT Fine-tuned\n"
        "                                  +--> Gradient Saliency XAI\n"
        "                                            |\n"
        "                                            v\n"
        "                            Uncertain-Band Thresholding (35-65% = 'uncertain')\n"
        "                                            |\n"
        "                                            v\n"
        "                           Local RAG Engine (TF-IDF cosine similarity)\n"
        "                           Retrieves top-3 fact-check articles\n"
        "                                            |\n"
        "                                            v\n"
        "                           PredictionResponse (JSON)\n"
        "                                            |\n"
        "                                            v\n"
        "                           TruthLens UI (React + Babel)", styles
    ))
    sp()
    sec(section_header("Automatic Model Routing", styles))
    sec(body(
        "When model_type='classical' is requested, the API inspects the character length of "
        "the input text. Inputs shorter than 200 characters are treated as headlines and routed "
        "to a headline-specific SVM. Longer inputs go to the full-article SVM. This dual-model "
        "approach avoids performance degradation on very short, ambiguous headlines.", styles
    ))
    story.append(PageBreak())

    # ── 4. ML PIPELINE ──────────────────────────────────────────────────────
    story.append(chapter_header("ML Pipeline - Stage by Stage", "4"))
    sp(10)

    sec(section_header("Stage 1 - Baseline (TF-IDF + Logistic Regression)", styles))
    sec(body(
        "The baseline establishes a performance floor. Text is cleaned, tokenized, and "
        "transformed into a TF-IDF matrix. A Logistic Regression with L2 penalty is trained "
        "and evaluated. This stage takes minutes to run and achieves ~94% accuracy - a "
        "surprisingly strong baseline that highlights the power of bag-of-words representations.", styles
    ))
    sec(bullets([
        "TF-IDF: max 100,000 features, (1,2)-gram range, sublinear TF scaling",
        "Logistic Regression: C=1.0, max_iter=1000, solver='lbfgs'",
        "Accuracy: ~94.1%",
    ], styles))
    sp()

    sec(section_header("Stage 2 - Classical Model Comparison", styles))
    sec(body(
        "Five classical models are compared under identical preprocessing: Logistic Regression, "
        "Naive Bayes, Linear SVM, Random Forest, and XGBoost. A Calibrated Linear SVM is "
        "selected for deployment based on F1 score, calibration quality, and inference speed.", styles
    ))
    sec(metrics_table(
        ["Model", "Accuracy", "F1 Score", "ROC AUC"],
        [
            ["Logistic Regression",   "94.1%", "94.0%", "0.990"],
            ["Naive Bayes",           "89.3%", "89.1%", "0.963"],
            ["Linear SVM (deployed)", "96.2%", "96.1%", "0.994"],
            ["Random Forest",         "93.8%", "93.7%", "0.988"],
            ["XGBoost",               "94.7%", "94.6%", "0.991"],
        ]
    ))
    sp()

    sec(section_header("Stage 3 - Deep Learning (Bidirectional GRU)", styles))
    sec(body(
        "A custom vocabulary is built from the training corpus and used to tokenize text "
        "into integer sequences padded to 300 tokens. A Bidirectional GRU processes the "
        "sequence in both directions. The final hidden state is concatenated, passed through "
        "dropout, and projected to binary output via sigmoid activation.", styles
    ))
    sec(bullets([
        "Embedding dim: 128 | Hidden dim: 256 (128 each direction) | Dropout: 0.3",
        "Optimizer: Adam (lr=3e-4) | Batch size: 64 | Epochs: 10 with early stopping",
        "Accuracy: ~97.1% | F1: ~97.0%",
    ], styles))
    sp()

    sec(section_header("Stage 4 - Transformer (DistilBERT)", styles))
    sec(body(
        "DistilBERT-base-uncased is fine-tuned for binary classification using HuggingFace "
        "Transformers. The [CLS] token representation is fed to a linear classification head. "
        "Mixed precision (fp16) training is used when a GPU is available.", styles
    ))
    sec(bullets([
        "Model: distilbert-base-uncased (66M parameters)",
        "Max sequence length: 256 tokens | Learning rate: 2e-5 | Warmup steps: 500",
        "Accuracy: ~98.2% | F1: ~98.1%",
    ], styles))
    story.append(PageBreak())

    # ── 5. XAI ─────────────────────────────────────────────────────────────
    story.append(chapter_header("Explainable AI (XAI)", "5"))
    sp(10)
    sec(body(
        "TruthLens provides human-readable explanations for every prediction. "
        "The strategy varies by model family.", styles
    ))
    sec(section_header("Classical Models - Coefficient Extraction", styles))
    sec(body(
        "For TF-IDF SVM models, the trained weight vector is a direct measure of feature "
        "importance. The top-K TF-IDF n-grams with the highest absolute weight contribution "
        "towards the predicted class are extracted and returned as flagged phrases. "
        "This is deterministic and fast (O(k log k)).", styles
    ))
    sp()
    sec(section_header("Neural Models - Gradient Saliency", styles))
    sec(body(
        "For GRU and DistilBERT models, gradient-based saliency is computed. The gradient "
        "of the predicted class score with respect to the input token embeddings is calculated "
        "via backpropagation. The L2 norm of the gradient at each token position gives its "
        "saliency score. Tokens with the highest saliency are returned as flagged phrases.", styles
    ))
    sec(info_box(
        "Both strategies return the same data format - a list of string phrases - so the "
        "frontend and API schema are completely agnostic to the model type being explained.", styles
    ))
    story.append(PageBreak())

    # ── 6. RAG ─────────────────────────────────────────────────────────────
    story.append(chapter_header("RAG Fact-Check Engine", "6"))
    sp(10)
    sec(body(
        "After classification, TruthLens runs a Retrieval-Augmented Generation (RAG) lookup "
        "against a curated local knowledge base of verified fact-check articles. This grounds "
        "the prediction in real-world evidence without requiring internet access or an LLM.", styles
    ))
    sec(section_header("How It Works", styles))
    sec(bullets([
        "Knowledge base: data/fact_checks.json - a JSON array of fact-check documents with title, verdict, source, url, and body.",
        "At startup: LocalRAG fits a TF-IDF vectorizer on all document bodies and stores the resulting document matrix.",
        "At inference: the input text is transformed by the same TF-IDF vectorizer, then cosine similarity is computed against all document vectors.",
        "Top-K (default 3) most similar documents above a threshold are returned as evidence.",
        "The evidence list is included in the API response JSON.",
    ], styles))
    sp()
    sec(section_header("Why TF-IDF Rather Than a Vector Database?", styles))
    sec(body(
        "A dense vector database (FAISS, Pinecone) would require an embedding model and "
        "significant memory. For this project's knowledge base size (~200 documents), "
        "sparse TF-IDF retrieval achieves equivalent recall with zero external dependencies "
        "and sub-millisecond query time.", styles
    ))
    story.append(PageBreak())

    # ── 7. REST API ─────────────────────────────────────────────────────────
    story.append(chapter_header("REST API", "7"))
    sp(10)
    sec(body(
        "The API is built with FastAPI and follows the OpenAPI 3.0 specification. All "
        "request/response schemas are defined as Pydantic v2 models for runtime validation "
        "and automatic documentation generation.", styles
    ))
    sec(section_header("Endpoints", styles))
    sec(kv_table([
        ("GET  /",        "Health check - returns loaded model names and routing config"),
        ("POST /predict", "Main inference endpoint - classifies text and returns full response"),
        ("GET  /docs",    "Interactive Swagger UI (auto-generated by FastAPI)"),
        ("GET  /ui",      "Serves the TruthLens React frontend"),
    ]))
    sp()
    sec(section_header("POST /predict - Request", styles))
    sec(code_block(
        '{\n'
        '  "text":               "string  (required) - article body or headline",\n'
        '  "title":              "string  (optional) - article title",\n'
        '  "model_type":         "classical | deep_learning | transformer",\n'
        '  "combine_title_text": "bool    (default: true)"\n'
        '}', styles
    ))
    sec(section_header("POST /predict - Response", styles))
    sec(code_block(
        '{\n'
        '  "label":            0 | 1 | -1,\n'
        '  "label_name":       "real" | "fake" | "uncertain",\n'
        '  "fake_probability": 0.97,\n'
        '  "real_probability": 0.03,\n'
        '  "model_tier":       "headline" | "article" | "deep_learning" | "transformer",\n'
        '  "model_type":       "classical" | "deep_learning" | "transformer",\n'
        '  "model_metadata":   { "model_name", "trained_at_utc", "dataset", "metrics" },\n'
        '  "flagged_phrases":  ["clickbait term", "conspiracy"],\n'
        '  "evidence": [\n'
        '    { "title": "...", "verdict": "False", "source": "Snopes",\n'
        '      "url": "https://...", "similarity_score": 0.89 }\n'
        '  ]\n'
        '}', styles
    ))
    story.append(PageBreak())

    # ── 8. UI ───────────────────────────────────────────────────────────────
    story.append(chapter_header("TruthLens UI", "8"))
    sp(10)
    sec(body(
        "The frontend is a single-page React application built with Babel standalone - "
        "no Node.js, no build step. It is served directly by FastAPI's StaticFiles mount at /ui.", styles
    ))
    sec(section_header("Key UI Features", styles))
    sec(bullets([
        "Animated scanning progress bar while waiting for the API response",
        "Verdict card - color-coded (red/green/amber) with large probability display",
        "Highlighted text panel - flagged phrases are underlined in the input text",
        "RAG Evidence panel - retrieved fact-check articles with verdict badges",
        "Model selector - switch between classical, GRU, and transformer at runtime",
        "Responsive layout for desktop and tablet",
    ], styles))
    story.append(PageBreak())

    # ── 9. PERFORMANCE ──────────────────────────────────────────────────────
    story.append(chapter_header("Model Performance & Benchmarks", "9"))
    sp(10)
    sec(section_header("Classification Metrics (WELFake, held-out 20% test set)", styles))
    sec(metrics_table(
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
        [
            ["SVM (classical)",    "96.2%", "96.1%", "96.3%", "96.1%", "0.994"],
            ["GRU (deep)",         "97.1%", "97.0%", "97.2%", "97.0%", "0.997"],
            ["DistilBERT (trans)", "98.2%", "98.2%", "98.2%", "98.1%", "0.999"],
        ]
    ))
    sp(8)
    sec(section_header("Inference Latency (single request, CPU)", styles))
    sec(metrics_table(
        ["Model", "Median Latency", "P99 Latency"],
        [
            ["SVM (classical)",   "< 5 ms",  "< 10 ms"],
            ["GRU (deep)",        "~120 ms", "~200 ms"],
            ["DistilBERT (trans)","~450 ms", "~700 ms"],
        ]
    ))
    sec(info_box(
        "The GRU provides the best accuracy/latency tradeoff for production and is the "
        "default model_type in the API.", styles
    ))
    story.append(PageBreak())

    # ── 10. TECH STACK ──────────────────────────────────────────────────────
    story.append(chapter_header("Tech Stack", "10"))
    sp(10)
    sec(kv_table([
        ("Python",         "3.12+ - primary language"),
        ("FastAPI",        "REST API framework with async support and auto OpenAPI docs"),
        ("Pydantic v2",    "Request/response schema validation"),
        ("scikit-learn",   "TF-IDF, SVM, Logistic Regression, evaluation metrics"),
        ("PyTorch",        "BiLSTM/GRU model definition and gradient saliency"),
        ("HuggingFace",    "DistilBERT tokenizer and fine-tuning (Trainer API)"),
        ("NLTK",           "Stopword removal, tokenization"),
        ("React + Babel",  "Frontend - no build step, served from FastAPI"),
        ("Docker",         "Containerization and deployment"),
        ("pytest",         "Test suite - 43 tests across all modules"),
        ("YAML",           "Configuration files - all hyperparameters externalized"),
    ]))
    story.append(PageBreak())

    # ── 11. PROJECT STRUCTURE ───────────────────────────────────────────────
    story.append(chapter_header("Project Structure", "11"))
    sp(10)
    sec(code_block(
        "fake-news-detection-system/\n"
        "|-- app/\n"
        "|   |-- main.py          # FastAPI app, lifespan, /predict endpoint\n"
        "|   +-- schemas.py       # Pydantic request/response models\n"
        "|-- src/\n"
        "|   |-- data/\n"
        "|   |   |-- clean.py     # Text cleaning and dateline stripping\n"
        "|   |   |-- load.py      # Dataset loading utilities\n"
        "|   |   +-- preprocess.py# Tokenization and preprocessing pipeline\n"
        "|   |-- features/\n"
        "|   |   +-- tfidf.py     # TF-IDF vectorizer training and persistence\n"
        "|   |-- models/\n"
        "|   |   |-- baseline.py  # Stage 1 Logistic Regression\n"
        "|   |   |-- classical.py # Stage 2 multi-model comparison\n"
        "|   |   |-- deep_learning.py  # GRU/BiLSTM architecture\n"
        "|   |   |-- transformer.py    # DistilBERT fine-tuning\n"
        "|   |   |-- inference.py # Unified inference for all model types\n"
        "|   |   +-- evaluate.py  # Metrics computation\n"
        "|   |-- explain/\n"
        "|   |   +-- explain.py   # XAI: coefficient extraction + gradient saliency\n"
        "|   +-- rag/\n"
        "|       +-- retriever.py # LocalRAG - TF-IDF fact-check retrieval\n"
        "|-- frontend/\n"
        "|   |-- index.html       # TruthLens React UI entry point\n"
        "|   +-- TruthLens.jsx    # React component tree\n"
        "|-- configs/             # YAML configs for each model stage\n"
        "|-- data/\n"
        "|   |-- raw/             # WELFake_Dataset.csv\n"
        "|   +-- fact_checks.json # RAG knowledge base\n"
        "|-- docs/                # Experiment reports per stage\n"
        "|-- notebooks/           # Jupyter notebooks (EDA, baselines, DL, transformers)\n"
        "|-- tests/               # pytest suite (43 tests)\n"
        "|-- utils/               # Config loader, logging, notebook helpers\n"
        "|-- Dockerfile\n"
        "|-- docker-compose.yml\n"
        "+-- README.md", styles
    ))
    story.append(PageBreak())

    # ── 12. SETUP ───────────────────────────────────────────────────────────
    story.append(chapter_header("Setup & Running Locally", "12"))
    sp(10)
    sec(section_header("1. Install", styles))
    sec(code_block(
        "git clone https://github.com/meetcodz/fake-news-detection-system.git\n"
        "cd fake-news-detection-system\n"
        'pip install -e ".[dev,notebook]"', styles
    ))
    sec(section_header("2. Train Models", styles))
    sec(code_block(
        "python -m src.models.train          # Stage 1 & 2 (classical, fast)\n"
        "python -m src.models.train_deep     # Stage 3 - GRU\n"
        "python -m src.models.train_transformer  # Stage 4 - DistilBERT", styles
    ))
    sec(section_header("3. Start API", styles))
    sec(code_block(
        "uvicorn app.main:app --reload\n"
        "# API:  http://127.0.0.1:8000\n"
        "# Docs: http://127.0.0.1:8000/docs\n"
        "# UI:   http://127.0.0.1:8000/ui", styles
    ))
    sec(section_header("4. Docker (one command)", styles))
    sec(code_block("docker compose up --build", styles))
    sec(section_header("5. Run Tests", styles))
    sec(code_block("python -m pytest tests/ -v --basetemp=./tmp_pytest", styles))
    story.append(PageBreak())

    # ── 13. INTERVIEW Q&A ───────────────────────────────────────────────────
    story.append(chapter_header("Interview Q&A Guide", "13"))
    sp(10)
    sec(body(
        "The following are the most commonly asked questions in ML engineering interviews "
        "for projects of this type, with recommended talking points.", styles
    ))

    qa = [
        (
            "Q: Why did you choose WELFake over other datasets?",
            "WELFake merges four independent datasets, which reduces source bias. With 72k "
            "samples it's large enough to evaluate neural models. It also has both title and "
            "body columns, letting me explore title-only vs. combined inputs."
        ),
        (
            "Q: What is TF-IDF and why use it as a baseline?",
            "TF-IDF weights each word by how often it appears in a document relative to how "
            "common it is across all documents. Common words like 'the' get near-zero weight; "
            "rare, discriminative words get high weight. It's interpretable, fast, and achieves "
            "surprisingly competitive accuracy - making it an ideal baseline."
        ),
        (
            "Q: Why a Bidirectional GRU instead of LSTM?",
            "GRUs have fewer parameters than LSTMs (no separate cell state), making them faster "
            "and less prone to overfitting. The bidirectional setup lets the model use both left "
            "and right context, which is important for understanding negation in news text."
        ),
        (
            "Q: How does your XAI work for neural models?",
            "I compute gradient saliency: backpropagate the predicted class score to the input "
            "embedding layer, then take the L2 norm of the gradient at each token. High gradient "
            "magnitude means that token had high influence. I return the top-K tokens as flagged "
            "phrases. No external library - just PyTorch's autograd."
        ),
        (
            "Q: What is RAG and how did you implement it without an LLM?",
            "RAG is the idea of grounding model outputs in retrieved documents. Normally RAG uses "
            "a dense embedding model. I implemented a lightweight version using sparse TF-IDF "
            "vectors: at startup, all fact-check docs are vectorized. At inference, cosine "
            "similarity finds the closest matching documents. No LLM, no API calls."
        ),
        (
            "Q: What is the uncertain-band logic?",
            "A classifier that outputs 0.51 should not be treated the same as one that outputs "
            "0.99. Predictions in the 35-65% probability zone return label=-1 and label_name='uncertain' "
            "instead of forcing a real/fake decision. This makes the system more honest about "
            "ambiguous inputs."
        ),
        (
            "Q: How is the project structured for maintainability?",
            "Each module has one responsibility (SOLID/SRP). Business logic lives exclusively in "
            "src/. The FastAPI app/ layer handles only HTTP concerns. All hyperparameters are in "
            "YAML configs - nothing is hardcoded. I can swap the model or change thresholds "
            "without touching any Python code."
        ),
        (
            "Q: What would you improve next?",
            "Three things: (1) Replace the local RAG knowledge base with a live fact-checking "
            "API (e.g. ClaimBuster) for real-time evidence. (2) Add PostgreSQL persistence to "
            "log predictions for drift monitoring. (3) Fine-tune DeBERTa-v3 - it consistently "
            "outperforms DistilBERT on NLI tasks and should push F1 to ~99%."
        ),
    ]

    for q, a in qa:
        story.append(KeepTogether([
            Paragraph(q, styles["qa_q"]),
            Paragraph(a, styles["qa_a"]),
            Spacer(1, 4),
        ]))

    doc.build(story)
    print(f"PDF written to: {output_path}")


if __name__ == "__main__":
    build_pdf("TruthLens_Project_Report.pdf")
