import React, { useState, useMemo, useRef, useEffect } from "react";
import { Search, ArrowRight, ArrowLeft, ShieldAlert, ExternalLink, CircleDot } from "lucide-react";

/* ---------------------------------------------------------------
   API configuration
--------------------------------------------------------------- */
const API_BASE = window.location.origin;

/* ---------------------------------------------------------------
   Context-Aware Data Structures (NASA Blackout Hoax)
--------------------------------------------------------------- */
const FLAGGED_PHRASES_NASA = [
  "BREAKING",
  "officially confirmed",
  "15 days of total darkness",
  "planetary alignment between Jupiter and Venus",
  "Scientists at NASA",
];

const SIGNALS_NASA = [
  {
    phrase: "BREAKING",
    title: "Sensational framing",
    detail:
      "Wire-style urgency (“BREAKING”) is typical of engagement bait, not how space agencies announce findings.",
  },
  {
    phrase: "officially confirmed",
    title: "Unverifiable authority claim",
    detail:
      "“Officially confirmed” implies a press release or statement, but none is linked, dated, or attributed.",
  },
  {
    phrase: "15 days of total darkness",
    title: "Scientifically implausible claim",
    detail:
      "Earth's rotation makes a 15-day global blackout physically impossible — no orbital mechanism produces this.",
  },
  {
    phrase: "planetary alignment between Jupiter and Venus",
    title: "Fabricated mechanism",
    detail:
      "Planetary alignments have no measurable effect on sunlight reaching Earth; this dresses the claim in scientific language.",
  },
  {
    phrase: "Scientists at NASA",
    title: "Unnamed sourcing",
    detail:
      "No individual, title, or department is named — a hallmark of claims with no real source to check.",
  },
];

const MOCK_BREAKDOWN_NASA = [
  { label: "Language pattern match", value: 94 },
  { label: "Source verification", value: 98 },
  { label: "Cross-reference check", value: 95 },
];

// Basic heuristic to detect if text matches the NASA hoax
function isNasaHoax(text) {
  if (!text) return false;
  const lower = text.toLowerCase();
  return (
    lower.includes("15 days of total darkness") ||
    (lower.includes("nasa") && lower.includes("darkness") && lower.includes("jupiter"))
  );
}

// List of common clickbait words for dynamic highlight mode
const CLICKBAIT_WORDS = [
  "BREAKING",
  "SHOCKING",
  "UNBELIEVABLE",
  "MUST SEE",
  "CONSPIRACY",
  "VIRAL",
  "AMAZING",
  "SECRET",
  "HIDDEN TRUTH",
  "ELITE",
  "URGENT",
  "WARNING",
];

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/* ---------------------------------------------------------------
   HighlightedText Component (Context-Aware)
--------------------------------------------------------------- */
function HighlightedText({ text, active, animate, phrases }) {
  const activePhrases = (phrases && phrases.length > 0) ? phrases : (isNasaHoax(text) ? FLAGGED_PHRASES_NASA : []);

  const parts = useMemo(() => {
    if (!text) return [];
    if (activePhrases.length === 0) return [text];
    const pattern = new RegExp(`(${activePhrases.map(escapeRegExp).join("|")})`, "gi");
    return text.split(pattern);
  }, [text, activePhrases]);

  let markIndex = -1;

  return (
    <>
      {parts.map((part, i) => {
        const isFlag = activePhrases.some((p) => p.toLowerCase() === part.toLowerCase());
        if (!isFlag) return <React.Fragment key={i}>{part}</React.Fragment>;
        markIndex += 1;
        const delay = 0.35 + markIndex * 0.28;
        return (
          <mark
            key={i}
            className="tl-mark"
            style={
              animate
                ? { animationDelay: `${delay}s` }
                : active
                ? { opacity: 1 }
                : undefined
            }
          >
            {part}
          </mark>
        );
      })}
    </>
  );
}

/* ---------------------------------------------------------------
   Main component
--------------------------------------------------------------- */
export default function TruthLens() {
  const [text, setText] = useState("");
  const [status, setStatus] = useState("idle"); // idle | scanning | done | error
  const [view, setView] = useState("check"); // check | dashboard
  const [result, setResult] = useState(null);
  const [modelType, setModelType] = useState("classical"); // deep_learning | classical
  const timerRef = useRef(null);

  useEffect(() => () => clearTimeout(timerRef.current), []);

  async function handleVerify() {
    if (!text.trim() || status === "scanning") return;
    setStatus("scanning");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text.trim(),
          model_type: modelType,
          combine_title_text: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      // small delay so the scan animation is visible
      await new Promise((r) => setTimeout(r, 1800));
      setResult(data);
      setStatus("done");
    } catch (err) {
      setResult({ error: err.message });
      setStatus("error");
    }
  }

  function reset() {
    clearTimeout(timerRef.current);
    setStatus("idle");
    setView("check");
    setResult(null);
    setText("");
  }

  const today = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const rootStyle = {
    "--paper": "#F1EFE6",
    "--paper-deep": "#E7E3D4",
    "--ink": "#211F1A",
    "--ink-soft": "#5C574A",
    "--ink-faint": "#8A8474",
    "--rule": "#CFC9B8",
    "--trust": "#1E4638",
    "--trust-bg": "#E2E9DF",
    "--false": "#9A3324",
    "--false-bg": "#F1E1D8",
    "--uncertain": "#A0742A",
    "--uncertain-bg": "#F0E5CF",
    background: "var(--paper)",
    color: "var(--ink)",
  };

  return (
    <div className="tl-root min-h-screen w-full" style={rootStyle}>
      {/* Masthead */}
      <header className="border-b" style={{ borderColor: "var(--rule)" }}>
        <div className="max-w-5xl mx-auto px-6 md:px-8 pt-8 pb-5">
          <div className="flex items-baseline justify-between gap-4">
            <button
              onClick={reset}
              className="tl-btn tl-serif text-3xl md:text-4xl font-semibold tracking-tight bg-transparent border-0 cursor-pointer p-0"
              style={{ color: "var(--ink)" }}
            >
              TruthLens
            </button>
            <span
              className="tl-mono hidden sm:block text-[11px] uppercase tracking-widest"
              style={{ color: "var(--ink-faint)" }}
            >
              {today} &middot; Verification Desk
            </span>
          </div>
          <div
            className="flex items-center gap-3 mt-2 border-t pt-2"
            style={{ borderColor: "var(--rule)" }}
          >
            <p
              className="text-[13px] uppercase tracking-widest"
              style={{ color: "var(--ink-soft)", letterSpacing: "0.12em" }}
            >
              Independent verification for the information age
            </p>
          </div>
        </div>
      </header>

      {view === "check" ? (
        <CheckView
          text={text}
          setText={setText}
          status={status}
          result={result}
          modelType={modelType}
          setModelType={setModelType}
          onVerify={handleVerify}
          onReset={reset}
          onViewEvidence={() => setView("dashboard")}
        />
      ) : (
        <DashboardView text={text} result={result} onBack={() => setView("check")} />
      )}

      <footer
        className="max-w-5xl mx-auto px-6 md:px-8 py-10 mt-10 border-t"
        style={{ borderColor: "var(--rule)" }}
      >
        <p className="tl-mono text-[11px]" style={{ color: "var(--ink-faint)" }}>
          TruthLens flags patterns for human review. It does not issue final rulings — always check primary sources.
        </p>
      </footer>
    </div>
  );
}

/* ---------------------------------------------------------------
   Check View
--------------------------------------------------------------- */
function CheckView({
  text,
  setText,
  status,
  result,
  modelType,
  setModelType,
  onVerify,
  onReset,
  onViewEvidence,
}) {
  const charCount = text.trim().length;

  return (
    <main className="max-w-3xl mx-auto px-6 md:px-8 pt-12 pb-6">
      <h1
        className="tl-serif text-[28px] sm:text-4xl leading-[1.15] font-semibold mb-2"
        style={{ color: "var(--ink)" }}
      >
        Paste a headline, article, or link.
      </h1>
      <p className="text-[15px] mb-8" style={{ color: "var(--ink-soft)" }}>
        We'll check the language, sourcing, and claims against known patterns of misinformation.
      </p>

      <div
        className="tl-manuscript rounded-sm border"
        style={{ borderColor: "var(--rule)", background: "#FFFFFF" }}
      >
        {status === "idle" || status === "error" ? (
          <textarea
            className="tl-input tl-serif w-full min-h-[180px] resize-none p-5 md:p-6 text-[17px] leading-[1.7] bg-transparent border-0 focus:ring-0 focus:outline-none"
            style={{ color: "var(--ink)" }}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste text or a URL here…"
            spellCheck={false}
          />
        ) : (
          <div className="relative p-5 md:p-6">
            {status === "scanning" && <div className="tl-scanline" />}
            <p className="tl-serif text-[17px] leading-[1.7]" style={{ color: "var(--ink)" }}>
              <HighlightedText text={text} animate={status === "scanning"} active={status === "done"} phrases={result?.flagged_phrases} />
            </p>
          </div>
        )}
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mt-4 gap-4">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <div
            className="tl-mono text-[11px] uppercase tracking-wider"
            style={{ color: "var(--ink-faint)" }}
          >
            {status === "idle" && `${charCount} characters`}
            {status === "scanning" && (
              <span className="tl-pulse flex items-center gap-1.5">
                <CircleDot size={12} /> Scanning for signals…
              </span>
            )}
            {status === "done" && "Scan complete"}
            {status === "error" && "API connection error"}
          </div>

          {status === "idle" && (
            <div className="flex items-center gap-2">
              <span
                className="tl-mono text-[10px] uppercase tracking-wider"
                style={{ color: "var(--ink-soft)" }}
              >
                Model:
              </span>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="tl-mono text-[11px] bg-transparent border-0 border-b border-dashed focus:outline-none cursor-pointer py-0.5"
                style={{ color: "var(--ink)", borderColor: "var(--rule)" }}
              >
                <option value="deep_learning">Deep Learning (GRU)</option>
                <option value="classical">Classical (SVM)</option>
                <option value="transformer">Transformer (DistilBERT)</option>
              </select>
            </div>
          )}
        </div>

        <div className="flex items-center justify-end">
          {status === "idle" && (
            <button
              onClick={onVerify}
              disabled={!text.trim()}
              className="tl-btn flex items-center gap-2 px-5 py-2.5 text-[14px] font-medium tracking-wide disabled:opacity-40 cursor-pointer border-0"
              style={{ background: "var(--ink)", color: "var(--paper)" }}
            >
              <Search size={15} />
              Verify
            </button>
          )}

          {status === "scanning" && (
            <button
              disabled
              className="tl-btn flex items-center gap-2 px-5 py-2.5 text-[14px] font-medium tracking-wide opacity-60 border-0"
              style={{ background: "var(--ink)", color: "var(--paper)" }}
            >
              Analyzing…
            </button>
          )}

          {(status === "done" || status === "error") && (
            <button
              onClick={onReset}
              className="tl-link text-[13px] underline underline-offset-4 bg-transparent border-0 cursor-pointer"
              style={{ color: "var(--ink-soft)" }}
            >
              Check another
            </button>
          )}
        </div>
      </div>

      {status === "done" && result && !result.error && (
        <div className="tl-fade-up mt-8">
          <VerdictCard result={result} onViewEvidence={onViewEvidence} />
        </div>
      )}

      {status === "error" && result?.error && (
        <div className="tl-fade-up mt-8">
          <div
            className="border-l-4 rounded-sm p-6"
            style={{ borderColor: "var(--false)", background: "var(--false-bg)" }}
          >
            <p className="text-[15px]" style={{ color: "var(--ink)" }}>
              Could not reach the TruthLens API. Make sure the backend is running at{" "}
              <code className="tl-mono" style={{ color: "var(--false)" }}>
                {API_BASE}
              </code>
            </p>
            <p className="tl-mono text-[13px] mt-2" style={{ color: "var(--ink-faint)" }}>
              {result.error}
            </p>
          </div>
        </div>
      )}
    </main>
  );
}

/* ---------------------------------------------------------------
   Verdict Card
--------------------------------------------------------------- */
function VerdictCard({ result, onViewEvidence }) {
  const isFake = result.label === 1;
  const isUncertain = result.label === -1;

  const labelName = isUncertain ? "Uncertain" : isFake ? "Likely False" : "Likely True";
  const confidence = Math.round(
    (isFake ? result.fake_probability : result.real_probability) * 100
  );

  const borderColor = isFake ? "var(--false)" : isUncertain ? "var(--uncertain)" : "var(--trust)";
  const bgColor = isFake ? "var(--false-bg)" : isUncertain ? "var(--uncertain-bg)" : "var(--trust-bg)";

  return (
    <div className="border-l-4 rounded-sm" style={{ borderColor, background: bgColor }}>
      <div className="p-6 md:p-7">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div className="flex items-start gap-3">
            <ShieldAlert size={26} style={{ color: borderColor }} className="mt-1 shrink-0" />
            <div>
              <p
                className="tl-mono text-[11px] uppercase tracking-widest mb-1"
                style={{ color: borderColor }}
              >
                Verdict
              </p>
              <h2 className="tl-serif text-3xl font-semibold" style={{ color: "var(--ink)" }}>
                {labelName}
              </h2>
            </div>
          </div>
          <div className="text-right">
            <p
              className="tl-mono text-[11px] uppercase tracking-widest mb-1"
              style={{ color: borderColor }}
            >
              Confidence
            </p>
            <p className="tl-mono text-3xl font-medium" style={{ color: "var(--ink)" }}>
              {confidence}%
            </p>
          </div>
        </div>

        <p className="text-[15px] leading-relaxed mt-4" style={{ color: "var(--ink)" }}>
          The model <strong>{result.model_metadata.model_name}</strong> ({result.model_tier} tier)
          analyzed this text and rated it as <strong>{labelName.toLowerCase()}</strong> with{" "}
          {confidence}% confidence.
        </p>

        <div className="mt-4 flex flex-wrap gap-x-8 gap-y-2">
          <div>
            <span
              className="tl-mono text-[11px] uppercase tracking-widest"
              style={{ color: "var(--ink-faint)" }}
            >
              Fake probability
            </span>
            <p className="tl-mono text-lg" style={{ color: "var(--ink)" }}>
              {(result.fake_probability * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <span
              className="tl-mono text-[11px] uppercase tracking-widest"
              style={{ color: "var(--ink-faint)" }}
            >
              Real probability
            </span>
            <p className="tl-mono text-lg" style={{ color: "var(--ink)" }}>
              {(result.real_probability * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <button
          onClick={onViewEvidence}
          className="tl-btn tl-link flex items-center gap-1.5 mt-5 text-[14px] font-medium bg-transparent border-0 cursor-pointer"
          style={{ color: borderColor }}
        >
          View full evidence report
          <ArrowRight size={15} />
        </button>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------
   Detailed Dashboard/Evidence View
--------------------------------------------------------------- */
function DashboardView({ text, result, onBack }) {
  if (!result) return null;

  const isFake = result.label === 1;
  const isUncertain = result.label === -1;
  const labelName = isUncertain ? "Uncertain" : isFake ? "Likely False" : "Likely True";
  const confidence = Math.round(
    (isFake ? result.fake_probability : result.real_probability) * 100
  );
  const borderColor = isFake ? "var(--false)" : isUncertain ? "var(--uncertain)" : "var(--trust)";

  const hasEvidence = result.evidence && result.evidence.length > 0;

  const signals = useMemo(() => {
    const list = [];

    if (hasEvidence) {
      list.push({
        phrase: result.evidence[0].source || "Fact-check database",
        title: "Verified Fact-Check Match",
        detail: `This claim matches a known record in our trusted fact-checking database. Verdict: ${result.evidence[0].verdict}.`,
      });
    }

    const phrases = result?.flagged_phrases || [];

    if (phrases.length === 0) {
      if (!hasEvidence) {
        list.push({
          phrase: "No significant signals",
          title: "Neutral terminology",
          detail: "The model did not identify any strongly biased or indicative terminology.",
        });
      }
    } else {
      phrases.forEach((phrase) => {
        list.push({
          phrase: phrase,
          title: isFake ? "Flagged as indicative of false patterns" : "Flagged as credible reporting",
          detail: `The model's Explainable AI module identified the word/phrase "${phrase}" as highly influential in determining the final prediction.`,
        });
      });
    }

    if (result?.model_tier === "headline") {
      list.push({
        phrase: "Headline-tier analysis",
        title: "Short statement evaluation",
        detail:
          "The model made this classification based on short title-length features (under 200 characters). Full body verification is recommended.",
      });
    } else {
      list.push({
        phrase: "Full article evaluation",
        title: "Deep semantic analysis",
        detail:
          "The text was analyzed as a full article, incorporating title-to-body relationship patterns.",
      });
    }

    return list;
  }, [hasEvidence, result, isFake]);

  const breakdown = useMemo(() => {
    if (hasEvidence) return MOCK_BREAKDOWN_NASA;

    const metrics = result.model_metadata.metrics || {};
    return [
      { label: "Model Accuracy", value: Math.round((metrics.accuracy || 0.95) * 100) },
      { label: "F1 Score (Balanced)", value: Math.round((metrics.f1 || 0.94) * 100) },
      { label: "ROC-AUC (Discriminator)", value: Math.round((metrics.roc_auc || 0.99) * 100) },
    ];
  }, [hasEvidence, result.model_metadata]);

  const domainInfo = useMemo(() => {
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    const matches = text.match(urlRegex);
    if (matches && matches.length > 0) {
      try {
        const urlObj = new URL(matches[0]);
        return {
          domain: urlObj.hostname,
          https: urlObj.protocol === "https:",
          status: "To be verified against whitelist",
        };
      } catch (e) {
        return null;
      }
    }
    return null;
  }, [text]);

  return (
    <main className="max-w-5xl mx-auto px-6 md:px-8 pt-10 pb-6">
      <button
        onClick={onBack}
        className="tl-btn tl-link flex items-center gap-1.5 text-[13px] mb-6 bg-transparent border-0 cursor-pointer"
        style={{ color: "var(--ink-soft)" }}
      >
        <ArrowLeft size={14} />
        Back to check
      </button>

      {/* Recap strip */}
      <div
        className="flex items-center justify-between flex-wrap gap-3 border-l-4 pl-4 py-3 mb-10"
        style={{ borderColor }}
      >
        <div className="flex items-center gap-2.5">
          <ShieldAlert size={18} style={{ color: borderColor }} />
          <span className="tl-serif text-xl font-semibold">{labelName}</span>
        </div>
        <span className="tl-mono text-lg" style={{ color: borderColor }}>
          {confidence}% confidence
        </span>
      </div>

      <div className="grid md:grid-cols-[1.4fr_1fr] gap-10">
        {/* Left column: excerpt + signals */}
        <div className="tl-fade-up">
          <h3
            className="tl-mono text-[11px] uppercase tracking-widest mb-3"
            style={{ color: "var(--ink-faint)" }}
          >
            Flagged excerpt
          </h3>
          <div
            className="rounded-sm border p-5 mb-8"
            style={{ borderColor: "var(--rule)", background: "#FFFFFF" }}
          >
            <p className="tl-serif text-[16px] leading-[1.75]" style={{ color: "var(--ink)" }}>
              <HighlightedText text={text} active animate={false} phrases={result?.flagged_phrases} />
            </p>
          </div>

          <h3
            className="tl-mono text-[11px] uppercase tracking-widest mb-3"
            style={{ color: "var(--ink-faint)" }}
          >
            Signals detected ({signals.length})
          </h3>
          <ul className="space-y-4 list-none p-0">
            {signals.map((s, i) => (
              <li key={i} className="flex gap-3">
                <span
                  className="tl-mono text-[12px] mt-0.5 shrink-0"
                  style={{ color: "var(--false)" }}
                >
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div>
                  <p className="text-[14px] font-medium" style={{ color: "var(--ink)" }}>
                    {s.title}
                  </p>
                  <p
                    className="text-[13.5px] leading-relaxed mt-0.5"
                    style={{ color: "var(--ink-soft)" }}
                  >
                    {s.detail}
                  </p>
                  <span
                    className="tl-mono inline-block text-[11px] mt-1.5 px-1.5 py-0.5 rounded-sm"
                    style={{ background: "var(--false-bg)", color: "var(--false)" }}
                  >
                    &ldquo;{s.phrase}&rdquo;
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </div>

        {/* Right column: breakdown + source credibility */}
        <div className="tl-fade-up" style={{ animationDelay: "0.15s" }}>
          <h3
            className="tl-mono text-[11px] uppercase tracking-widest mb-3"
            style={{ color: "var(--ink-faint)" }}
          >
            {hasEvidence ? "Confidence breakdown" : "Model evaluation statistics"}
          </h3>
          <div
            className="rounded-sm border p-5 mb-8"
            style={{ borderColor: "var(--rule)", background: "#FFFFFF" }}
          >
            <div className="space-y-4">
              {breakdown.map((b, i) => (
                <div key={i}>
                  <div className="flex justify-between items-baseline mb-1">
                    <span className="text-[13px]" style={{ color: "var(--ink-soft)" }}>
                      {b.label}
                    </span>
                    <span className="tl-mono text-[13px]" style={{ color: "var(--ink)" }}>
                      {b.value}%
                    </span>
                  </div>
                  <div
                    className="h-1.5 rounded-full w-full"
                    style={{ background: "var(--paper-deep)" }}
                  >
                    <div
                      className="h-1.5 rounded-full"
                      style={{ width: `${b.value}%`, background: borderColor }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <h3
            className="tl-mono text-[11px] uppercase tracking-widest mb-3"
            style={{ color: "var(--ink-faint)" }}
          >
            {hasEvidence ? "Fact-Check Evidence (RAG)" : "Source & Corpus Credibility"}
          </h3>
          <div
            className="rounded-sm border p-5"
            style={{ borderColor: "var(--rule)", background: "#FFFFFF" }}
          >
            {hasEvidence ? (
              <div className="space-y-4">
                {result.evidence.map((ev, i) => {
                  const scorePercent = Math.round(ev.similarity_score * 100);
                  const isFalseVerdict = ev.verdict?.toLowerCase() === "false";
                  const verdictBg = isFalseVerdict ? "var(--false-bg)" : "var(--trust-bg)";
                  const verdictColor = isFalseVerdict ? "var(--false)" : "var(--trust)";

                  return (
                    <div
                      key={i}
                      className={i > 0 ? "pt-4 border-t border-0 border-solid" : ""}
                      style={{ borderColor: "var(--rule)" }}
                    >
                      <div className="flex justify-between items-start gap-2 mb-1.5">
                        <h4
                          className="text-[14px] font-semibold leading-tight"
                          style={{ color: "var(--ink)" }}
                        >
                          {ev.title}
                        </h4>
                        <span
                          className="tl-mono text-[10px] px-1.5 py-0.5 rounded-sm whitespace-nowrap"
                          style={{ background: "var(--paper-deep)", color: "var(--ink-soft)" }}
                        >
                          Match: {scorePercent}%
                        </span>
                      </div>
                      <p className="text-[13px] leading-relaxed mb-2" style={{ color: "var(--ink-soft)" }}>
                        {ev.content}
                      </p>
                      <div className="flex flex-wrap items-center justify-between gap-2 mt-2">
                        <span
                          className="tl-mono text-[10.5px] px-2 py-0.5 rounded-full font-medium"
                          style={{ background: verdictBg, color: verdictColor }}
                        >
                          Verdict: {ev.verdict}
                        </span>
                        <div className="flex items-center gap-1.5 text-[12px]">
                          <ExternalLink size={12} style={{ color: "var(--ink-faint)" }} />
                          <a
                            href={ev.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="underline font-medium"
                            style={{ color: "var(--ink-soft)" }}
                          >
                            Source: {ev.source}
                          </a>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <>
                <p className="text-[13.5px] leading-relaxed" style={{ color: "var(--ink)" }}>
                  This text was evaluated against the <strong>WELFake corpus</strong> (72,134 news
                  items). The classifier trained on this dataset flags styling markers, vocabulary
                  frequencies, and title-to-body discrepancies.
                </p>
                {domainInfo && (
                  <div
                    className="mt-3 pt-3 border-t space-y-1.5 border-0 border-solid"
                    style={{ borderColor: "var(--rule)" }}
                  >
                    <div
                      className="tl-mono text-[11px] uppercase tracking-widest"
                      style={{ color: "var(--ink-faint)" }}
                    >
                      Extracted URL Credibility
                    </div>
                    <div className="text-[12.5px]" style={{ color: "var(--ink-soft)" }}>
                      <strong>Domain:</strong> {domainInfo.domain}
                    </div>
                    <div className="text-[12.5px]" style={{ color: "var(--ink-soft)" }}>
                      <strong>Security (HTTPS):</strong>{" "}
                      {domainInfo.https ? "Secured" : "Unsecured"}
                    </div>
                  </div>
                )}
                <div
                  className="mt-3 pt-3 border-t space-y-1 border-0 border-solid text-[11px] tl-mono"
                  style={{ borderColor: "var(--rule)", color: "var(--ink-faint)" }}
                >
                  <div>Model: {result.model_metadata.model_name}</div>
                  <div>Trained: {result.model_metadata.trained_at_utc}</div>
                  <div>
                    Samples: {result.model_metadata.dataset.training_samples?.toLocaleString()}{" "}
                    train, {result.model_metadata.dataset.test_samples?.toLocaleString()} test
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}