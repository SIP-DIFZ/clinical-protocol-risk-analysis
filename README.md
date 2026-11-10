# Clinical Protocol Risk Analyzer

Python app (CLI + Web UI) that analyzes large clinical trial protocol documents and identifies:

- **Health risks** (patient safety, intervention burden, monitoring gaps, etc.)
- **Financial risks** (recruitment cost risk, timeline slippage, site burden, operational overhead, etc.)
- **Ethical risks** (consent quality, vulnerable populations, fairness, privacy, placebo ethics)
- **Eligibility risks** (restrictive criteria, biomarker/lab thresholds, treatment history gates)
- **Intervention risks** (first-in-human complexity, escalation, procedures, administration burden)

Outputs per run:

- `analysis.json` (structured machine-readable risk data)
- `report.md` (executive summary)
- `report.pdf` (executive summary PDF)
- `mitigated_protocol.md` (protocol draft with colored edits)
- `mitigated_protocol.pdf` (protocol draft PDF with colored edits)
- `mitigated_protocol.docx` (protocol draft DOCX with colored edits)
- `mitigated_protocol_patches.txt` (Editor Agent patch log)

Each risk now includes `factor_triggers` aligned to explicit health and financial rubric factors (e.g., first-in-human exposure, invasive procedures, restrictive eligibility, anticipated dropout).

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create environment file:

```powershell
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

## 2) Run Web App (recommended)

```powershell
python -m streamlit run .\protocol_risk_analyzer\web_app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

Web app features:

- Upload `.pdf`, `.txt`, or `.docx` protocol
- Configure model/chunking settings (model defaults to `OPENAI_MODEL` or `gpt-4.1-mini`)
- Run analysis
- Live analysis progress indicator with percent complete and estimated remaining time
- Interactive dashboard with Health/Financial/Ethical/Eligibility/Intervention tabs, filters, and drill-down details
- Download `analysis.json`, `report.md`, and `report.pdf`
- Mitigated Protocol tab: select all or specific risks, run Editor Agent patch mode, and download `.md`, `.pdf`, `.docx` revised protocol outputs

The analysis pipeline uses a multi-agent flow:

- Chunk orchestrator (parallel task dispatch)
- Health risk agent (participant safety)
- Financial risk agent (funding/feasibility)
- Ethical risk agent (ethics/compliance burden)
- Eligibility risk agent (recruitment feasibility constraints)
- Intervention risk agent (intervention complexity and execution risks)
- Judge/aggregator agent (final duplicate/near-duplicate merge only)
- Editor Agent (risk-selected protocol patch generation with tone/style matching)

## 3) Run CLI analysis

```powershell
python -m protocol_risk_analyzer analyze ".\sample_protocol.pdf" --output-dir ".\out"
```

Or via console script:

```powershell
protocol-risk-analyzer analyze ".\sample_protocol.pdf" --output-dir ".\out"
```

## 4) CLI options

```powershell
python -m protocol_risk_analyzer analyze --help
```

Key options:

- `--model`: LLM model name (default from `OPENAI_MODEL` or `gpt-4.1-mini`)
- `--chunk-chars`: approximate chunk size in characters
- `--overlap-chars`: overlap between adjacent chunks

Optional environment variable:

- `ANALYSIS_MAX_WORKERS`: controls parallel agent worker count (default auto, capped for stability)
- `OPENAI_TIMEOUT_SECONDS`: per-request timeout used for LLM calls (default `120`)
- `OPENAI_MAX_RETRIES`: retry attempts for transient LLM/API failures (default `2`)

## Supported document types

- `.pdf`
- `.txt`
- `.docx`

### DOCX analysis note

- When a `.docx` file is uploaded, the analyzer converts it to PDF first for risk extraction consistency.
- The mitigated protocol generation still uses the original `.docx` to preserve editable document formatting.
- DOCX-to-PDF conversion uses Microsoft Word via `docx2pdf`.

## Notes for very large protocols

- The analyzer chunks documents page-aware, then extracts risks per chunk.
- Final risk list is aggregated and de-duplicated by risk title + domain.
- Keep chunk sizes moderate (10k-20k characters) for accuracy and cost control.

## Disclaimer

This tool supports risk screening and prioritization, not medical or legal advice. Final decisions should involve qualified clinical, biostatistical, regulatory, and legal experts.
