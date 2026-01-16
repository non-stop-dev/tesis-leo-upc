---
description: Workflow for writing academic paper sections on MSME survival analysis with GNN
---

# Academic Paper Writing Workflow

This workflow guides the LLM to write high-quality academic paper sections for the MSME survival prediction research project.

> [!CAUTION]
> ## ⚠️ CRITICAL: ACADEMIC INTEGRITY - NEVER FABRICATE DATA
> 
> **This is non-negotiable. Fabricating data in academic research is a serious offense that can result in:**
> - Academic expulsion
> - Professional sanctions
> - Legal consequences (fraud, falsification of documents)
> 
> ### STRICT RULES:
> 
> 1. **NEVER invent statistics, percentages, or numerical claims** (e.g., "studies show 40% improvement...")
> 2. **NEVER fabricate author names or citations** that don't exist
> 3. **NEVER attribute claims to sources without verifying** the source actually says that
> 4. **NEVER make up institutional reports** (e.g., fake INEI, World Bank, or OIT data)
> 
> ### IF YOU CANNOT VERIFY A CLAIM:
> 
> **Option A**: Do not include the claim at all. Skip it.
> 
> **Option B**: Write a placeholder with clear TODO marker:
> ```
> [TODO: buscar fuente de datos que respalde esta afirmación]
> ```
> 
> **Option C**: Rephrase as a hypothesis or theoretical expectation:
> ```
> "Teóricamente, se esperaría que la formalización mejore el acceso a crédito, 
> aunque esta relación requiere validación empírica con datos específicos."
> ```
> 
> ### VERIFICATION CHECKLIST:
> - [ ] Is this statistic from the original paper (`Entrega 5 Paper/Paper/`)?
> - [ ] Is this from `GEMINI.md` or user-provided data?
> - [ ] Can I find this in actual Stata output (`*.do` files)?
> - [ ] Is this a well-known published result I can cite properly?
> 
> **If NO to all above → DO NOT INCLUDE or mark as [TODO]**

---

## Step 0: Style Guidelines - Academic Economist

This is a formal scientific paper, not a blog post or technical documentation.

### 🚫 STRICTLY PROHIBITED:
1.  **Bold text for emphasis**: Do **NOT** use `**text**` to highlight concepts (e.g., **Productivity**, **Hypothesis**). Use italics `*text*` sparingly if absolutely necessary for definitions, but prefer plain text.
2.  **Excessive Bullet Points**: Do not use lists for explaining arguments. Use cohesive paragraphs with transition words (e.g., "Furthermore," "Conversely," "In this context").
3.  **Too Many Sub-headers**: Avoid fragmenting the text with H3/H4 headers every few paragraphs. Allow the argument to flow naturally.

### ✅ REQUIRED TONE:
*   **Formal & Objective**: Use passive voice where appropriate for method descriptions.
*   **Nuanced**: Avoid absolute terms ("always", "never"). Use "suggests", "indicates", "appears to".
*   **Cohesive**: Connect paragraphs logically. The text should look like a page from *Econometrica* or *The American Economic Review*.

---

## Project Context

- **Topic**: Effects of business formalization on MSME survival in Peru, extended with Graph Neural Networks
- **Previous Paper**: `Entrega 5 Paper/Paper/` (Logit-based analysis)
- **Current Paper**: `Entrega 6 (paper GNN)/paper-sections/` (GNN extension)
- **Context Guide**: `GEMINI.md` (read this first for full project context)

---

## Step 1: Read Context Files FIRST

Before writing any section, **always read**:

1. `GEMINI.md` - Project context, methodology, and GNN rationale
2. Previous paper sections in `Entrega 5 Paper/Paper/`:
   - `1. INTRODUCCIÓN.md`
   - `2. HECHOS ESTILIZADOS.md`
   - `3. METODOLOGÍA.md`
   - `4. CONCLUSIONES.md`
3. Existing GNN paper sections in `Entrega 6 (paper GNN)/paper-sections/`
4. `referencias_bibliograficas.md` - Check existing references to avoid duplicates

---

## Step 2: Source Quality Requirements

### REQUIRED: High-Quality Academic Sources

| Source Type | Examples | Priority |
|-------------|----------|----------|
| Peer-reviewed journals | Econometrica, AER, QJE, JMLR, NeurIPS | ✅ Highest |
| Conference proceedings | ICML, ICLR, KDD, NeurIPS | ✅ High |
| Books from academic publishers | Princeton UP, MIT Press, Cambridge | ✅ High |
| Working papers from institutions | NBER, World Bank, ILO, CEPAL | ⚠️ Acceptable |
| Government statistics | INEI, SUNAT, PRODUCE | ✅ For Peruvian data |

### AVOID: Low-Quality Sources

- Wikipedia, blogs, Medium articles
- Generic textbooks without specific citations
- Unpublished manuscripts without institutional affiliation
- Sources older than 20 years (unless seminal papers like Jovanovic 1982)

---

## Step 3: Cross-Reference with Previous Paper

**CRITICAL**: Before making any claim, check if the previous paper already cited relevant sources.

### Process:

1. Search `Entrega 5 Paper/Paper/*.md` for relevant citations
2. If a claim was made in the previous paper:
   - Use the SAME source cited there (maintains consistency)
   - Quote the specific finding (e.g., "Yamada (2009) finds 15% lower closure probability")
3. If adding a NEW claim:
   - Find a high-quality source to back it up
   - Do NOT make generic statements like "literature suggests..." without specific citations

### Example - GOOD vs BAD:

❌ **BAD (generic)**:
> "La formalización mejora la viabilidad empresarial según la literatura."

✅ **GOOD (specific with source)**:
> "Yamada (2009) encuentra una probabilidad de cierre 15% menor para firmas con RUC, mientras que Chacaltana (2016) reporta brechas de productividad de hasta ocho veces a favor de las empresas formales."

---

## Step 4: Citation Format (APA 7)

### Inline Citations

| Format | Example |
|--------|---------|
| Single author | Jovanovic (1982) |
| Two authors | Cameron y Miller (2015) |
| Three+ authors | Hamilton et al. (2017) |
| Multiple citations | (Kipf & Welling, 2017; Hamilton et al., 2017) |
| Direct quote | "exact quote" (Author, Year, p. X) |

### Reference File

**ALWAYS** add full references to:
```
Entrega 6 (paper GNN)/paper-sections/referencias_bibliograficas.md
```

Reference format (APA 7):
```markdown
Author, A. A., & Author, B. B. (Year). Title of article. *Journal Name*, *Volume*(Issue), pages. https://doi.org/xxxxx
```

---

## Step 5: Section Writing Checklist

Before submitting any section, verify:

- [ ] All claims have specific citations (no generic "literature suggests")
- [ ] Citations use APA 7 format
- [ ] All cited sources are added to `referencias_bibliograficas.md`
- [ ] Cross-referenced with previous paper sources
- [ ] No orphan citations (mentioned in text but not in references)
- [ ] No orphan references (in bibliography but not cited in text)

---

## Step 6: Key Authors by Topic

### Supervivencia Empresarial (Peru)
- Jovanovic (1982) - Firm selection model
- Chacaltana (2016) - Productivity gaps, formalization
- Yamada (2009) - Closure probability
- Aliaga (2017) - MSME credit access

### Econometría y Métodos
- Cameron & Miller (2015) - Cluster-robust inference
- Angrist & Pischke (2009) - Causal inference
- Wooldridge (2010) - Econometric methods

### Graph Neural Networks
- Kipf & Welling (2017) - GCN
- Hamilton et al. (2017) - GraphSAGE
- Gilmer et al. (2017) - Message passing
- Ying et al. (2018) - GNN embeddings

### Economía de Redes
- Jackson (2008) - Network economics
- Glaeser et al. (2010) - Urban economics
- Gibbons & Overman (2012) - Spatial econometrics

---

## File Locations

| Content | Path |
|---------|------|
| Previous paper | `Entrega 5 Paper/Paper/*.md` |
| Current paper sections | `Entrega 6 (paper GNN)/paper-sections/` |
| Bibliography | `Entrega 6 (paper GNN)/paper-sections/referencias_bibliograficas.md` |
| Project context | `GEMINI.md` |
| Stata analysis code | `Base de datos 5to censo económico anual/Base curada/*.do` |
