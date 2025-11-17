# PAPER GUIDELINES - WORKING PAPER
## Influence of Formalization on MYPE Survival in Peru 2022

**Course:** 1AEF0068  
**Format:** Working Paper  
**Total Pages:** 15-20 pages

---

## STRUCTURE & PAGE ALLOCATION

### 1. INTRODUCCIÓN (5 pages)
**Content Requirements:**

#### 1.1 Research Motivation
- **Economic Context:** MYPE importance in Peruvian economy (>95% of firms, 73.1% employment)
- **Problem Statement:** High informality (75%), COVID-19 impact, structural barriers
- **Policy Relevance:** Connection to ODS 8 (Decent Work and Economic Growth)
- **Research Gap:** Lack of regional heterogeneity analysis in post-COVID context with census data

#### 1.2 Literature Review (Estudios Previos)
**Key authors that validate the study:**

**Theoretical Foundation:**
- **Jovanovic (1982):** Industrial selection model - firms discover efficiency (θ) through market performance
- **Chacaltana (2016):** Formalization increases productivity 8x in Peru
- **Yamada (2009):** Formal MYPEs have 15% lower closure probability (HR=0.85)

**Regional Heterogeneity:**
- **Liedholm (2002):** Urban firms have 25% higher survival (Latin America/Africa)
- **Tonetto et al. (2024):** Primary product regions show higher survival (68-69%) vs metros (62%)
- **Carrión-Cauja (2021):** Tax effects vary by firm size in Ecuador

**Digital Transformation:**
- **Solomon et al. (2024):** Social media knowledge (OR: 2.89) and use (OR: 3.78) increase tech adoption
- **León & Valcárcel (2022):** Internet use increases log(profits) by 0.231 in Peru

**COVID-19 Impact:**
- **Varona & Gonzales (2021):** Elasticities -0.15 (short-run), -0.24 (long-run)
- **Yamada et al. (2023):** Labor rigidity reduces formal employment 0.9-2.2% per quarter

#### 1.3 Research Questions and Objectives
- **Main Question:** How does formalization influence MYPE survival in Peru 2022, and how do these effects vary by geographic region?
- **General Objective:** Analyze formalization's influence on MYPE survival
- **Specific Objectives:**
  1. Evaluate RUC effect on survival probability
  2. Determine regional variation (Costa, Sierra, Selva) in formalization effects

#### 1.4 Hypothesis
- **Central Hypothesis:** Formal MYPEs (with RUC) have higher survival probability
- **Specific Hypothesis:** Formalization effect is more pronounced in Costa due to better infrastructure, market access, and digital connectivity

#### 1.5 Brief Model Teórico Summary
- **Jovanovic (1982) Framework:** P(Survive) = F(θ, x_t)
- **Mechanisms:**
  - Formalization (RUC) reduces costs c(q_t, x_t) → increases π
  - Digitalization increases production q_t → increases π
  - Geographic location affects market access and infrastructure
  - Sector modulates competition and cost structures

---

### 2. APROXIMACIÓN METODOLÓGICA (7 pages)

#### 2.1 Data Source (½ page)
- **Source:** V Censo Nacional Económico 2022 (INEI)
- **Period:** April-August 2022, financial data for fiscal year 2021
- **Original Database:** 1.9+ million observations
- **Cleaned Database:** 1,377,931 MYPEs (96.6% micro, 3.4% small)
- **Regional Distribution:** 59.67% Costa, 32.47% Sierra, 7.87% Selva
- **Econometric Software:** Stata 17

#### 2.2 Variables Justification (1.5 pages)

**Dependent Variable:**
- **op2021_ajustado:** Binary (0=not operative, 1=operative in 2021)
- **Adjustment Logic:** Firms with sales >5000 soles marked as operative; firms with 0 sales marked as non-operative

**Independent Variable (Main):**
- **RUC:** Binary (0=without RUC/informal, 1=with RUC/formal)
- **Justification:** Core measure of formalization in Peru per Jovanovic framework

**Control Variables:**

*Geographic & Structural:*
- **region:** Categorical (0=Costa, 1=Sierra, 2=Selva) - captures infrastructure and market access differences
- **sector:** Categorical (0=Comercial, 1=Servicios, 2=Productivo) - competition intensity
- **tamano_empresa:** Binary (0=Micro ≤150 UIT, 1=Small >150 and ≤1700 UIT)

*Digital Transformation:*
- **digital_score:** Ordinal (0-3) - innovation capturing post-COVID digital intensity
  - 0 = no digital instruments
  - 1 = at least 1 instrument (website, Facebook, other social media)
  - 2 = at least 2 instruments
  - 3 = 3+ instruments
- **Justification:** Solomon et al. (2024) evidence of digital adoption impact

*Management & Demographics:*
- **sexo_gerente:** Binary (0=Woman, 1=Man) - Barriga et al. (2022) show efficiency and formalization gaps

*Economic Performance:*
- **productividad_k:** Labor productivity (VA/Workers)/1000 - Jovanovic's θ proxy
- **tributos_k:** Taxes paid/1000 - formalization cost indicator
- **salarios_k:** Salaries+benefits/1000 - operational commitment signal

*Operational:*
- **tipo_local:** Categorical (0=Own, 1=Rented, 2=Other) - stability signal
- **regimen:** Categorical tax regime (0=RUS, 1=RER, 2=RG, 3=RMT) - Carrión-Cauja (2021) differential effects

**Variables Excluded and Why:**
- **Sales (ventas_soles_2021):** Removed due to high collinearity with productivity (VIF issue)
- **Interest rates on loans:** Not available in dataset; only 6.68% of MYPEs access formal financing (Aliaga, 2017)
- **Recovered taxes & net profit:** Not relevant to Jovanovic survival framework

#### 2.3 Econometric Model (1.5 pages)

**Model Specification:**
```
logit op2021_ajustado c.ruc##(i.region) i.sector tamano_empresa sexo_gerente 
  productividad_k digital_score tributos_k salarios_k i.tipo_local 
  i.regimen, vce(cluster ciiu_2dig)
```

**Log-odds Equation:**
```
ln(P(Y=1)/(1-P(Y=1))) = β₀ + β₁RUC + β₂Sierra + β₃Selva + 
  β₄(RUC×Sierra) + β₅(RUC×Selva) + β₆Sector + β₇Size + 
  β₈Gender + β₉Productivity + β₁₀DigitalScore + 
  β₁₁Taxes + β₁₂Salaries + β₁₃LocalType + β₁₄Regime + ε
```

**Interpretation of Interaction Terms:**
- **β₁:** Effect of RUC in Costa (base region)
- **β₄:** DIFFERENCE in RUC effect between Sierra and Costa
- **β₅:** DIFFERENCE in RUC effect between Selva and Costa
- **Total Effect:**
  - In Costa: β₁
  - In Sierra: β₁ + β₄
  - In Selva: β₁ + β₅

**Model Justification:**
- **Why Logit:** Binary dependent variable (operative/non-operative)
- **Why Interactions:** Capture regional heterogeneity in formalization effects
- **Why Clustered SE:** Control for industry heterogeneity (ciiu_2dig)

#### 2.4 Problems Encountered & Solutions (2 pages)

**Problem 1: Variable Inconsistencies**
- **Issue:** Firms marked "non-operative" had sales >5000 soles; firms marked "operative" had 0 sales
- **Solution:** Created `op2021_ajustado` with correction threshold at 5000 soles
- **Robustness:** Threshold is arbitrary; sensitivity analysis recommended

**Problem 2: Negative Productivity Values**
- **Issue:** Mathematically impossible negative labor productivity
- **Solution:** Truncated negative values at minimum positive value
- **Justification:** Maintains data integrity without artificial imputation

**Problem 3: Extreme Outliers**
- **Issue:** Outliers distorting coefficient estimates
- **Solution:** Winsorization at 1% and 99% percentiles for continuous variables
- **Variables Affected:** productivity, tributos, salarios

**Problem 4: Multicollinearity**
- **Issue:** High VIF for sales and productivity (both measure firm size/performance)
- **Solution:** Removed sales variable, kept productivity
- **Justification:** Productivity is more theoretically grounded in Jovanovic framework (θ proxy)

**Problem 5: Magnitude Issues & Convergence**
- **Issue:** Large magnitudes (millions of soles) causing convergence problems
- **Solution:** Scaled by 1000 (productivity_k, tributos_k, salarios_k)
- **Why not log:** Avoid "percentage of percentage" interpretation in logit model

**Problem 6: Sample Imbalance**
- **Issue:** 96.6% microempresas vs 3.4% pequeñas empresas
- **Solution:** Controlled with `tamano_empresa` binary and clustered SE by industry
- **Limitation:** Acknowledged as potential bias source

#### 2.5 Diagnostic Tests (1 page)
- **Multicollinearity:** VIF test using auxiliary linear regression
- **Model Fit:** fitstat command (pseudo-R², McFadden, AIC, BIC)
- **Predictive Accuracy:** estat classification (sensitivity/specificity)
- **ROC Curve:** lroc for discriminatory power
- **Marginal Effects:** `margins region, dydx(ruc)` for economic interpretation

#### 2.6 Analysis of Tables (1.5 pages)
**DO NOT paste full 1.5-page tables. Instead:**

**Table Structure:**
1. **Descriptive Statistics Table (Summarized)**
   - Regional distribution of MYPEs
   - Formalization rates by region and sector
   - Digital score distribution
   - Mean productivity, taxes, salaries by region

2. **Logit Regression Results (Summarized)**
   - Present only key coefficients with interpretation:
     - RUC main effect (β₁)
     - Interaction terms (β₄, β₅)
     - Control variables (only significant ones)
   - Report: Coefficient, Odds Ratio, p-value, interpretation

3. **Marginal Effects Table (Summarized)**
   - Marginal effect of RUC by region
   - Confidence intervals
   - Economic interpretation (percentage point change)

**How to Summarize:**
- "Table X shows that formal MYPEs have 23.4 percentage points higher probability of survival in Costa (p<0.01), compared to 18.7 pp in Sierra (p<0.01) and 15.2 pp in Selva (p<0.05)."
- Focus on effect sizes, statistical significance, and economic meaning
- Use visual aids (simplified coefficient plots) if possible

---

### 3. CONCLUSIONES DEL ANÁLISIS ECONOMÉTRICO Y LIMITACIONES (3 pages)

#### 3.1 Conclusions (1.5 pages)

**Main Findings:**

1. **Formalization Effect (Overall):**
   - Report RUC coefficient magnitude, odds ratio, statistical significance
   - Economic interpretation: "Having RUC increases survival probability by X percentage points"
   - Connection to Jovanovic: "Formalization reduces transaction costs, increasing efficiency θ"

2. **Regional Heterogeneity:**
   - Compare total effects across Costa, Sierra, Selva
   - Test results: `test 1.region#c.ruc 2.region#c.ruc` (joint significance)
   - Interpretation: "Costa shows stronger formalization benefits due to better infrastructure, confirming hypothesis"
   - Sierra and Selva results: barriers limiting formalization effectiveness

3. **Control Variables (Key Findings Only):**
   - **Digital Score:** Effect size and significance → post-COVID digital transformation importance
   - **Productivity:** Positive effect confirming Jovanovic's θ mechanism
   - **Sector:** Differential survival by economic activity
   - **Gender:** Report if significant gap exists

4. **Validation Against Literature:**
   - Compare findings to Chacaltana (2016), Yamada (2009), Liedholm (2002)
   - Consistency or divergence explanations

#### 3.2 Limitations (¾ page)

**Data Limitations:**
1. **Cross-sectional Data:** Cannot establish causality definitively; only association
2. **Post-COVID Context:** 2021 data reflects pandemic recovery, not normal conditions
3. **Sample Imbalance:** 96.6% microempresas may bias results toward micro firm dynamics
4. **Survival Definition:** Binary operative/non-operative misses intensity (partially operative)
5. **Arbitrary Threshold:** 5000 soles correction threshold lacks theoretical foundation

**Methodological Limitations:**
1. **Omitted Variables:** Cannot control for entrepreneurial ability, management quality, social networks
2. **Selection Bias:** MYPEs that formalized may be inherently more efficient (self-selection)
3. **No Time Dimension:** Cannot track formalization→survival trajectory over time
4. **Regional Aggregation:** Costa/Sierra/Selva masks within-region heterogeneity

**Contextual Limitations:**
1. **Generalizability:** Results specific to Peru 2021-2022; may not apply to other countries/periods
2. **Informal Sector:** Cannot observe informal firms without census registration

#### 3.3 Policy Recommendations (¾ page)

**Evidence-Based Recommendations:**

1. **Differentiated Formalization Policies by Region:**
   - **Costa:** Leverage digital infrastructure for streamlined RUC registration
   - **Sierra:** Invest in institutional infrastructure and market access to maximize formalization benefits
   - **Selva:** Address connectivity barriers; consider mobile formalization units

2. **Digital Transformation Support:**
   - Given digital_score positive effect, subsidize digital adoption for MYPEs
   - Target: training in social media, e-commerce platforms (low-cost tools)

3. **Simplify Tax Regimes:**
   - Given regime effects, streamline RUS/RER/RMT to reduce compliance costs
   - Progressive taxation that accounts for firm size (micro vs small)

4. **Post-COVID Recovery Programs:**
   - Prioritize formalized MYPEs for credit access, technical assistance
   - Link formalization to pandemic relief programs to incentivize RUC registration

5. **Monitor Regional Disparities:**
   - Design SUNAT/PRODUCE programs that account for geographic heterogeneity
   - Regional targets: increase formalization effectiveness in Sierra/Selva

**Connection to ODS 8:**
- All recommendations promote decent work and sustainable economic growth
- Formalization as pathway to social security, labor rights, business stability

---

## FORMATTING GUIDELINES

### Page Layout
- **Font:** Times New Roman, 12pt
- **Line Spacing:** 1.5 or double-spaced
- **Margins:** 1 inch (2.54 cm) all sides
- **Page Numbers:** Bottom center

### Headings
- **Section 1 (INTRODUCCIÓN):** Bold, 14pt
- **Section 2 (APROXIMACIÓN METODOLÓGICA):** Bold, 14pt
- **Section 3 (CONCLUSIONES):** Bold, 14pt
- **Subsections:** Bold, 12pt

### Tables and Figures
- **Caption:** Above table/figure, italicized
- **Source:** Below table/figure, 10pt
- **Numbering:** Sequential (Table 1, Table 2...; Figure 1, Figure 2...)
- **In-Text Reference:** "As shown in Table X..." (not "see table below")

### Equations
- **Centered:** on separate line
- **Numbered:** right-aligned in parentheses if referenced
- **Variables:** Italicized

### Citations (In-Text)
- **Format:** (Author, Year) or Author (Year)
- **Multiple Authors:** (Author1 & Author2, Year) for 2; (Author1 et al., Year) for 3+
- **Multiple Works:** (Author1, Year1; Author2, Year2) chronological order

---

## 4. BIBLIOGRAFÍA (Unlimited - "Puede ser copia y pega de la tesis")

**Include all cited works from:**
- Introduction literature review
- Methodological approach references
- Theoretical framework citations

**Format:** APA 7th edition

**Key References to Include:**
- Jovanovic, B. (1982). Selection and the Evolution of Industry. *Econometrica*.
- Chacaltana, J. (2016). Formalización y productividad en Perú.
- Yamada, G. (2009). Supervivencia de microempresas formales en Perú.
- Liedholm, C. (2002). Small firm dynamics in Africa and Latin America.
- Tonetto, E., et al. (2024). Regional heterogeneity in firm survival.
- Solomon, M., et al. (2024). Digital adoption in SMEs.
- León, J., & Valcárcel, J. (2022). Internet use and profits in Peru.
- Varona, L., & Gonzales, J. (2021). COVID-19 elasticities on MYPEs.
- INEI (2022). V Censo Nacional Económico 2022.
- All other cited works in thesis

---

## CONTENT FROM THESIS SECTIONS

### Primary Source Sections:
1. **Entrega 3 (final)/Tesis por seccion/2. INTRODUCCIÓN.md**
   - Extract: Motivation, literature review, research questions, hypothesis
   - Condense from ~17 pages to 5 pages

2. **Entrega 3 (final)/Tesis por seccion/6. APROXIMACIÓN METODOLÓGICA.md**
   - Extract: Data source, variables, model specification, diagnostics
   - Condense econometric details, emphasize problems/solutions

3. **Entrega 3 (final)/Tesis por seccion/7. Conclusiones Preliminares.md** OR **conclusiones_preliminares_revisadas.md**
   - Extract: Main findings, limitations, policy recommendations
   - Add econometric results when available

4. **Entrega 3 (final)/Tesis por seccion/resultados_econometricos_para_tesis.md**
   - Extract: Regression results, marginal effects, hypothesis tests
   - Summarize tables as per guidelines

### Supporting Sections:
- **3. ESTUDIOS PREVIOS.md:** Synthesize key literature
- **4. MODELO TEÓRICO.md:** Condense Jovanovic framework
- **5. HECHOS ESTILIZADOS.md:** Regional context statistics
- **8. REFERENCIAS.md:** Copy-paste bibliography

---

## QUALITY CHECKLIST

### Content Requirements:
- [ ] Clear research motivation and policy relevance
- [ ] Comprehensive but concise literature review
- [ ] Well-defined research question and hypothesis
- [ ] Detailed variable justification with theoretical grounding
- [ ] Transparent discussion of problems encountered and solutions
- [ ] Summarized tables (NO 1.5-page table pastes)
- [ ] Economic interpretation of coefficients (not just statistical significance)
- [ ] Honest discussion of limitations
- [ ] Evidence-based policy recommendations linked to findings
- [ ] Connection to ODS 8 throughout

### Methodological Rigor:
- [ ] Jovanovic (1982) framework consistently applied
- [ ] Regional heterogeneity central to analysis
- [ ] Digital Score innovation explained and justified
- [ ] Interaction terms correctly interpreted (total effects calculated)
- [ ] Diagnostic tests reported
- [ ] Sample imbalance acknowledged and addressed

### Writing Quality:
- [ ] Concise and direct (15-20 pages strictly enforced)
- [ ] Academic tone, bilingual Spanish-English capacity
- [ ] No redundancy between sections
- [ ] Technical terms used correctly
- [ ] Peruvian context (RUC, SUNAT, INEI, UIT) properly explained

### Formatting:
- [ ] Page limits respected: Intro (5), Methodology (7), Conclusions (3)
- [ ] Tables and figures properly captioned and sourced
- [ ] Equations centered and numbered if referenced
- [ ] Citations in APA 7th format
- [ ] Bibliography complete

---

## KEY REMINDERS

1. **Total Length:** 15-20 pages (excluding bibliography)
2. **Focus:** Econometric analysis, not descriptive statistics
3. **Audience:** Academic (professor evaluation) - assume econometric literacy
4. **Tone:** Rigorous, evidence-based, policy-relevant
5. **Innovation:** Digital Score (0-3) as post-COVID digital intensity measure
6. **Core Contribution:** Regional heterogeneity in formalization effects using census data
7. **Context:** Post-COVID-19 Peru (2021 data), high informality (75%), ODS 8 alignment
8. **UIT 2021:** 4,400 soles (critical for size classification)
9. **Sample:** 1,377,931 MYPEs (96.6% micro, 3.4% small)
10. **Statistical Software:** Stata 17

---

## WORKFLOW RECOMMENDATION

1. **Draft Introducción (5 pages):**
   - Condense 2. INTRODUCCIÓN.md
   - Synthesize 3. ESTUDIOS PREVIOS.md into literature review
   - Extract hypothesis and objectives from Matriz de Consistencia

2. **Draft Aproximación Metodológica (7 pages):**
   - Copy data source from 6. APROXIMACIÓN METODOLÓGICA.md
   - Justify each variable with literature citations
   - Detail problems from Stata cleaning script (limpieza.do) and regression script (4.regresiones1.do)
   - Summarize tables from resultados_econometricos_para_tesis.md

3. **Draft Conclusiones (3 pages):**
   - Interpret regression results economically
   - Compare to literature expectations
   - List limitations transparently
   - Derive policy recommendations from findings

4. **Add Bibliografía:**
   - Copy-paste from 8. REFERENCIAS.md
   - Ensure all in-text citations included

5. **Review & Edit:**
   - Check page limits strictly
   - Remove redundancy
   - Verify all tables summarized (not pasted whole)
   - Confirm Jovanovic framework thread throughout

---

**END OF GUIDELINES**
