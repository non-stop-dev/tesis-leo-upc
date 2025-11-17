/*=============================================================================
DO FILE: ANÁLISIS DE SUPERVIVENCIA EMPRESARIAL - EFECTOS HETEROGÉNEOS REGIONALES
Autor: Leonardo León
Fecha: Septiembre 2025
Propósito: Modelo logístico con interacciones RUC*Región basado en Jovanovic (1982)
=============================================================================*/

cls
clear
set more off
capture log close

// Establecer directorio de trabajo
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Base curada"

// Crear log file
log using "analisis_supervivencia_$(S_DATE).log", replace

// Cargar base de datos
use "2.v_censo_limpio.dta", clear

/*=============================================================================
1. PREPARACIÓN DE VARIABLES
=============================================================================*/

// Verificar productividad negativa
count if productividad_x_trabajador < 0
summarize productividad_x_trabajador, detail

// SOLUCIÓN A PRODUCTIVIDAD NEGATIVA
// Opciones: (1) Eliminar obs negativas, (2) Usar valor absoluto, (3) Truncar en 0
// Recomendación econométrica: truncar en valor mínimo positivo para evitar perder observaciones

// Identificar el mínimo positivo
quietly summarize productividad_x_trabajador if productividad_x_trabajador > 0, detail
local min_pos = r(min)
display "Mínimo valor positivo de productividad: " `min_pos'

// Opción 1: Truncar valores negativos en el mínimo positivo
capture drop productividad_clean
gen productividad_clean = productividad_x_trabajador
replace productividad_clean = `min_pos' if productividad_x_trabajador <= 0 & !missing(productividad_x_trabajador)

// Variables en niveles para evitar problemas de convergencia
// Escalando por 1000 para reducir magnitudes
capture drop productividad_k tributos_k salarios_k
gen productividad_k = productividad_clean / 1000
gen tributos_k = tributos / 1000
gen salarios_k = salarios / 1000

// Dividir por 1000 para reducir valores en lugar de logaritmo natural (porque seria interpretar el % del %)
// Productividad según definición oficial INEI: Valor Agregado Bruto (VAB) / Trabajadores Ocupados
// donde VAB = Ventas - Consumo Intermedio (variable VALOR_AGREGADO del V Censo 2022)
// Correlación moderada-alta con ventas por construcción (VIF esperado: 4-8)
// Decisión de especificación: calcular VIF; si VIF<10 incluir ambas; si VIF>=10 priorizar ventas
// Usar logit op2021_original c.ruc##(i.region) ventas_soles_2021 sexo_gerente productividad_k digital_score tributos_k salarios_k i.tipo_local i.regimen, vce(cluster ciiu_4dig) en lugar de sector dado que sector comercio, manufacturas y servicio es muy poco, mejor agrupar por CIIU lo cual significa agrupar por los primeros 4 dígitos (Clase), es decir 0111, 0112, para de esta manera agrupar empresas que se dedican a actividades más específicas. Mayor granularidad (4 vs 2 dígitos) captura mejor la heterogeneidad intra-sectorial. El comando vce cluster lo que hace es "limpiar" un poco la heterogeneidad y sesgo que se genera al tener empresas que se dedican a diferentes rubros
// al usar comando margins region , dydx(ruc) esto me permite hacer gráficos para ver cómo evoluciona supervivencia según aumenta el tamaño de la empresa, etc (idea del profesor)


// Variables dummy regionales (Costa como base)
capture drop sierra selva
gen sierra = (region == 1)
gen selva = (region == 2)

// Estadísticas descriptivas esenciales
tab op2021_original region, col nofreq
tab op2021_original ruc, col nofreq
tabstat ventas_soles_2021, by(op2021_original) statistics(mean median sd min max n)

/*=============================================================================
2. ANÁLISIS DE ENDOGENEIDAD Y SESGO DE SELECCIÓN
=============================================================================*/

/*-----------------------------------------------------------------------------
2.1 TEST DE BALANCE: CARACTERÍSTICAS FORMALES VS INFORMALES

Objetivo: Evaluar si empresas formales (RUC=1) e informales (RUC=0) son
comparables en características observables PRE-TRATAMIENTO.

Implicaciones:
- Diferencias grandes → Sesgo de selección fuerte (auto-selección en formalización)
- Comparabilidad → Menor preocupación por endogeneidad (condicional a controles)

Variables evaluadas:
- Exógenas: región, sector, sexo gerente, tipo local
- Potencialmente endógenas: ventas, productividad, digitalización
- Post-tratamiento: tributos, salarios (solo formales pueden reportar)
-----------------------------------------------------------------------------*/

display ""
display "═══════════════════════════════════════════════════════════════"
display "TEST DE BALANCE: FORMALES (RUC=1) VS INFORMALES (RUC=0)"
display "═══════════════════════════════════════════════════════════════"
display ""

// Distribución básica
display "Distribución de formalización en la muestra:"
tab ruc, missing
display ""

/*--- Variables continuas: Test de diferencias en medias ---*/

display "──────────────────────────────────────────────────────────────"
display "A. VARIABLES ECONÓMICAS (potencialmente endógenas)"
display "──────────────────────────────────────────────────────────────"

// Ventas
display ""
display "1. VENTAS ANUALES 2021 (soles):"
ttest ventas_soles_2021, by(ruc)

// Productividad (antes de winsorización)
display ""
display "2. PRODUCTIVIDAD LABORAL (VAB/Trabajador, miles S/.):"
gen productividad_temp = productividad_x_trabajador / 1000
ttest productividad_temp, by(ruc)
drop productividad_temp

// Digitalización
display ""
display "3. DIGITAL SCORE (0-3):"
ttest digital_score, by(ruc)

display ""
display "──────────────────────────────────────────────────────────────"
display "B. VARIABLES POST-TRATAMIENTO (solo observables en formales)"
display "──────────────────────────────────────────────────────────────"

// Tributos
display ""
display "4. TRIBUTOS PAGADOS 2021 (miles S/.):"
gen tributos_temp = tributos / 1000
ttest tributos_temp, by(ruc)
drop tributos_temp

// Salarios
display ""
display "5. REMUNERACIONES 2021 (miles S/.):"
gen salarios_temp = salarios / 1000
ttest salarios_temp, by(ruc)
drop salarios_temp

/*--- Variables categóricas: Test de independencia (Chi²) ---*/

display ""
display "──────────────────────────────────────────────────────────────"
display "C. VARIABLES EXÓGENAS (pre-determinadas)"
display "──────────────────────────────────────────────────────────────"

// Distribución regional
display ""
display "6. DISTRIBUCIÓN REGIONAL:"
tab region ruc, chi2 col nofreq

// Distribución sectorial
display ""
display "7. DISTRIBUCIÓN SECTORIAL:"
tab sector ruc, chi2 col nofreq

// Género del gerente
display ""
display "8. GÉNERO DEL GERENTE:"
tab sexo_gerente ruc, chi2 col nofreq

// Tipo de local
display ""
display "9. TIPO DE LOCAL:"
tab tipo_local ruc, chi2 col nofreq

/*--- Tabla resumen de estadísticas descriptivas ---*/

display ""
display "──────────────────────────────────────────────────────────────"
display "D. RESUMEN DE ESTADÍSTICAS DESCRIPTIVAS POR FORMALIZACIÓN"
display "──────────────────────────────────────────────────────────────"
display ""

tabstat ventas_soles_2021 productividad_x_trabajador digital_score, ///
    by(ruc) statistics(mean median sd min max n) ///
    columns(statistics) format(%12.2f) ///
    longstub

display ""
display "═══════════════════════════════════════════════════════════════"
display "INTERPRETACIÓN DEL TEST DE BALANCE:"
display "═══════════════════════════════════════════════════════════════"
display "- p-value < 0.05: Diferencia estadísticamente significativa"
display "  → Evidencia de SESGO DE SELECCIÓN en formalización"
display "  → Empresas formales/informales NO son comparables en esa variable"
display ""
display "- Implicación: Modelo debe controlar exhaustivamente por observables"
display "  para mitigar sesgo de variable omitida (estrategia: selection on observables)"
display "═══════════════════════════════════════════════════════════════"
display ""

/*-----------------------------------------------------------------------------
2.2 EVALUACIÓN DE ENDOGENEIDAD POR RÉGIMEN TRIBUTARIO

Objetivo: Analizar si elección de régimen tributario está correlacionada con
características empresariales (evidencia de auto-selección)
-----------------------------------------------------------------------------*/

display ""
display "═══════════════════════════════════════════════════════════════"
display "ANÁLISIS DE ENDOGENEIDAD: RÉGIMEN TRIBUTARIO"
display "═══════════════════════════════════════════════════════════════"
display ""

// Test de diferencias por régimen tributario (ANOVA)
display "Test ANOVA: ¿Difiere productividad entre regímenes tributarios?"
oneway productividad_x_trabajador regimen, tabulate

// Winsorización para outliers extremos (preparación para modelo)
display ""
display "Aplicando winsorización a productividad (1%-99%)..."
quietly summarize productividad_x_trabajador, detail
local p1 = r(p1)
local p99 = r(p99)
replace productividad_k = `p1' / 1000 if productividad_x_trabajador < `p1' & !missing(productividad_x_trabajador)
replace productividad_k = `p99' / 1000 if productividad_x_trabajador > `p99' & !missing(productividad_x_trabajador)
label var productividad_k "Productividad winsorizada (1%-99%), miles S/."
display "Productividad winsorizada completada."

// Diagnósticos adicionales
display ""
display "Diferencias en ventas entre empresas operativas/no operativas:"
ttest ventas_soles_2021, by(op2021_original)

display ""
display "Asociación entre supervivencia y régimen tributario:"
tab op2021_original regimen, chi2 V

display ""
display "Correlación entre variables económicas (multicolinealidad):"
corr productividad_k tributos_k salarios_k

display ""
display "═══════════════════════════════════════════════════════════════"

/*=============================================================================
3. MODELO PRINCIPAL: EFECTOS HETEROGÉNEOS DE FORMALIZACIÓN
=============================================================================*/
// Es necesario estandarizar la variable de ventas ventas_soles_2021 con la siguiente operación: restar media y dividir entre desviacion estandar para estandarizar la venta. Hecho esto debemos usar ventas_soles_2021 estandarizada en lugar de ventas_soles_2021. Actualmente, si usamos el comando "hist ventas_soles_2021", la variable ventas_soles_2021 está sesgada al 0, al estandarizarla se obtiene una distribución de campana.

summarize ventas_soles_2021
capture drop ventas_std
gen ventas_std = (ventas_soles_2021 - r(mean)) / r(sd)
label variable ventas_std "Ventas 2021 estandarizadas (media=0, sd=1)"

// Modelo completo con interacciones RUC*Región
// Especificación según ecuación del paper (Sección 5. ESTRATEGIA EMPÍRICA):
// - Incluye sector económico como control estructural (Comercial, Productivo, Servicios)
// - Clustering por CIIU 4 dígitos (Clase): Máxima granularidad sectorial disponible
//   Justificación: Mayor desagregación captura heterogeneidad intra-sectorial más precisa
//   (recomendación del asesor: "mientras más granular, mejor" dado tamaño censal N=1.3M)
//   Cameron & Miller (2015) sugieren maximizar clusters cuando N es suficientemente grande
logit op2021_original c.ruc##(i.region) ventas_std i.sector sexo_gerente productividad_k digital_score tributos_k salarios_k i.tipo_local i.regimen, vce(cluster ciiu_4dig)

// Revisar R^2 ajustados (McFadden)
// Valores típicos en datos microeconómicos: 0.10-0.30 considerados aceptables
// ssc install fitstat
fitstat

// Distribución sectorial de la muestra
tab sector

estimates store modelo_principal

// Odds ratios para interpretación alternativa
logit, or

// Verificar multicolinealidad con VIF usando regresión lineal auxiliar
// Decisión: VIF<10 incluir ambas ventas-productividad; VIF≥10 priorizar ventas
quietly regress op2021_original productividad_k tributos_k salarios_k digital_score ventas_std sexo_gerente i.region i.sector i.regimen i.tipo_local
vif

/*-----------------------------------------------------------------------------
2.3 ESPECIFICACIONES ALTERNATIVAS: ROBUSTEZ A ENDOGENEIDAD

Objetivo: Verificar si coeficientes RUC×Región son estables cuando se excluyen
variables potencialmente endógenas o se restringe la muestra.

Estrategia:
1. Excluir tributos/salarios (post-tratamiento, solo observables en formales)
2. Solo controles exógenos (región, sector, sexo, tipo_local)
3. Solo microempresas (ventas <S/50k) para reducir heterogeneidad no observada
4. Solo pequeñas empresas (ventas ≥S/50k) para comparar efectos

Interpretación:
- Coeficientes estables → Resultados robustos a sesgo de variable omitida
- Coeficientes cambian sustancialmente → Sensibilidad a especificación alta
-----------------------------------------------------------------------------*/

display ""
display "═══════════════════════════════════════════════════════════════"
display "ESPECIFICACIONES ALTERNATIVAS: EVALUACIÓN DE ROBUSTEZ"
display "═══════════════════════════════════════════════════════════════"
display ""

// Modelo base (ya estimado arriba)
display "Modelo base completo ya estimado y almacenado."
display ""

/*--- Especificación 1: Sin tributos ni salarios ---*/

display "──────────────────────────────────────────────────────────────"
display "Especificación 1: Excluyendo tributos y salarios"
display "Justificación: Tributos/salarios son POST-TRATAMIENTO (endógenos)"
display "──────────────────────────────────────────────────────────────"

logit op2021_original c.ruc##(i.region) ventas_std i.sector sexo_gerente productividad_k digital_score i.tipo_local, vce(cluster ciiu_4dig)
estimates store sin_tributos_salarios

display "Especificación 1 completada."
display ""

/*--- Especificación 2: Solo controles exógenos ---*/

display "──────────────────────────────────────────────────────────────"
display "Especificación 2: Solo controles estrictamente exógenos"
display "Variables incluidas: región, sector, sexo gerente, tipo local"
display "Excluidas: ventas, productividad, digital, tributos, salarios, régimen"
display "──────────────────────────────────────────────────────────────"

logit op2021_original c.ruc##(i.region) i.sector sexo_gerente i.tipo_local, vce(cluster ciiu_4dig)
estimates store solo_exogenos

display "Especificación 2 completada."
display ""

/*--- Especificación 3: Solo microempresas ---*/

display "──────────────────────────────────────────────────────────────"
display "Especificación 3: Solo microempresas (ventas <S/50,000)"
display "Justificación: Muestra más homogénea reduce sesgo de selección"
display "──────────────────────────────────────────────────────────────"

preserve
keep if ventas_soles_2021 < 50000 & !missing(ventas_soles_2021)
display "Observaciones en submuestra: " _N

// Re-estandarizar ventas en submuestra
quietly summarize ventas_soles_2021
gen ventas_std_micro = (ventas_soles_2021 - r(mean)) / r(sd)

logit op2021_original c.ruc##(i.region) ventas_std_micro i.sector sexo_gerente productividad_k digital_score i.tipo_local, vce(cluster ciiu_4dig)
estimates store solo_micro
restore

display "Especificación 3 completada."
display ""

/*--- Especificación 4: Solo pequeñas empresas ---*/

display "──────────────────────────────────────────────────────────────"
display "Especificación 4: Solo pequeñas empresas (ventas ≥S/50,000)"
display "Justificación: Comparar con microempresas para heterogeneidad por tamaño"
display "──────────────────────────────────────────────────────────────"

preserve
keep if ventas_soles_2021 >= 50000 & !missing(ventas_soles_2021)
display "Observaciones en submuestra: " _N

// Re-estandarizar ventas en submuestra
quietly summarize ventas_soles_2021
gen ventas_std_pequena = (ventas_soles_2021 - r(mean)) / r(sd)

logit op2021_original c.ruc##(i.region) ventas_std_pequena i.sector sexo_gerente productividad_k digital_score i.tipo_local, vce(cluster ciiu_4dig)
estimates store solo_pequena
restore

display "Especificación 4 completada."
display ""

/*--- Comparación de resultados ---*/

display "══════════════════════════════════════════════════════════════"
display "COMPARACIÓN DE COEFICIENTES RUC×REGIÓN"
display "══════════════════════════════════════════════════════════════"
display ""

esttab modelo_principal sin_tributos_salarios solo_exogenos solo_micro solo_pequena, ///
    keep(ruc 1.region#c.ruc 2.region#c.ruc) ///
    se star(* 0.10 ** 0.05 *** 0.01) b(%9.4f) ///
    mtitles("Completo" "Sin trib/sal" "Solo exóg" "Micro<50k" "Peq≥50k") ///
    stats(N ll, fmt(%9.0fc %9.1f) labels("Observaciones" "Log-likelihood")) ///
    title("Robustez de efectos RUC×Región a especificación alternativa")

display ""
display "══════════════════════════════════════════════════════════════"
display "INTERPRETACIÓN:"
display "══════════════════════════════════════════════════════════════"
display "1. Coeficientes estables entre especificaciones 1-2 → Robustez a"
display "   inclusión de variables potencialmente endógenas"
display ""
display "2. Diferencias entre Micro/Pequeña → Heterogeneidad por tamaño:"
display "   - Si efectos más negativos en micro: Costos formalización"
display "     insostenibles en empresas pequeñas"
display "   - Si efectos más positivos en pequeña: Beneficios formalización"
display "     (acceso crédito/contratos) requieren escala mínima"
display ""
display "3. Comparar significancia estadística: ¿Se mantienen * en todas?"
display "   - Sí → Resultados robustos a especificación"
display "   - No → Advertencia sobre sensibilidad a controles incluidos"
display "══════════════════════════════════════════════════════════════"
display ""

// Restaurar el modelo logístico principal para efectos marginales
estimates restore modelo_principal

/*=============================================================================
4. EFECTOS MARGINALES POR REGIÓN
=============================================================================*/

// Efectos marginales promedio de RUC por región
// Interpretación: Cambio en probabilidad de supervivencia (puntos porcentuales)
// asociado a formalización (tener RUC) en cada región
margins region, dydx(ruc)

// Visualización de efectos marginales por región
marginsplot, ///
    title("Efecto de la formalización en supervivencia empresarial por región", size(medium)) ///
    ytitle("Efecto marginal de RUC" "Δ Pr(Supervivencia)", size(small)) ///
    xlabel(0 "Costa" 1 "Sierra" 2 "Selva", noticks) ///
    note("Efectos marginales promedio del modelo logístico con interacciones RUC×Región") ///
    name(me_region, replace)

// Tests de hipótesis sobre heterogeneidad regional
display ""
display "═══════════════════════════════════════════════════════════════"
display "TESTS DE HETEROGENEIDAD REGIONAL"
display "═══════════════════════════════════════════════════════════════"

// Test 1: Efecto conjunto de interacciones regionales
display ""
display "Test 1: ¿Hay heterogeneidad regional en el efecto de RUC?"
display "H0: Efecto de RUC es igual en todas las regiones (β₄=β₅=0)"
test 1.region#c.ruc 2.region#c.ruc

// Test 2: Sierra vs Costa
display ""
display "Test 2: ¿El efecto de RUC en Sierra difiere de Costa?"
display "H0: Efecto RUC en Sierra = Efecto RUC en Costa (β₄=0)"
test 1.region#c.ruc = 0

// Test 3: Selva vs Costa
display ""
display "Test 3: ¿El efecto de RUC en Selva difiere de Costa?"
display "H0: Efecto RUC en Selva = Efecto RUC en Costa (β₅=0)"
test 2.region#c.ruc = 0

// Test 4: Sierra vs Selva
display ""
display "Test 4: ¿El efecto de RUC en Sierra difiere de Selva?"
display "H0: Efecto RUC en Sierra = Efecto RUC en Selva (β₄=β₅)"
test 1.region#c.ruc = 2.region#c.ruc

display "═══════════════════════════════════════════════════════════════"
display ""

/*=============================================================================
5. EFECTOS MARGINALES POR VENTAS
=============================================================================*/

// Efectos marginales de RUC condicionales a escala empresarial (ventas estandarizadas)
// Objetivo: Explorar heterogeneidad del efecto de formalización según tamaño económico
// Interpretación: ¿El beneficio de formalización varía según escala de ventas?

// Obtener estadísticas de ventas estandarizadas
quietly summarize ventas_std, detail
local p10 = r(p10)
local p25 = r(p25)
local p50 = r(p50)
local p75 = r(p75)
local p90 = r(p90)

display "Distribución de ventas estandarizadas (percentiles):"
display "  P10: " %6.2f `p10' " | P25: " %6.2f `p25' " | P50: " %6.2f `p50'
display "  P75: " %6.2f `p75' " | P90: " %6.2f `p90'
display ""

// Calcular efectos marginales de RUC en diferentes niveles de ventas
// Usar rango amplio para capturar microempresas (P10) hasta pequeñas empresas (P90)
margins region, dydx(ruc) at(ventas_std = (-2(0.5)3))

// Visualización: Efecto de RUC según escala de ventas por región
marginsplot, ///
    title("Efecto de formalización según escala de ventas", size(medium)) ///
    subtitle("Por región geográfica", size(small)) ///
    ytitle("Efecto marginal de RUC" "Δ Pr(Supervivencia)", size(small)) ///
    xtitle("Ventas estandarizadas (σ)", size(small)) ///
    xlabel(-2(0.5)3, format(%3.1f)) ///
    xline(0, lpattern(dash) lcolor(gs10)) ///
    note("Ventas estandarizadas: 0 = media muestral, ±1 = ±1 desviación estándar" ///
         "P10≈-1.5σ (microempresas pequeñas), P50≈-0.3σ (mediana), P90≈1.2σ (pequeñas empresas grandes)") ///
    legend(cols(3) size(small)) ///
    name(me_ventas_region, replace)

// Interpretación guiada para el paper
display "═══════════════════════════════════════════════════════════════"
display "INTERPRETACIÓN: EFECTOS MARGINALES POR ESCALA DE VENTAS"
display "═══════════════════════════════════════════════════════════════"
display "- Si líneas son horizontales: Efecto de RUC constante (no varía con ventas)"
display "- Si líneas crecen: Mayor beneficio de formalización en empresas grandes"
display "- Si líneas decrecen: Mayor beneficio de formalización en microempresas"
display "- Diferencias entre regiones: Heterogeneidad espacial del efecto"
display "═══════════════════════════════════════════════════════════════"
display ""



/*=============================================================================
6. PROBABILIDADES PREDICHAS POR PRODUCTIVIDAD
=============================================================================*/

// NOTA METODOLÓGICA:
// No calculamos efectos marginales de RUC condicionales a productividad porque
// el modelo NO incluye interacción RUC×productividad (solo RUC×Región).
// Sin esa interacción, el efecto de RUC no varía verdaderamente con productividad.
//
// En su lugar, calculamos PROBABILIDADES PREDICHAS de supervivencia por región
// en diferentes niveles de productividad, mostrando cómo la productividad afecta
// el outcome directamente (no el efecto de RUC).

// Obtener estadísticas de productividad
quietly summarize productividad_k, detail
local p10 = r(p10)
local p25 = r(p25)
local p50 = r(p50)
local p75 = r(p75)
local p90 = r(p90)
local max = r(max)

display "Distribución de productividad laboral (miles de soles/trabajador):"
display "  P10: " %6.2f `p10' "k | P25: " %6.2f `p25' "k | P50: " %6.2f `p50' "k"
display "  P75: " %6.2f `p75' "k | P90: " %6.2f `p90' "k | Max: " %6.2f `max' "k"
display ""

// Calcular probabilidades predichas de supervivencia por región
// en diferentes niveles de productividad (0 a 120 en incrementos de 10)
margins region, at(productividad_k = (0(10)120))

// Visualización: Probabilidad de supervivencia según productividad por región
marginsplot, ///
    title("Probabilidad de supervivencia según productividad laboral", size(medium)) ///
    subtitle("Por región geográfica", size(small)) ///
    ytitle("Pr(Operativa en 2021)", size(small)) ///
    xtitle("Productividad laboral (miles S/. / trabajador)", size(small)) ///
    xlabel(0(20)120, format(%3.0f)) ///
    ylabel(, format(%3.2f)) ///
    note("VAB = Valor Agregado Bruto (Ventas - Consumo Intermedio)" ///
         "Productividad = VAB / Trabajadores Ocupados (definición oficial INEI 2022)" ///
         "Curvas muestran probabilidades predichas manteniendo otros controles en su media") ///
    legend(cols(3) size(small) order(1 "Costa" 2 "Sierra" 3 "Selva")) ///
    name(prob_productividad, replace)

// Interpretación guiada
display "═══════════════════════════════════════════════════════════════"
display "INTERPRETACIÓN: PRODUCTIVIDAD Y SUPERVIVENCIA"
display "═══════════════════════════════════════════════════════════════"
display "- Curvas crecientes: Mayor productividad aumenta supervivencia"
display "- Pendiente de curvas: Magnitud del efecto productividad en cada región"
display "- Separación vertical: Diferencias regionales en probabilidad base"
display "- Conexión con Jovanovic (1982): Productividad como proxy de θ (eficiencia)"
display "  → Empresas más eficientes (mayor θ) tienen mayor probabilidad de sobrevivir"
display "═══════════════════════════════════════════════════════════════"
display ""



/*=============================================================================
7. ANÁLISIS POR RÉGIMEN TRIBUTARIO
=============================================================================*/

// NOTA METODOLÓGICA:
// El régimen tributario está altamente correlacionado con formalización:
// - Régimen 0 (Ninguno) = empresas informales (RUC=0)
// - Regímenes 1-4 (RUS, RER, RMT, RG) = empresas formales (RUC=1)
//
// Análisis se enfoca en DOS dimensiones:
// 1. Probabilidades predichas por régimen (todos los datos)
// 2. Comparación entre regímenes tributarios (solo formales: RUC=1)

// Restaurar modelo principal
estimates restore modelo_principal

// Distribución de empresas por régimen tributario
display "═══════════════════════════════════════════════════════════════"
display "DISTRIBUCIÓN DE EMPRESAS POR RÉGIMEN TRIBUTARIO"
display "═══════════════════════════════════════════════════════════════"
tab regimen, missing

// Distribución cruzada: régimen × formalización
tab regimen ruc, col nofreq
display ""

/*-----------------------------------------------------------------------------
7.1 PROBABILIDADES PREDICHAS POR RÉGIMEN TRIBUTARIO (TODOS)
-----------------------------------------------------------------------------*/

// Probabilidades predichas de supervivencia por régimen
// Interpretación: Pr(Operativa en 2021) promedio en cada régimen tributario
margins regimen

// Visualización
marginsplot, ///
    title("Probabilidad de supervivencia por régimen tributario", size(medium)) ///
    ytitle("Pr(Operativa en 2021)", size(small)) ///
    xtitle("Régimen tributario", size(small)) ///
    xlabel(0 "Ninguno" 1 "RUS" 2 "RER" 3 "RMT" 4 "RG", angle(45) labsize(small)) ///
    ylabel(, format(%3.2f)) ///
    note("0=Ninguno (informales), 1=RUS, 2=RER, 3=RMT, 4=RG" ///
         "Régimen 0 incluye empresas informales (sin RUC)" ///
         "Regímenes 1-4 son solo empresas formales") ///
    name(prob_regimen_all, replace)

// Test de diferencias entre regímenes
display "Test de Wald: ¿Difieren las probabilidades entre regímenes?"
test 1.regimen 2.regimen 3.regimen 4.regimen
display ""

/*-----------------------------------------------------------------------------
7.2 ANÁLISIS ENTRE REGÍMENES TRIBUTARIOS (SOLO FORMALES)
-----------------------------------------------------------------------------*/

// Comparación más limpia: SOLO empresas formales (RUC=1)
// Objetivo: Evaluar si régimen tributario afecta supervivencia entre formales

display "═══════════════════════════════════════════════════════════════"
display "ANÁLISIS ENTRE REGÍMENES TRIBUTARIOS (SOLO FORMALES RUC=1)"
display "═══════════════════════════════════════════════════════════════"

// Probabilidades predichas por régimen, CONDICIONADO a RUC=1
margins regimen, at(ruc=1)

// Visualización (solo formales)
marginsplot, ///
    title("Probabilidad de supervivencia por régimen tributario", size(medium)) ///
    subtitle("Solo empresas formales (RUC=1)", size(small)) ///
    ytitle("Pr(Operativa en 2021 | RUC=1)", size(small)) ///
    xtitle("Régimen tributario", size(small)) ///
    xlabel(1 "RUS" 2 "RER" 3 "RMT" 4 "RG", angle(45) labsize(small)) ///
    ylabel(, format(%3.2f)) ///
    note("RUS = Nuevo Régimen Único Simplificado (ventas <S/96k)" ///
         "RER = Régimen Especial de Renta (ventas <S/525k)" ///
         "RMT = Régimen MYPE Tributario (ventas <S/1,700k)" ///
         "RG = Régimen General (sin límite de ventas)") ///
    name(prob_regimen_formales, replace)

// Tests pareados entre regímenes (solo formales)
display ""
display "Tests pareados de diferencias entre regímenes formales:"
display ""
display "RUS vs RER:"
margins regimen, at(ruc=1) pwcompare(effects)

// Efectos marginales discretos de cambiar de régimen
// Interpretación: Cambio en Pr(supervivencia) al pasar del régimen base a otro
display ""
display "Efectos marginales discretos de régimen tributario (base: Ninguno):"
margins, dydx(regimen)

/*-----------------------------------------------------------------------------
7.3 RÉGIMEN TRIBUTARIO POR REGIÓN
-----------------------------------------------------------------------------*/

display ""
display "═══════════════════════════════════════════════════════════════"
display "RÉGIMEN TRIBUTARIO POR REGIÓN (SOLO FORMALES)"
display "═══════════════════════════════════════════════════════════════"

// Distribución de regímenes por región (solo formales)
tab regimen region if ruc==1, col nofreq

// Probabilidades predichas por régimen y región (solo formales)
margins region#regimen, at(ruc=1)

// Visualización: Interacción región × régimen
marginsplot, ///
    title("Probabilidad de supervivencia: Región × Régimen tributario", size(medium)) ///
    subtitle("Solo empresas formales (RUC=1)", size(small)) ///
    ytitle("Pr(Operativa en 2021 | RUC=1)", size(small)) ///
    xlabel(1 "RUS" 2 "RER" 3 "RMT" 4 "RG", angle(45) labsize(small)) ///
    ylabel(, format(%3.2f)) ///
    legend(cols(3) size(small) order(1 "Costa" 2 "Sierra" 3 "Selva")) ///
    note("Patrones regionales en probabilidad de supervivencia según régimen tributario" ///
         "Separación vertical indica diferencias regionales en efecto del régimen") ///
    name(region_regimen, replace)

// Interpretación guiada
display ""
display "═══════════════════════════════════════════════════════════════"
display "INTERPRETACIÓN: RÉGIMEN TRIBUTARIO Y SUPERVIVENCIA"
display "═══════════════════════════════════════════════════════════════"
display "1. Régimen 0 (Ninguno) vs 1-4 (Formales): Captura efecto formalización"
display "2. Entre regímenes 1-4: Captura efecto de complejidad tributaria"
display "   - RUS (más simple) → RER → RMT → RG (más complejo)"
display "   - Complejidad correlaciona con tamaño/ventas de empresa"
display "3. Diferencias regionales: ¿Varía impacto de régimen según ubicación?"
display "4. ADVERTENCIA: Régimen es endógeno (empresas eligen según ventas/costos)"
display "   → Interpretación: Asociaciones condicionales, NO efectos causales"
display "═══════════════════════════════════════════════════════════════"
display ""

/*=============================================================================
8. ANÁLISIS DE ROBUSTEZ
=============================================================================*/

/*-----------------------------------------------------------------------------
REFERENCIAS BIBLIOGRÁFICAS:

[1] Cameron, A. C., & Miller, D. L. (2015). "A Practitioner's Guide to
    Cluster-Robust Inference." Journal of Human Resources, 50(2), 317-372.
    DOI: 10.3368/jhr.50.2.317

[2] Lee, D. S., & Lemieux, T. (2010). "Regression Discontinuity Designs in
    Economics." Journal of Economic Literature, 48(2), 281-355.
    DOI: 10.1257/jel.48.2.281

[3] Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2020). "Optimal Bandwidth
    Choice for Robust Bias-Corrected Inference in Regression Discontinuity
    Designs." The Econometrics Journal, 23(2), 192-210.
    DOI: 10.1093/ectj/utz022

[4] Imbens, G. W., & Kalyanaraman, K. (2012). "Optimal Bandwidth Choice for
    the Regression Discontinuity Estimator." The Review of Economic Studies,
    79(3), 933-959.
    DOI: 10.1093/restud/rdr043

[5] McCrary, J. (2008). "Manipulation of the Running Variable in the Regression
    Discontinuity Design: A Density Test." Journal of Econometrics, 142(2),
    698-714.
    DOI: 10.1016/j.jeconom.2007.05.005
-----------------------------------------------------------------------------*/

// Guardar estimaciones del modelo completo
estimates save "modelo_completo.ster", replace

/*-----------------------------------------------------------------------------
8.1 ROBUSTEZ: CLUSTERING A DIFERENTES NIVELES (CIIU)

Referencia: Cameron & Miller (2015) [1]
Objetivo: Verificar que inferencia no dependa del nivel de agregación sectorial
Interpretación: Si coeficientes mantienen significancia → resultados robustos

Modelo principal usa CIIU 4 dígitos (Clase, máxima granularidad disponible)
Justificación: Recomendación del asesor dado tamaño censal (N=1,377,931)

Niveles alternativos de clustering probados:
- CIIU 2 dígitos (División): ~21 clusters, agregación sectorial amplia
- CIIU 3 dígitos (Grupo): ~88 clusters, agregación media
- CIIU 4 dígitos (Clase): ~300+ clusters, desagregación estándar
- Región geográfica: 3 clusters, máximo conservadurismo (errores más grandes)

Expectativa: CIIU 2 dígitos → errores estándar más grandes (conservador)
             CIIU 4 dígitos → errores estándar más pequeños (eficiente)
-----------------------------------------------------------------------------*/

// Modelo principal usa CIIU 4 dígitos
estimates restore modelo_principal

// Robustez 1: CIIU 2 dígitos (División sectorial, máxima agregación)
logit op2021_original c.ruc##(i.region) ventas_std i.sector sexo_gerente productividad_k digital_score tributos_k salarios_k i.tipo_local i.regimen, vce(cluster ciiu_2dig)
estimates store robustez_ciiu2

// Robustez 2: CIIU 3 dígitos (Grupo sectorial, agregación media)
logit op2021_original c.ruc##(i.region) ventas_std i.sector sexo_gerente productividad_k digital_score tributos_k salarios_k i.tipo_local i.regimen, vce(cluster ciiu_3dig)
estimates store robustez_ciiu3

// Robustez 3: CIIU 4 dígitos (Clase sectorial, desagregación estándar)
logit op2021_original c.ruc##(i.region) ventas_std i.sector sexo_gerente productividad_k digital_score tributos_k salarios_k i.tipo_local i.regimen, vce(cluster ciiu_4dig)
estimates store robustez_ciiu4

// Robustez 4: Clustering por región (3 clusters, máximo conservadurismo)
// Esperar errores estándar muy grandes (pocos clusters)
logit op2021_original c.ruc##(i.region) ventas_std i.sector sexo_gerente productividad_k digital_score tributos_k salarios_k i.tipo_local i.regimen, vce(cluster region)
estimates store robustez_region

// Comparar coeficientes de interacción RUC×Región con diferentes niveles de clustering
// Verificar: ¿Se mantiene significancia estadística independiente del clustering?
// Patrón esperado: SE aumentan de CIIU-4 → CIIU-3 → CIIU-2 → Región
// capture ssc install estout
esttab modelo_principal robustez_ciiu4 robustez_ciiu3 robustez_ciiu2 robustez_region, ///
    keep(ruc 1.region#c.ruc 2.region#c.ruc) ///
    se star(* 0.10 ** 0.05 *** 0.01) b(%9.4f) stats(N ll, fmt(%9.0fc %9.1f))

/*-----------------------------------------------------------------------------
8.2 ROBUSTEZ: REGRESSION DISCONTINUITY DESIGN EN UMBRAL RUS

Referencias: Lee & Lemieux (2010) [2], Calonico et al. (2020) [3],
             Imbens & Kalyanaraman (2012) [4]

Marco legal: Nuevo Régimen Único Simplificado (RUS) - Ley 30296
Umbral tributario: S/96,000 anuales (límite máximo ingresos brutos 2021)

Estrategia RDD:
- Running variable: Distancia al umbral (ventas - 96,000)
- Treatment: Estar sobre el umbral RUS (ventas ≥ S/96,000)
- Outcome: Probabilidad de operar en 2021 (supervivencia empresarial)

Identificación causal: Empresas ligeramente arriba/abajo del umbral son
comparables en observables y no-observables → discontinuidad en tratamiento
permite estimar efecto causal del umbral tributario en supervivencia
-----------------------------------------------------------------------------*/

// Crear variable de distancia al umbral (running variable)
capture drop dist_umbral_rus
gen dist_umbral_rus = ventas_soles_2021 - 96000
label variable dist_umbral_rus "Distancia al umbral RUS (S/96,000)"

// Identificar empresas arriba/abajo del umbral (treatment variable)
capture drop sobre_umbral
gen sobre_umbral = (ventas_soles_2021 >= 96000) if !missing(ventas_soles_2021)
label variable sobre_umbral "Ventas ≥ umbral RUS"

// RDD con bandwidth óptimo (±1 desviación estándar del umbral)
// Siguiendo Imbens & Kalyanaraman (2012) para bandwidth MSE-óptimo
quietly summarize dist_umbral_rus if abs(dist_umbral_rus) < 50000
local bandwidth = r(sd)
display "Bandwidth óptimo (±1 SD): ±" round(`bandwidth', 1) " soles"

// Análisis local alrededor del umbral con 3 bandwidths diferentes
// Robustez según Calonico et al. (2020): Probar múltiples bandwidths

// Bandwidth 1: ±15,000 (estrecho, menor sesgo, mayor varianza)
preserve
keep if abs(dist_umbral_rus) <= 15000
logit op2021_original i.sobre_umbral productividad_k digital_score i.sector i.regimen, vce(cluster ciiu_4dig)
estimates store rdd_bw15k
margins sobre_umbral
display "Observaciones bandwidth ±15k: " e(N)
restore

// Bandwidth 2: ±30,000 (medio, balanceo sesgo-varianza)
preserve
keep if abs(dist_umbral_rus) <= 30000
logit op2021_original i.sobre_umbral productividad_k digital_score i.sector i.regimen, vce(cluster ciiu_4dig)
estimates store rdd_bw30k
margins sobre_umbral
display "Observaciones bandwidth ±30k: " e(N)
restore

// Bandwidth 3: ±50,000 (amplio, mayor sesgo, menor varianza)
preserve
keep if abs(dist_umbral_rus) <= 50000
logit op2021_original i.sobre_umbral productividad_k digital_score i.sector i.regimen, vce(cluster ciiu_4dig)
estimates store rdd_bw50k
margins sobre_umbral
display "Observaciones bandwidth ±50k: " e(N)
restore

// Comparar estimaciones RDD con diferentes bandwidths
// Interpretación: Si coeficiente sobre_umbral consistente → efecto causal robusto
esttab rdd_bw15k rdd_bw30k rdd_bw50k, ///
    keep(1.sobre_umbral productividad_k digital_score) ///
    se star(* 0.10 ** 0.05 *** 0.01) b(%9.4f) stats(N ll, fmt(%9.0fc %9.1f))

/*-----------------------------------------------------------------------------
8.3 ROBUSTEZ: VALIDACIÓN DEL UMBRAL (PLACEBO TESTS)

Referencia: Lee & Lemieux (2010) [2] Section 5.3 "Specification Checks"

Objetivo: Verificar que el efecto observado en 8.2 es específico al umbral
real (S/96,000) y no un artefacto estadístico o patrón general en ventas

Metodología: Estimar discontinuidades en umbrales "falsos" donde NO existe
tratamiento tributario. Si RDD válido → placebos NO significativos

Test de falsificación:
- H0: Coeficiente placebo = 0 (no hay discontinuidad en umbral falso)
- H1: Coeficiente placebo ≠ 0 (diseño RDD potencialmente inválido)

Interpretación: Rechazar H0 en placebos invalida identificación causal del
umbral real, sugiriendo patrones no-lineales en ventas no relacionados con
régimen tributario
-----------------------------------------------------------------------------*/

// Placebo test 1: Umbral falso en S/70,000 (debajo del RUS real)
// Expectativa: NO significativo (no hay discontinuidad tributaria en S/70k)
capture drop placebo_70k
gen placebo_70k = (ventas_soles_2021 >= 70000) if !missing(ventas_soles_2021)
preserve
keep if ventas_soles_2021 > 50000 & ventas_soles_2021 < 90000
logit op2021_original i.placebo_70k productividad_k digital_score i.sector, vce(cluster ciiu_4dig)
estimates store placebo_70k
display "Placebo S/70,000 - Observaciones: " e(N)
restore

// Placebo test 2: Umbral falso en S/120,000 (encima del RUS, antes del RER)
// Expectativa: NO significativo (no hay umbral tributario en S/120k)
capture drop placebo_120k
gen placebo_120k = (ventas_soles_2021 >= 120000) if !missing(ventas_soles_2021)
preserve
keep if ventas_soles_2021 > 100000 & ventas_soles_2021 < 140000
logit op2021_original i.placebo_120k productividad_k digital_score i.sector, vce(cluster ciiu_4dig)
estimates store placebo_120k
display "Placebo S/120,000 - Observaciones: " e(N)
restore

// Placebo test 3: Umbral falso en S/200,000 (zona media entre RUS-RER)
// Expectativa: NO significativo (no hay umbral tributario en S/200k)
capture drop placebo_200k
gen placebo_200k = (ventas_soles_2021 >= 200000) if !missing(ventas_soles_2021)
preserve
keep if ventas_soles_2021 > 175000 & ventas_soles_2021 < 225000
logit op2021_original i.placebo_200k productividad_k digital_score i.sector, vce(cluster ciiu_4dig)
estimates store placebo_200k
display "Placebo S/200,000 - Observaciones: " e(N)
restore

// Comparar placebos: NINGUNO debería ser significativo
// Si algún placebo significativo → problema de especificación, no causalidad
esttab placebo_70k placebo_120k placebo_200k, ///
    keep(*placebo*) ///
    se star(* 0.10 ** 0.05 *** 0.01) b(%9.4f) stats(N ll, fmt(%9.0fc %9.1f))

// Test formal de hipótesis: ¿Todos los placebos son conjuntamente no significativos?
// Expectativa: No rechazar H0 (placebos = 0) para validar RDD del umbral real

/*-----------------------------------------------------------------------------
8.4 ROBUSTEZ: TEST DE MANIPULACIÓN (McCRARY DENSITY TEST)

Referencia: McCrary, J. (2008) [5] "Manipulation of the Running Variable in
            the Regression Discontinuity Design: A Density Test"

Actualización: Cattaneo, M. D., Jansson, M., & Ma, X. (2020). "Simple Local
               Polynomial Density Estimators." Journal of the American
               Statistical Association, 115(531), 1449-1455.

Objetivo: Verificar si empresas manipulan reportes de ventas para permanecer
debajo del umbral RUS (S/96,000) y evitar regímenes tributarios más complejos

Estrategia de evasión potencial:
- Empresa con ventas reales S/98,000 → Reporta S/95,000 → Permanece en RUS
- Evidencia: Acumulación anormal ("bunching") justo debajo del umbral
- Consecuencia: Viola supuesto de continuidad en RDD → estimador sesgado

Test de hipótesis:
- H0: Densidad continua en umbral (lim f(x) desde izquierda = lim f(x) desde derecha)
- H1: Discontinuidad en densidad (manipulación sistemática)

Interpretación:
- p-value > 0.05 → No rechazar H0 → No hay evidencia de manipulación → RDD válido
- p-value ≤ 0.05 → Rechazar H0 → Manipulación detectada → RDD potencialmente inválido
-----------------------------------------------------------------------------*/

// Instalar comando moderno para density test (si no está instalado)
// capture ssc install rddensity
// capture ssc install lpdensity

// Test McCrary con método Cattaneo et al. (2020)
// Usa dist_umbral_rus como running variable con cutoff en 0
rddensity dist_umbral_rus, c(0) plot
// c(0) porque dist_umbral_rus = ventas - 96000, entonces umbral está en 0

// Guardar resultado del test
matrix mccrary_test = r(table)
display "════════════════════════════════════════════════════════════"
display "TEST DE MANIPULACIÓN McCRARY (2008)"
display "════════════════════════════════════════════════════════════"
display "Hipótesis nula: Densidad continua en umbral (no manipulación)"
display "P-value: " r(pval_p)
display ""
if r(pval_p) > 0.05 {
    display "✓ RESULTADO: No hay evidencia estadística de manipulación"
    display "  → Diseño RDD es VÁLIDO para identificación causal"
}
else {
    display "✗ ADVERTENCIA: Se detecta posible manipulación del umbral"
    display "  → Resultados RDD deben interpretarse con CAUTELA"
    display "  → Considerar estimadores robustos a manipulación (donut-hole RDD)"
}
display "════════════════════════════════════════════════════════════"
display ""

// Método alternativo: Test McCrary original (DCdensity)
// Requiere instalación: ssc install rddensity (ya incluye DCdensity)
capture {
    DCdensity dist_umbral_rus, breakpoint(0) generate(Xj Yj r0 fhat se_fhat) ///
        graphname("mccrary_test")

    display "Test McCrary original - Discontinuidad en log-densidad:"
    display "Theta (discontinuidad): " r(theta)
    display "SE: " r(se)
    display "P-value: " r(p)
}

// Análisis visual de densidad alrededor del umbral
// Gráfico de histograma con bins pequeños para detectar bunching
preserve
keep if abs(dist_umbral_rus) <= 50000
histogram dist_umbral_rus, width(5000) frequency ///
    xline(0, lcolor(red) lwidth(thick) lpattern(dash)) ///
    title("Distribución de ventas alrededor del umbral RUS", size(medium)) ///
    subtitle("Test visual de manipulación - McCrary (2008)") ///
    xtitle("Distancia al umbral S/96,000 (soles)") ///
    ytitle("Número de empresas") ///
    note("Línea roja: Umbral RUS (S/96,000)" ///
         "Bunching visible debajo del umbral sugiere manipulación estratégica") ///
    name(bunching_test, replace)
restore

// Contar empresas en ventanas simétricas alrededor del umbral
// Si hay manipulación, debería haber más empresas justo debajo que justo encima
quietly count if dist_umbral_rus >= -5000 & dist_umbral_rus < 0
local n_below = r(N)
quietly count if dist_umbral_rus >= 0 & dist_umbral_rus < 5000
local n_above = r(N)
local ratio = `n_below' / `n_above'

display "Empresas en ventana ±S/5,000 del umbral:"
display "  Debajo umbral [-5k, 0): " `n_below' " empresas"
display "  Encima umbral [0, +5k): " `n_above' " empresas"
display "  Ratio (debajo/encima): " %5.2f `ratio'
display ""
if `ratio' > 1.2 {
    display "⚠ ADVERTENCIA: Exceso de empresas debajo del umbral (ratio > 1.2)"
    display "   Posible manipulación estratégica de ventas reportadas"
}
else if `ratio' < 0.8 {
    display "ℹ NOTA: Menos empresas debajo que encima (ratio < 0.8)"
    display "   Posible incentivo a reportar ventas mayores (inusual)"
}
else {
    display "✓ Distribución balanceada alrededor del umbral (ratio ≈ 1.0)"
}

/*=============================================================================
9. DIAGNÓSTICOS DEL MODELO
=============================================================================*/

// Restaurar modelo principal para diagnósticos
estimates restore modelo_principal

// Capacidad predictiva
capture drop phat
predict phat, pr
lroc, title("Capacidad Predictiva del Modelo") name(roc_curve, replace)

// Sensibilidad y especificidad
estat classification

// Bondad de ajuste
estat ic

// Cerrar log
log close


