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

// Variables logarítmicas para estabilizar escalas
gen ln_ventas = ln(ventas_soles_2021 + 1)
gen ln_productividad = ln(productividad_x_trabajador + 1)
gen ln_tributos = ln(tributos + 1)
gen ln_salarios = ln(salarios + 1)

// Dividir por 1000 para reducir valores en lugar de logaritmo natural (porque seria interpretar el % del %)
// Eliminar ventas, quedarme con productividad por multicolinealidad (dado que ambas variables contienen venta - Verificar que productividad la calculé diviendo ventas totales entre trabajadores)
// Usar logit op2021_original c.ruc##(i.region) tamano_empresa sexo_gerente ln_ventas ln_productividad digital_score ln_tributos ln_salarios i.tipo_local i.regimen, vce(cluster CIIU) en lugar de sector dado que sector comercio, manufacturas y servicio es muy poco, mejor agrupar por CIIU lo cual significa agrupar por los primeros 2 numeros, es decir todos los 01, todos los 02, para de esta manera agrupar empresas que se dedican a la misma actividad. El comando vce cluste lo que hace es "limpiar" un poco la heterogeneidad y sesgo que se genera al tener empresas que se dedican a diferentes rubros
// al usar comando margins region , dydx(ruc) esto me permite hacer gráficos para ver cómo evoluciona supervivencia según aumenta el tamaño de la empresa, etc (idea del profesor)


// Variables dummy regionales (Costa como base)
gen sierra = (region == 1)
gen selva = (region == 2)
tab region
// Análisis de tamaños muestrales por categorías
tab op2021_original tamano_empresa, col row
tab regimen, missing
tab regimen op2021_original, col row

// Estadísticas descriptivas básicas
tab op2021_original region, col nofreq
tab op2021_original ruc, col nofreq
bysort op2021_original: summarize ruc sierra selva digital_score ventas_soles_2021

/*=============================================================================
2. ANÁLISIS DE ENDOGENEIDAD: CARACTERÍSTICAS POR RÉGIMEN TRIBUTARIO
=============================================================================*/

// Análisis descriptivo por régimen tributario
tab regimen op2021_original, col

// Características económicas por régimen
bysort regimen: summarize ventas_soles_2021 productividad_x_trabajador tributos salarios

// Test de diferencias en características observables
oneway ln_ventas regimen, tabulate
oneway ln_productividad regimen, tabulate

/*=============================================================================
3. MODELO PRINCIPAL: EFECTOS HETEROGÉNEOS DE FORMALIZACIÓN
=============================================================================*/

// Modelo completo con interacciones RUC*Región
logit op2021_original c.ruc##(i.region) tamano_empresa sexo_gerente ln_ventas ln_productividad digital_score ln_tributos ln_salarios i.tipo_local i.regimen, vce(cluster CIIU)

tab sector

estimates store modelo_principal

// Odds ratios
logit, or

/*=============================================================================
4. EFECTOS MARGINALES POR REGIÓN
=============================================================================*/

// Efectos marginales por región
margins region , dydx(ruc)
margins, dydx(ruc) at(sierra=1 selva=0) // Sierra
margins, dydx(ruc) at(sierra=0 selva=1) // Selva

// Tests de hipótesis sobre heterogeneidad regional
test (1.sierra#c.ruc = 0) (1.selva#c.ruc = 0)
test c.ruc#1.sierra = 0
test c.ruc#1.selva = 0
test c.ruc#1.sierra = c.ruc#1.selva

/*=============================================================================
5. ANÁLISIS DE ROBUSTEZ POR MUESTREO
=============================================================================*/

// Guardar estimaciones del modelo completo
estimates save "modelo_completo.ster", replace

// 5.1 Muestreo balanceado por tamaño de empresa
preserve
count if tamano_empresa == 1
local n_pequenas = r(N)
sample `n_pequenas' if tamano_empresa == 0, count
logit op2021_original c.ruc##(sierra selva) i.sector tamano_empresa sexo_gerente ///
      ln_ventas ln_productividad digital_score ln_tributos ln_salarios ///
      i.tipo_local i.regimen, robust
estimates store modelo_balanceado_tamano
margins, dydx(ruc) at(sierra=0 selva=0)
margins, dydx(ruc) at(sierra=1 selva=0)
margins, dydx(ruc) at(sierra=0 selva=1)
restore

// 5.2 Muestreo estratificado por régimen tributario
preserve
sample 80000 if regimen == 0, count
sample 80000 if regimen == 1, count
sample 80000 if regimen == 2, count
sample 80000 if regimen == 3, count
sample 80000 if regimen == 4, count
logit op2021_original c.ruc##(sierra selva) i.sector tamano_empresa sexo_gerente ///
      ln_ventas ln_productividad digital_score ln_tributos ln_salarios ///
      i.tipo_local i.regimen, robust
estimates store modelo_balanceado_regimen
margins, dydx(ruc) at(sierra=0 selva=0)
margins, dydx(ruc) at(sierra=1 selva=0)
margins, dydx(ruc) at(sierra=0 selva=1)
restore

/*=============================================================================
6. ANÁLISIS DE ROBUSTEZ: AUTOSELECCIÓN Y ENDOGENEIDAD
=============================================================================*/

// Análisis por quintiles de ventas
xtile ventas_quintil = ln_ventas, nq(5)

// Efectos dentro de cada quintil
forvalues q = 1/5 {
    quietly logit op2021_original i.regimen if ventas_quintil==`q', robust
    display "Quintil `q':"
    margins regimen
}

// Variable binaria para régimen avanzado
gen regimen_avanzado = (regimen == 3 | regimen == 4)
label var regimen_avanzado "RMT o RG"

// Modelo de selección en régimen
logit regimen_avanzado ln_ventas ln_productividad digital_score tamano_empresa i.sector i.region, robust

// Propensity Score Matching
logit regimen_avanzado ln_ventas ln_productividad digital_score i.sector i.region tamano_empresa, robust
predict pscore, pr

// PSM simple (alternativa más rápida)
// teffects nnmatch muy lento con N>1M, usar solo si necesario

// Análisis cerca del umbral RUS (S/96,000)
preserve
keep if ventas_soles_2021 > 80000 & ventas_soles_2021 < 110000
logit op2021_original i.regimen ln_productividad digital_score i.sector, robust
margins regimen
restore

/*=============================================================================
7. DIAGNÓSTICOS DEL MODELO
=============================================================================*/

// Bondad de ajuste
estat ic
estat gof, group(10)

// Capacidad predictiva
predict phat, pr
lroc, title("Capacidad Predictiva del Modelo") name(roc_curve, replace)

// Tabla de clasificación
estat classification

// Verificar multicolinealidad
corr ruc sierra selva ln_ventas ln_productividad

// Cerrar log
log close
