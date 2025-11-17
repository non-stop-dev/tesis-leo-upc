/*=============================================================================
ANÁLISIS DE SUPERVIVENCIA EMPRESARIAL - VERSIÓN PAPER (COMPACTA)
Autor: Leonardo León
Fecha: Octubre 2025
Propósito: Generar solo resultados necesarios para 7. RESULTADOS.md (PAPER)

NOTA: Esta es una versión simplificada de 4.regresiones1.do
      Solo genera resultados para defensa de tesis (no incluye RDD, placebos, McCrary)
=============================================================================*/

cls
clear all
set more off
capture log close

// Establecer directorio
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Base curada"

// Crear log
log using "resultados_paper_$(S_DATE).log", replace

// Cargar datos
use "2.v_censo_limpio.dta", clear

/*=============================================================================
1. PREPARACIÓN DE VARIABLES
=============================================================================*/

// Productividad: truncar valores negativos
quietly summarize productividad_x_trabajador if productividad_x_trabajador > 0
local min_pos = r(min)
gen productividad_clean = productividad_x_trabajador
replace productividad_clean = `min_pos' if productividad_x_trabajador <= 0 & !missing(productividad_x_trabajador)

// Escalamiento (dividir por 1000)
gen productividad_k = productividad_clean / 1000
gen tributos_k = tributos / 1000
gen salarios_k = salarios / 1000

// Winsorización al 1%-99%
quietly summarize productividad_x_trabajador, detail
replace productividad_k = r(p1)/1000 if productividad_x_trabajador < r(p1) & !missing(productividad_x_trabajador)
replace productividad_k = r(p99)/1000 if productividad_x_trabajador > r(p99) & !missing(productividad_x_trabajador)
 

// Dividir ventas entre 1000 en lugar de estandarizar ventas
summarize ventas_soles_2021
gen ventas_k = ventas_soles_2021 / 1000
label var ventas_k "Ventas 2021 dividido entre 1000"

// Dummies regionales (Costa como base)
gen sierra = (region == 1)
gen selva = (region == 2)

/*=============================================================================
2. TEST DE BALANCE: TABLA 1 DEL PAPER (Sección 1.2)
=============================================================================*/

display ""
display "══════════════════════════════════════════════════════════"
display "TABLA 1: TEST DE BALANCE - FORMALES VS INFORMALES"
display "══════════════════════════════════════════════════════════"
display ""

// Variables continuas (diferencias en medias)
ttest ventas_k, by(ruc)
gen productividad_temp = productividad_x_trabajador / 1000
ttest productividad_temp, by(ruc)
drop productividad_temp
ttest digital_score, by(ruc)

// Variables categóricas (Chi²)
tab region ruc, col nofreq chi2
tab sector ruc, col nofreq chi2

// Estadísticas por régimen (Sección 1.3)
tabstat productividad_x_trabajador ventas_k, by(regimen) stat(mean n)

/*=============================================================================
3. MODELO PRINCIPAL: TABLA 2 DEL PAPER (Sección 2.2)
=============================================================================*/

display ""
display "══════════════════════════════════════════════════════════"
display "TABLA 2: MODELO LOGÍSTICO PRINCIPAL"
display "══════════════════════════════════════════════════════════"
display ""

// Modelo completo con interacciones RUC×Región
logit op2021_original c.ruc##(i.region) ventas_k i.sector i.sexo_gerente productividad_k digital_score tributos_k salarios_k i.tipo_local i.regimen, vce(cluster ciiu_4dig)

// Odds Ratios (IMPORTANTE: ejecutar ANTES de estimates store para que use el modelo activo)
logit, or

// Guardar estimaciones para uso posterior
estimates store modelo_principal

// Bondad de ajuste
fitstat

// VIF test (multicolinealidad)
quietly regress op2021_original productividad_k tributos_k salarios_k digital_score ventas_k i.sexo_gerente i.region i.sector i.regimen i.tipo_local
vif


/*=============================================================================
5. EFECTOS MARGINALES (Secciones 2.3, 3.1, 3.2)
=============================================================================*/

// Restaurar modelo principal
estimates restore modelo_principal

display ""
display "══════════════════════════════════════════════════════════"
display "EFECTOS MARGINALES POR REGIÓN (Sección 2.3)"
display "══════════════════════════════════════════════════════════"
display ""

// Efecto marginal de RUC por región
margins region, dydx(ruc)
marginsplot, title("Efecto marginal de RUC por región") ///
            ytitle("Cambio en Pr(Supervivencia)") xtitle("Región")

// Tests de heterogeneidad
test 1.region#c.ruc 2.region#c.ruc  // H0: Efectos iguales en todas regiones
test 1.region#c.ruc = 0             // H0: Sierra = Costa
test 2.region#c.ruc = 0             // H0: Selva = Costa
test 1.region#c.ruc = 2.region#c.ruc // H0: Sierra = Selva

display ""
display "══════════════════════════════════════════════════════════"
display "EFECTOS POR VENTAS (Sección 3.1)"
display "══════════════════════════════════════════════════════════"
display ""

// Efecto de RUC según escala de ventas
margins region, dydx(ruc) at(ventas_k = (0 (1000) 7478))
marginsplot, title("Efecto de RUC según escala de ventas") ///
            ytitle("Efecto marginal de RUC") xtitle("Ventas (en miles)")

/*=============================================================================
6. ANÁLISIS POR RÉGIMEN TRIBUTARIO (Sección 4.2)
=============================================================================*/

display "══════════════════════════════════════════════════════════"
display "PROBABILIDADES POR RÉGIMEN TRIBUTARIO (Sección 4.2)"
display "══════════════════════════════════════════════════════════"

// Probabilidades predichas por régimen
margins regimen
marginsplot, title("Probabilidad de supervivencia por régimen tributario") ///
            ytitle("Pr(Supervivencia)") xlabel(, angle(45))

// Test de diferencias entre regímenes
test 1.regimen 2.regimen 3.regimen 4.regimen

// Probabilidades condicionales a formalización (RUC=1)
margins regimen, at(ruc=1)

// Comparaciones por pares
margins regimen, at(ruc=1) pwcompare(effects)

/*=============================================================================
7. DIAGNÓSTICOS DEL MODELO (Sección 5.2)
=============================================================================*/

display "══════════════════════════════════════════════════════════"
display "DIAGNÓSTICOS: CAPACIDAD PREDICTIVA"
display "══════════════════════════════════════════════════════════"

// Matriz de clasificación
estat classification

// Curva ROC
lroc, title("Curva ROC - Capacidad predictiva del modelo")

// Criterios de información
estat ic

/*=============================================================================
8. EXPORTACIÓN DE TABLAS CONSOLIDADAS (Formato Paper)
=============================================================================*/

display ""
display "══════════════════════════════════════════════════════════"
display "EXPORTANDO TABLAS PARA PAPER..."
display "══════════════════════════════════════════════════════════"
display ""

// Definir ruta de salida
global output "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Entrega 3 (final)/Paper"

// TABLA 1: Modelo Principal (Coeficientes + OR)
estimates restore modelo_principal
outreg2 using "$output/Tabla1_ModeloPrincipal.xls", excel replace se label ///
        bdec(3) sdec(3) ///
        ctitle("Modelo Principal") ///
        addtext(Errores estandar, Cluster-robust (CIIU 4 digitos), Significancia, *** p<0.01 ** p<0.05 * p<0.10) ///
        addstat(Pseudo R2, e(r2_p), Log-likelihood, e(ll), N, e(N))

// TABLA 2: Especificaciones Alternativas (Robustez)
// Nota: Solo reportamos coeficientes de RUC e interacciones RUC x Region

// Columna (1): Modelo Completo
estimates restore modelo_principal
outreg2 using "$output/Tabla2_Robustez.xls", excel replace se label ///
        bdec(3) sdec(3) ///
        keep(ruc 1.region#c.ruc 2.region#c.ruc) ///
        ctitle("(1) Completo") ///
        addtext(Errores estandar, Cluster-robust, Controles, Todos) ///
        addstat(Pseudo R2, e(r2_p), N, e(N))

// Columna (2): Sin tributos/salarios
estimates restore sin_tributos_salarios
outreg2 using "$output/Tabla2_Robustez.xls", excel append se label ///
        bdec(3) sdec(3) ///
        keep(ruc 1.region#c.ruc 2.region#c.ruc) ///
        ctitle("(2) Sin trib/sal") ///
        addtext(Errores estandar, Cluster-robust, Controles, Sin trib/sal) ///
        addstat(Pseudo R2, e(r2_p), N, e(N))

// Columna (3): Solo controles exogenos
estimates restore solo_exogenos
outreg2 using "$output/Tabla2_Robustez.xls", excel append se label ///
        bdec(3) sdec(3) ///
        keep(ruc 1.region#c.ruc 2.region#c.ruc) ///
        ctitle("(3) Solo exog") ///
        addtext(Errores estandar, Cluster-robust, Controles, Solo exogenos) ///
        addstat(Pseudo R2, e(r2_p), N, e(N))

// Columna (4): Solo microempresas
estimates restore solo_micro
outreg2 using "$output/Tabla2_Robustez.xls", excel append se label ///
        bdec(3) sdec(3) ///
        keep(ruc 1.region#c.ruc 2.region#c.ruc) ///
        ctitle("(4) Micro <50k") ///
        addtext(Errores estandar, Cluster-robust, Controles, Microempresas) ///
        addstat(Pseudo R2, e(r2_p), N, e(N))

// Columna (5): Solo pequenas empresas
estimates restore solo_pequena
outreg2 using "$output/Tabla2_Robustez.xls", excel append se label ///
        bdec(3) sdec(3) ///
        keep(ruc 1.region#c.ruc 2.region#c.ruc) ///
        ctitle("(5) Pequena >=50k") ///
        addtext(Errores estandar, Cluster-robust, Controles, Pequenas) ///
        addstat(Pseudo R2, e(r2_p), N, e(N))

// TABLA 3: Efectos Marginales por Region
estimates restore modelo_principal
margins region, dydx(ruc) post
outreg2 using "$output/Tabla3_EfectosMarginales.xls", excel replace label ///
        bdec(4) ///
        ctitle("Efecto Marginal (pp)") ///
        addtext(Nota, Intervalos de confianza al 95% entre parentesis, Unidad, puntos porcentuales de cambio en Pr(Supervivencia))

display ""
display "══════════════════════════════════════════════════════════"
display "✓ Tablas exportadas exitosamente a: $output"
display "══════════════════════════════════════════════════════════"
display ""


//Notas:
// Instrumentalizar la variable regimen para verificar si el regimen aumenta probabilidad de supervivencia o la mayor probabilidad de supervivencia instrínseca (por factores específicos de la empresa, como prouctividad por trabajador, etc.) hace que la empresa se encuentre en regimenes mas complicados (problema de simultaneidad)
