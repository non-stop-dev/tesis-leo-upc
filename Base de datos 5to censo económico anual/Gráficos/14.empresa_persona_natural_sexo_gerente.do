* ==============================================================================
* CREACIÓN DE TABLA: EMPRESAS POR SEXO DEL CONDUCTOR SEGÚN SEGMENTO EMPRESARIAL
* CON BARRA ADICIONAL DE PARTICIPACIÓN EN EL TOTAL
* Fuente: INEI - Directorio Central de Empresas y Establecimientos, 2021
* ==============================================================================

clear all
set more off

* Crear la base de datos con los valores del Cuadro N° 4.1
input str30 segmento_empresarial str10 sexo long absoluto float porcentaje

"Total" "Hombres" 1078111 49.1
"Total" "Mujeres" 1115935 50.9
"Microempresa" "Hombres" 1073114 49.1
"Microempresa" "Mujeres" 1113024 50.9
"Pequeña empresa" "Hombres" 4782 63.4
"Pequeña empresa" "Mujeres" 2760 36.6
"Gran y mediana empresa" "Hombres" 215 58.7
"Gran y mediana empresa" "Mujeres" 151 41.3

end

* Etiquetar las variables
label variable segmento_empresarial "Segmento empresarial"
label variable sexo "Sexo del conductor"
label variable absoluto "Número de empresas"
label variable porcentaje "Porcentaje (%)"

* Crear variable numérica para ordenamiento
gen orden = .
replace orden = 1 if segmento_empresarial == "Total"
replace orden = 2 if segmento_empresarial == "Microempresa"
replace orden = 3 if segmento_empresarial == "Pequeña empresa"
replace orden = 4 if segmento_empresarial == "Gran y mediana empresa"

* Ordenar los datos
sort orden sexo

* Mostrar la tabla
display ""
display "TABLA: EMPRESAS POR SEXO DEL CONDUCTOR SEGÚN SEGMENTO EMPRESARIAL, 2021"
display "======================================================================"
list segmento_empresarial sexo absoluto porcentaje, clean noobs separator(2)

* Verificaciones
display ""
display "VERIFICACIONES:"
display "==============="

* Verificar totales por segmento
bysort segmento_empresarial: egen total_segmento = total(absoluto)
display "Totales por segmento empresarial:"
list segmento_empresarial total_segmento if sexo == "Hombres", clean noobs

* Verificar que porcentajes sumen 100%
bysort segmento_empresarial: egen suma_porcentajes = total(porcentaje)
display ""
display "Suma de porcentajes por segmento (debe ser 100%):"
list segmento_empresarial suma_porcentajes if sexo == "Hombres", clean noobs

* ==============================================================================
* GRÁFICO DE BARRAS: EMPRESAS POR SEXO Y PARTICIPACIÓN EN EL TOTAL
* ==============================================================================

* Preparar datos para el gráfico
preserve

* Primero calcular el total general real (suma de la fila "Total")
egen total_general_real = total(absoluto) if segmento_empresarial == "Total"
summarize total_general_real
local total_general = r(max)

display "Total general de empresas: `total_general'"

* Ahora trabajar solo con los segmentos (excluir "Total")
drop if segmento_empresarial == "Total"

* Calcular participación de cada segmento en el total real
bysort segmento_empresarial: egen total_por_segmento = total(absoluto)
gen participacion_total = (total_por_segmento / `total_general') * 100

* Mostrar los cálculos para verificación
display ""
display "CÁLCULOS DE PARTICIPACIÓN:"
display "========================="
list segmento_empresarial total_por_segmento participacion_total if sexo == "Hombres", clean noobs

* Reordenar correctamente los segmentos
drop orden
gen orden = .
replace orden = 1 if segmento_empresarial == "Microempresa"
replace orden = 2 if segmento_empresarial == "Pequeña empresa"
replace orden = 3 if segmento_empresarial == "Gran y mediana empresa"

* Crear variables separadas para hombres y mujeres para gráfico agrupado
reshape wide absoluto porcentaje participacion_total total_por_segmento, i(segmento_empresarial orden) j(sexo) string

* Renombrar variables para mayor claridad
rename absolutoHombres abs_hombres
rename absolutoMujeres abs_mujeres
rename porcentajeHombres pct_hombres
rename porcentajeMujeres pct_mujeres
rename participacion_totalHombres participacion_total

* Crear etiquetas más cortas para el gráfico en el orden correcto
gen segmento_corto = ""
replace segmento_corto = "Microempresa" if segmento_empresarial == "Microempresa"
replace segmento_corto = "Pequeña empresa" if segmento_empresarial == "Pequeña empresa"
replace segmento_corto = ("Mediana y grande") if segmento_empresarial == "Gran y mediana empresa"

* Ordenar correctamente
sort orden

* Gráfico de barras con porcentajes por sexo + participación en el total
graph bar pct_hombres pct_mujeres participacion_total, over(segmento_corto, label(labsize(small)) sort(orden)) ///
    bar(1, color("82 130 190")) ///
    bar(2, color("236 160 160")) ///
    bar(3, color("255 140 0")) ///
    blabel(bar, format(%9.2f) color("60 60 60") size(vsmall) position(outside)) ///
    title("Empresas Persona Natural. Distribución por Sexo" "del Gerente y Participación Total (Perú, 2021)", ///
          size(medsmall) margin(b+5)) ///
    ytitle("Porcentaje (%)", margin(r+1)) ///
    ylabel(0(20)100, nogrid angle(horizontal) format(%9.0f)) ///
    yscale(range(0 100)) ///
    legend(label(1 "Hombres") label(2 "Mujeres") label(3 "% del Total de Empresas") ///
           position(6) ring(1) span cols(3) size(vsmall) bmargin(t+2) bmargin(l+4)) ///
    note("Fuente: INEI – Directorio Central de Empresas y Establecimientos. Elaboración Propia." ///
         "La barra naranja muestra el porcentaje que representa cada segmento del total de empresas.", ///
         size(vsmall) margin(t+3)) ///
    plotregion(margin(l=10 r=10 t=10 b=0)) ///
    graphregion(margin(l=10 r=10 t=10 b=10)) ///
    scheme(s1color)

