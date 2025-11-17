* ==============================================================================
* TABLA Y GRÁFICO DE PIE: EMPRESAS POR ORGANIZACIÓN JURÍDICA SEGÚN SEGMENTO
* ==============================================================================

clear all
set more off

* Crear la base de datos con los valores del Cuadro N° 3 en orden consistente
input str35 organizacion_juridica str25 segmento long absoluto float porcentaje orden_org

"Persona Natural" "Total" 2194046 73.6 1
"Sociedad Anónima" "Total" 374852 12.6 2
"Sociedad Civil" "Total" 9481 0.3 3
"Sociedad Comercial de Resp. Ltda." "Total" 68452 2.3 4
"Empresa Individual de Resp. Ltda." "Total" 246340 8.3 5
"Asociaciones" "Total" 48222 1.6 6
"Otros" "Total" 38314 1.3 7

"Persona Natural" "Microempresa" 2186138 75.4 1
"Sociedad Anónima" "Microempresa" 328988 11.4 2
"Sociedad Civil" "Microempresa" 8957 0.3 3
"Sociedad Comercial de Resp. Ltda." "Microempresa" 61059 2.1 4
"Empresa Individual de Resp. Ltda." "Microempresa" 230783 8.0 5
"Asociaciones" "Microempresa" 47295 1.6 6
"Otros" "Microempresa" 35401 1.2 7

"Persona Natural" "Pequeña empresa" 7542 11.2 1
"Sociedad Anónima" "Pequeña empresa" 35951 53.4 2
"Sociedad Civil" "Pequeña empresa" 423 0.6 3
"Sociedad Comercial de Resp. Ltda." "Pequeña empresa" 6288 9.3 4
"Empresa Individual de Resp. Ltda." "Pequeña empresa" 14264 21.2 5
"Asociaciones" "Pequeña empresa" 714 1.1 6
"Otros" "Pequeña empresa" 2139 3.2 7

"Persona Natural" "Gran y mediana empresa" 366 2.7 1
"Sociedad Anónima" "Gran y mediana empresa" 9913 72.0 2
"Sociedad Civil" "Gran y mediana empresa" 101 0.8 3
"Sociedad Comercial de Resp. Ltda." "Gran y mediana empresa" 1105 8.0 4
"Empresa Individual de Resp. Ltda." "Gran y mediana empresa" 1293 9.4 5
"Asociaciones" "Gran y mediana empresa" 213 1.5 6
"Otros" "Gran y mediana empresa" 774 5.6 7

end

* Etiquetar las variables
label variable organizacion_juridica "Organización jurídica"
label variable segmento "Segmento empresarial"
label variable absoluto "Número de empresas"
label variable porcentaje "Porcentaje (%)"
label variable orden_org "Orden de organización"

* Crear etiquetas más cortas para los gráficos
gen org_corta = ""
replace org_corta = "Persona Natural" if organizacion_juridica == "Persona Natural"
replace org_corta = "Sociedad Anónima" if organizacion_juridica == "Sociedad Anónima"
replace org_corta = "Sociedad Civil" if organizacion_juridica == "Sociedad Civil"
replace org_corta = "SRL" if organizacion_juridica == "Sociedad Comercial de Resp. Ltda."
replace org_corta = "EIRL" if organizacion_juridica == "Empresa Individual de Resp. Ltda."
replace org_corta = "Asociaciones" if organizacion_juridica == "Asociaciones"
replace org_corta = "Otros" if organizacion_juridica == "Otros"

* Ordenar por segmento y luego por orden de organización
sort segmento orden_org

* Mostrar la tabla
display ""
display "TABLA: EMPRESAS POR ORGANIZACIÓN JURÍDICA SEGÚN SEGMENTO EMPRESARIAL, 2021"
display "============================================================================="
list organizacion_juridica segmento absoluto porcentaje, clean noobs separator(7)

* ==============================================================================
* GRÁFICOS DE PIE POR SEGMENTO EMPRESARIAL
* ==============================================================================

* Gráfico de pie para el Total
preserve
keep if segmento == "Total"
sort orden_org

graph pie porcentaje, over(org_corta) sort(orden_org) ///
    pie(1, color("255 230 153")) ///
    pie(2, color("255 204 153")) ///
    pie(3, color("255 179 102")) ///
    pie(4, color("255 153 102")) ///
    pie(5, color("255 128 77")) ///
    pie(6, color("255 102 51")) ///
    pie(7, color("204 85 34")) ///
    plabel(1 percent, format(%9.1f) size(small) color(black) gap(5)) ///
    plabel(2 percent, format(%9.1f) size(small) color(black) gap(5)) ///
    plabel(3 percent, format(%9.1f) size(vsmall) color(black) gap(15)) ///
    plabel(4 percent, format(%9.1f) size(vsmall) color(black) gap(15)) ///
    plabel(5 percent, format(%9.1f) size(small) color(black) gap(10)) ///
    plabel(6 percent, format(%9.1f) size(vsmall) color(black) gap(15)) ///
    plabel(7 percent, format(%9.1f) size(vsmall) color(black) gap(15)) ///
    title("Distribución de Empresas por Organización Jurídica", ///
          size(medsmall) margin(b+3)) ///
    subtitle("Total Nacional, Perú 2021", size(small) margin(b+2)) ///
    legend(position(3) ring(1) size(vsmall) cols(1)) ///
    note("Fuente: INEI – Directorio Central de Empresas y Establecimientos. Elaboración Propia.", ///
         size(vsmall) margin(t+2)) ///
    graphregion(margin(l=10 r=10 t=5 b=5)) ///
    scheme(s1color)

