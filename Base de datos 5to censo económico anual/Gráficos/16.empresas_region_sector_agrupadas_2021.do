* ==============================================================================
* TABLA: EMPRESAS POR DEPARTAMENTO Y SECTOR ECONÓMICO - PERÚ 2021
* Sectores: Manufactura, Comercio, Servicios
* ==============================================================================

clear all
set more off

* Crear la base de datos con empresas por departamento y sector
input str35 departamento str15 sector long absoluto float porcentaje str10 region

"Nacional" "Manufactura" 226737 100.0 "Total"
"Amazonas" "Manufactura" 1171 0.5 "Selva"
"Áncash" "Manufactura" 5198 2.3 "Sierra"
"Apurímac" "Manufactura" 1950 0.9 "Sierra"
"Arequipa" "Manufactura" 12695 5.4 "Sierra"
"Ayacucho" "Manufactura" 2490 1.1 "Sierra"
"Cajamarca" "Manufactura" 4989 2.2 "Sierra"
"Callao" "Manufactura" 7260 3.2 "Costa"
"Cusco" "Manufactura" 8331 3.7 "Sierra"
"Huancavelica" "Manufactura" 976 0.4 "Sierra"
"Huánuco" "Manufactura" 3409 1.5 "Sierra"
"Ica" "Manufactura" 3841 1.7 "Costa"
"Junín" "Manufactura" 7836 3.5 "Sierra"
"La Libertad" "Manufactura" 11744 5.1 "Costa"
"Lambayeque" "Manufactura" 6495 2.9 "Costa"
"Lima" "Manufactura" 116895 51.6 "Costa"
"Loreto" "Manufactura" 3155 1.4 "Selva"
"Madre de Dios" "Manufactura" 1309 0.6 "Selva"
"Moquegua" "Manufactura" 1072 0.5 "Costa"
"Pasco" "Manufactura" 1006 0.4 "Sierra"
"Piura" "Manufactura" 7426 3.3 "Costa"
"Puno" "Manufactura" 7163 3.2 "Sierra"
"San Martín" "Manufactura" 4222 1.9 "Selva"
"Tacna" "Manufactura" 2912 1.3 "Costa"
"Tumbes" "Manufactura" 753 0.3 "Costa"
"Ucayali" "Manufactura" 2439 1.1 "Selva"
"Lima Metropolitana" "Manufactura" 113361 50.0 "Costa"
"Resto de Lima" "Manufactura" 3534 1.6 "Costa"

"Nacional" "Comercio" 1324819 100.0 "Total"
"Amazonas" "Comercio" 7010 0.6 "Selva"
"Áncash" "Comercio" 39423 3.0 "Sierra"
"Apurímac" "Comercio" 11579 0.9 "Sierra"
"Arequipa" "Comercio" 73015 5.5 "Sierra"
"Ayacucho" "Comercio" 16610 1.3 "Sierra"
"Cajamarca" "Comercio" 29646 2.2 "Sierra"
"Callao" "Comercio" 41595 3.1 "Costa"
"Cusco" "Comercio" 52055 3.9 "Sierra"
"Huancavelica" "Comercio" 6306 0.5 "Sierra"
"Huánuco" "Comercio" 20999 1.6 "Sierra"
"Ica" "Comercio" 37329 2.8 "Costa"
"Junín" "Comercio" 50925 3.8 "Sierra"
"La Libertad" "Comercio" 75046 5.7 "Costa"
"Lambayeque" "Comercio" 44374 3.3 "Costa"
"Lima" "Comercio" 593130 44.8 "Costa"
"Loreto" "Comercio" 26771 2.0 "Selva"
"Madre de Dios" "Comercio" 9280 0.7 "Selva"
"Moquegua" "Comercio" 7567 0.6 "Costa"
"Pasco" "Comercio" 7546 0.6 "Sierra"
"Piura" "Comercio" 60322 4.6 "Costa"
"Puno" "Comercio" 32371 2.4 "Sierra"
"San Martín" "Comercio" 27693 2.1 "Selva"
"Tacna" "Comercio" 21839 1.6 "Costa"
"Tumbes" "Comercio" 11065 0.8 "Costa"
"Ucayali" "Comercio" 21323 1.6 "Selva"
"Lima Metropolitana" "Comercio" 557390 42.3 "Costa"
"Resto de Lima" "Comercio" 35740 2.8 "Costa"

"Nacional" "Servicios" 1246659 100.0 "Total"
"Amazonas" "Servicios" 10335 0.8 "Selva"
"Áncash" "Servicios" 33590 2.7 "Sierra"
"Apurímac" "Servicios" 13338 1.1 "Sierra"
"Arequipa" "Servicios" 67746 5.3 "Sierra"
"Ayacucho" "Servicios" 18024 1.4 "Sierra"
"Cajamarca" "Servicios" 30687 2.5 "Sierra"
"Callao" "Servicios" 47421 3.8 "Costa"
"Cusco" "Servicios" 47552 3.8 "Sierra"
"Huancavelica" "Servicios" 5733 0.5 "Sierra"
"Huánuco" "Servicios" 18652 1.5 "Sierra"
"Ica" "Servicios" 28068 2.3 "Costa"
"Junín" "Servicios" 41333 3.3 "Sierra"
"La Libertad" "Servicios" 61011 4.9 "Costa"
"Lambayeque" "Servicios" 50018 4.0 "Costa"
"Lima" "Servicios" 575774 46.2 "Costa"
"Loreto" "Servicios" 21896 1.8 "Selva"
"Madre de Dios" "Servicios" 7477 0.6 "Selva"
"Moquegua" "Servicios" 9128 0.7 "Costa"
"Pasco" "Servicios" 7088 0.6 "Sierra"
"Piura" "Servicios" 55458 4.4 "Costa"
"Puno" "Servicios" 27129 2.2 "Sierra"
"San Martín" "Servicios" 25138 2.0 "Selva"
"Tacna" "Servicios" 17068 1.4 "Costa"
"Tumbes" "Servicios" 9582 0.8 "Costa"
"Ucayali" "Servicios" 17413 1.4 "Selva"
"Lima Metropolitana" "Servicios" 543976 43.6 "Costa"
"Resto de Lima" "Servicios" 31798 2.6 "Costa"

end

* Verificar y corregir la clasificación regional según la nueva especificación
replace region = "Costa" if inlist(departamento, "Callao", "Ica", "La Libertad", "Lambayeque", "Lima", "Moquegua", "Piura", "Tacna", "Tumbes")
replace region = "Costa" if inlist(departamento, "Lima Metropolitana", "Resto de Lima")
replace region = "Sierra" if inlist(departamento, "Áncash", "Apurímac", "Arequipa", "Ayacucho")
replace region = "Sierra" if inlist(departamento, "Cajamarca", "Cusco", "Huancavelica", "Huánuco", "Junín", "Pasco", "Puno")
replace region = "Selva" if inlist(departamento, "Amazonas", "Loreto", "Madre de Dios", "San Martín", "Ucayali")

* Etiquetar las variables
label variable departamento "Departamento"
label variable sector "Sector económico"
label variable absoluto "Número de empresas"
label variable porcentaje "Porcentaje del total nacional por sector"
label variable region "Región natural"

* Mostrar la tabla completa por sector
display ""
display "TABLA: EMPRESAS POR DEPARTAMENTO Y SECTOR ECONÓMICO - PERÚ 2021"
display "================================================================"
display ""
display "SECTOR MANUFACTURA:"
display "==================="
list departamento absoluto porcentaje region if sector == "Manufactura", clean noobs

display ""
display "SECTOR COMERCIO:"
display "================"
list departamento absoluto porcentaje region if sector == "Comercio", clean noobs

display ""
display "SECTOR SERVICIOS:"
display "================="
list departamento absoluto porcentaje region if sector == "Servicios", clean noobs

* ==============================================================================
* AGREGAR TOTALES POR REGIÓN Y SECTOR
* ==============================================================================

* Calcular totales por región y sector (excluyendo el total nacional)
preserve
drop if departamento == "Nacional"

* Agrupar por región y sector
collapse (sum) absoluto, by(region sector)

* Calcular porcentajes por sector
gen porcentaje = .
replace porcentaje = (absoluto/226737)*100 if sector == "Manufactura"
replace porcentaje = (absoluto/1324819)*100 if sector == "Comercio"
replace porcentaje = (absoluto/1246659)*100 if sector == "Servicios"

* Mostrar totales por región y sector
display ""
display "TOTALES POR REGIÓN NATURAL Y SECTOR:"
display "====================================="
sort sector region
list sector region absoluto porcentaje, clean noobs

restore

* ==============================================================================
* PREPARAR DATOS PARA GRÁFICOS
* ==============================================================================

* ==============================================================================
* PREPARAR DATOS PARA GRÁFICOS DE BARRAS AGRUPADAS POR REGIÓN
* ==============================================================================

* Crear base resumida por región y sector para gráficos
preserve
drop if departamento == "Nacional" | departamento == "Lima Metropolitana" | departamento == "Resto de Lima"

* Agrupar por región y sector
collapse (sum) absoluto, by(region sector)

* Calcular porcentajes por sector
gen porcentaje = .
replace porcentaje = (absoluto/226737)*100 if sector == "Manufactura"
replace porcentaje = (absoluto/1324819)*100 if sector == "Comercio" 
replace porcentaje = (absoluto/1246659)*100 if sector == "Servicios"

* Mostrar resumen antes de crear gráficos
display ""
display "RESUMEN POR REGIÓN Y SECTOR PARA GRÁFICOS:"
display "==========================================="
list region sector absoluto porcentaje, clean noobs separator(3)

* Reorganizar datos para gráfico de barras agrupadas
reshape wide absoluto porcentaje, i(region) j(sector) string

* Calcular totales por región
gen total_empresas = absolutoComercio + absolutoManufactura + absolutoServicios
gen porcentaje_total = (total_empresas/(1324819+226737+1246659))*100

* Convertir a miles para mejor visualización
gen comercio_miles = absolutoComercio/1000
gen manufactura_miles = absolutoManufactura/1000
gen servicios_miles = absolutoServicios/1000
gen total_miles = total_empresas/1000

* Crear etiquetas de región ordenadas
gen orden_region = .
replace orden_region = 1 if region == "Costa"
replace orden_region = 2 if region == "Sierra"
replace orden_region = 3 if region == "Selva"

sort orden_region

* Mostrar datos finales para gráfico
display ""
display "DATOS FINALES PARA GRÁFICO (en miles de empresas):"
display "=================================================="
list region manufactura_miles comercio_miles servicios_miles total_miles, clean noobs

* ==============================================================================
* GRÁFICO DE BARRAS AGRUPADAS POR REGIÓN Y SECTOR (SIN TOTALES)
* ==============================================================================

graph bar manufactura_miles comercio_miles servicios_miles, ///
    over(region, label(labsize(small)) sort(orden_region)) ///
    bar(1, color("82 130 190")) ///
    bar(2, color("236 160 160")) ///
    bar(3, color("160 208 200")) ///
    blabel(bar, format(%9.0f) color("60 60 60") size(vsmall) position(outside)) ///
    title("Distribución de Empresas por Región según Sector Económico" "(Perú, 2021)", ///
          size(medsmall) margin(b+5)) ///
    ytitle("Número de empresas (miles)", margin(r+4)) ///
    ylabel(0(250)1000, nogrid angle(horizontal) format(%9.0f)) ///
    yscale(range(0 750)) ///
    legend(label(1 "Manufactura") label(2 "Comercio") label(3 "Servicios") ///
           position(6) ring(1) span cols(3) size(small) bmargin(t+2)) ///
    note("Fuente: INEI – Directorio Central de Empresas y Establecimientos. Elaboración Propia.", ///
         size(vsmall) margin(t+3)) ///
    plotregion(margin(l=10 r=10 t=10 b=0)) ///
    graphregion(margin(l=10 r=10 t=10 b=10)) ///
    scheme(s1color)

