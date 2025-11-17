// Establecer la ruta de trabajo
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

// Limpiar cualquier dato previo
clear

// Ingresar los datos
input year sales_mypes pib_pct mypes_informal_pct mypes_count
2007 91798  29 89 .
2008 98077  28 88 .
2009 110498 29 86 .
2010 111444 27 88 .
2011 116160 25 88 .
2012 126417 25 85 .
2013 125860 23 85 1513
2014 126570 22 83 1592
2015 131562 22 83 1682
2016 135798 21 80 1728
2017 135264 19 81 1900
2018 140061 19 83 2211
2019 148276 19 84 2377
2020 60489  8  85 1780
end

// Convertir ventas a miles de millones para mejor legibilidad
gen sales_billions = sales_mypes / 1000
gen mypes_formal_pct = 100 - mypes_informal_pct

// Calcular las líneas de tendencia
regress sales_billions year
predict sales_trend, xb

regress pib_pct year
predict pib_trend, xb

regress mypes_informal_pct year
predict informal_trend, xb

// Calcular la tendencia para el número de MYPEs
regress mypes_count year if mypes_count != .
predict mypes_count_trend, xb

// Crear variable auxiliar para posicionar las etiquetas más abajo dentro de las barras
gen label_pos = sales_billions * 0.7

// Crear el gráfico (barras para ventas, líneas para % PIB y % MYPEs informales)
twoway (bar sales_billions year, color("70 146 207") barwidth(0.8)) ///
       (scatter label_pos year, msymbol(none) mlabel(sales_billions) mlabposition(0) mlabcolor(white) mlabsize(small) mlabformat(%9.1f) mlabangle(90)) ///
       (connected pib_pct year, yaxis(2) lcolor("42 41 38") mcolor("42 41 38") msymbol(circle) msize(small) lpattern(dash)) ///
       (scatter pib_pct year, yaxis(2) mlabel(pib_pct) mlabposition(12) mlabgap(1.5) mlabcolor("42 41 38") msymbol(none) mlabsize(small) mlabformat(%9.0f) mlabstyle(box)) ///
       (connected mypes_informal_pct year, yaxis(2) lcolor("100 100 100") mcolor("100 100 100") msymbol(triangle) msize(small) lpattern(solid)) ///
       (scatter mypes_informal_pct year, yaxis(2) mlabel(mypes_informal_pct) mlabposition(12) mlabgap(1.5) mlabcolor("100 100 100") msymbol(none) mlabsize(small) mlabformat(%9.0f) mlabstyle(box)), ///
       title("Ventas Totales de MYPEs, Contribución al PIB y" "Porcentaje de MYPEs informales en Perú 2007-2020", margin(b+5)) ///
       xtitle("Año") ///
       ytitle("Ventas Totales (miles de millones)", axis(1) margin(r+4)) ///
       ytitle("Porcentaje (%)", axis(2) margin(l+4)) ///
       xlabel(2006(2)2021, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(0(50)150, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(0(20)100, axis(2) nogrid angle(horizontal) format(%9.0f)) /// Escala ajustada para incluir ambos porcentajes
       legend(label(1 "Ventas Totales (miles de millones)") label(3 "Contribución al PIB (%)") label(5 "MYPEs Informales (%)") order(1 3 5) position(6) ring(1) span cols(3) size(small) bmargin(t+2)) ///
       // note("Fuente: ENAHO - ComexPerú. Elaboración propia") ///
       plotregion(margin(l=10 r=10 t=10 b=10)) ///
       scheme(s1color)
	   
// Crear el gráfico de líneas de tendencia con puntos más pequeños
twoway (connected sales_trend year, lcolor("70 146 207") lwidth(medium) lpattern(solid) msymbol(circle) msize(small) mcolor("70 146 207")) ///
       (connected pib_trend year, yaxis(2) lcolor("42 41 38") lwidth(medium) lpattern(dash) msymbol(circle) msize(small) mcolor("42 41 38")) ///
       (connected informal_trend year, yaxis(2) lcolor("100 100 100") lwidth(medium) lpattern(dash) msymbol(triangle) msize(small) mcolor("100 100 100")), ///
       title("Tendencias de Ventas Totales de MYPEs, Contribución al PIB y" "Porcentaje de MYPEs Informales en Perú 2007-2020", margin(b+5)) ///
       xtitle("Año") ///
       ytitle("Ventas Totales (miles de millones)", axis(1) margin(r+4)) ///
       ytitle("Porcentaje (%)", axis(2) margin(l+4)) ///
       xlabel(2006(2)2021, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(50(25)150, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(0(20)100, axis(2) nogrid angle(horizontal) format(%9.0f)) ///
       legend(label(1 "Tendencia Ventas Totales de MYPEs (miles de millones)") label(2 "Tendencia Contribución al PIB (%)") label(3 "Tendencia MYPEs Informales (%)") order(1 2 3) position(6) ring(1) span cols(2) stack size(small) bmargin(t+2)) ///
       // note("Fuente: ENAHO - ComexPerú. Elaboración propia") ///
       plotregion(margin(l=10 r=10 t=10 b=10)) ///
       scheme(s1color)
	   
	   
// Crear el gráfico (barras para número de MYPEs, línea para % MYPEs formales)
twoway (bar mypes_count year if year >= 2012, color("70 146 207") barwidth(0.8)) ///
       (scatter mypes_count year if year >= 2012, msymbol(none) mlabel(mypes_count) mlabposition(12) mlabgap(1) mlabcolor("3 81 143") mlabsize(small) mlabformat(%9.0f)) ///
       (connected mypes_formal_pct year if year >= 2013, yaxis(2) lcolor("100 100 100") mcolor("100 100 100") msymbol(circle) msize(small) lpattern(dash)) ///
       (scatter mypes_formal_pct year if year >= 2013, yaxis(2) mlabel(mypes_formal_pct) mlabposition(12) mlabgap(1.5) mlabcolor("100 100 100") msymbol(none) mlabsize(small) mlabformat(%9.0f) mlabstyle(box)) ///
       (connected mypes_informal_pct year if year >= 2013, yaxis(2) lcolor("42 41 38") mcolor("42 41 38") msymbol(triangle) msize(small) lpattern(solid)) ///
       (scatter mypes_informal_pct year if year >= 2013, yaxis(2) mlabel(mypes_informal_pct) mlabposition(12) mlabgap(1.5) mlabcolor("42 41 38") msymbol(none) mlabsize(small) mlabformat(%9.0f) mlabstyle(box)), ///
       title("Número de MYPEs, Porcentaje de MYPEs Formales e Informales en Perú" "2013-2021", margin(b+5)) ///
       xtitle("Año") ///
       ytitle("Número de MYPEs (miles)", axis(1) margin(r+4)) ///
       ytitle("Porcentaje (%)", axis(2) margin(l+2 r+4)) ///
       xlabel(2012(1)2022, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(1000(500)2500, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(0(20)100, axis(2) nogrid angle(horizontal) format(%9.0f)) /// Escala ajustada para ambos porcentajes
       legend(label(1 "Número de MYPEs (miles)") label(3 "MYPEs Formales (%)") label(5 "MYPEs Informales (%)") order(1 3 5) position(6) ring(1) span cols(3) size(small) bmargin(t+2)) ///
       // note("Fuente: ENAHO - ComexPerú. Elaboración propia") ///
       plotregion(margin(l=10 r=10 t=10 b=10)) ///
       scheme(s1color)
	   

// Exportar el gráfico
graph export "ventas_mypes_pib_informal_2007_2020.png", replace width(1200)
