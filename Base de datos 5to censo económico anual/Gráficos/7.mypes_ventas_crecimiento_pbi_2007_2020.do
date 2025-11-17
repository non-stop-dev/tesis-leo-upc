// Establecer la ruta de trabajo
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

// Limpiar cualquier dato previo
clear

// Ingresar los datos
input year sales_mypes pbi_total
2007 91798  319693
2008 98077  348870
2009 110498 352693
2010 111444 382081
2011 116160 406256
2012 126417 431199
2013 125860 456435
2014 126570 467308
2015 131562 482506
2016 135798 501581
2017 135264 514215
2018 140061 534665
2019 148276 546161
2020 60489  485474
end

// Convertir ventas y PBI a miles de millones
gen sales_billions = sales_mypes / 1000
gen pbi_billions = pbi_total / 1000

// Calcular líneas de tendencia
regress sales_billions year
predict sales_trend, xb

regress pbi_billions year
predict pbi_trend, xb

// Crear el gráfico (barras para ventas, línea para PBI total, líneas de tendencia)
twoway (bar sales_billions year, color("70 146 207") barwidth(0.8)) ///
       (scatter sales_billions year, msymbol(none) mlabel(sales_billions) mlabposition(12) mlabgap(1) mlabcolor("3 81 143") mlabsize(small) mlabformat(%9.1f)) ///
       (connected pbi_billions year, yaxis(2) lcolor(black) mcolor(black) msymbol(circle) msize(medium) lpattern(dash)) ///
       (scatter pbi_billions year, yaxis(2) mlabel(pbi_billions) mlabposition(12) mlabgap(1.5) mlabcolor(black) msymbol(none) mlabsize(small) mlabformat(%9.0f) mlabstyle(box)) ///
       (line sales_trend year, lcolor("70 146 207") lwidth(vthin) lpattern(dash)) /// Línea de tendencia para ventas
       (line pbi_trend year, yaxis(2) lcolor("50 50 50") lwidth(vthin) lpattern(dash)), /// Línea de tendencia para PBI
       title("Ventas Totales de MYPEs y PBI Total del Perú" "2007-2020", margin(b+5)) ///
       xtitle("Año") ///
       ytitle("Ventas MYPEs (miles de millones de soles)", axis(1) margin(r+4)) ///
       ytitle("PBI Total (miles de millones de soles)", axis(2) margin(l+4)) ///
	   xlabel(2006(2)2021, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(0(50)150, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(300(50)600, axis(2) nogrid angle(horizontal) format(%9.0f)) ///
       legend(label(1 "Ventas Totales MYPEs (miles de millones)") label(3 "PBI Total (miles de millones)") label(5 "Tendencia Ventas MYPEs") label(6 "Tendencia PBI Total") order(1 3 5 6) position(6) ring(1) span cols(2) stack size(small) bmargin(t+2)) ///
       // note("Fuente: Elaboración propia con datos del BCRP y [fuente de MYPEs]") ///
       plotregion(margin(l=10 r=10 t=10 b=10)) ///
       scheme(s1color)

// Exportar el gráfico
graph export "ventas_mypes_pbi_total_2007_2020_with_trends.png", replace width(1200)
