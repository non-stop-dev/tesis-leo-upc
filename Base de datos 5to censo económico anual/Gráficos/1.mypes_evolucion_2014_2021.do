cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

*------------------------ Evolución de las MYPEs Formales 2014-2021 -----------------------*

// Limpiar cualquier dato previo
clear

// Ingresar los datos manualmente
input year mypes_formales tasa_variacion
2014 1592232 5.2
2015 1682681 5.7
2016 1728777 2.7
2017 1899584 9.9
2018 2211981 16.4
2019 2377244 7.5
2020 1780117 -25.1
2021 2118293 19.0
end

// Convertir mypes_formales a miles
gen mypes_formales_miles = mypes_formales / 1000

// Calcular la línea de tendencia para mypes_formales_miles
regress mypes_formales_miles year
predict mypes_trend, xb

// Calcular la línea de tendencia para tasa_variacion
regress tasa_variacion year
predict tasa_trend, xb

// Guardar los datos (opcional, si quieres conservarlos)
save "mypes_evolucion_2014_2021.dta", replace

// Crear el gráfico con los ajustes solicitados
twoway (bar mypes_formales_miles year, color("70 146 207") barwidth(0.8)) ///
       (scatter mypes_formales_miles year, msymbol(none) mlabel(mypes_formales_miles) mlabposition(12) mlabgap(1) mlabcolor("3 81 143") mlabsize(small) mlabformat(%9.0f)) ///
       (line mypes_trend year, lcolor("53 146 230") lwidth(thin) lpattern(dash)) ///
       (connected tasa_variacion year, yaxis(2) lcolor("235 148 19") mcolor("235 148 19") msymbol(circle) msize(medium) lpattern(dash)) ///
       (scatter tasa_variacion year, yaxis(2) mlabel(tasa_variacion) mlabposition(12) mlabgap(1.5) mlabcolor("192 115 0") msymbol(none) mlabsize(small) mlabformat(%9.1f) mlabstyle(box)) ///
       (line tasa_trend year, yaxis(2) lcolor("255 180 90") lwidth(vthin) lpattern(dash)), ///
       title("Evolución de las MYPEs Formales (2014-2021)", margin(b+5)) ///
       xtitle("Año") ///
       ytitle("Número de MYPEs Formales (miles)", axis(1) margin(r+4)) ///
       ytitle("Tasa de Variación (%)", axis(2) margin(l+4)) ///
       ylabel(1300(200)2500, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(-30(10)20, axis(2) nogrid angle(horizontal)) ///
	   legend(label(1 "MYPEs Formales (en miles)") label(3 "Tendencia MYPEs") label(4 "Tasa de Variación (%)") label(6 "Tendencia Tasa de Variación") order(1 4 3 6) position(6) ring(1) span cols(2) size(small) bmargin(t+2))
       // note("Nota: El estrato empresarial es determinado de acuerdo con la ley N° 30056." "Fuente: Sunat, Registro Único del Contribuyente 2013-2017 / PRODUCE-OEE. Elaboración Propia") ///
       plotregion(margin(l=10 r=10 t=10 b=10)) ///
       scheme(s1color)
	   
	   
// Exportar el gráfico (opcional)
graph export "mypes_evolucion_2014_2021.png", replace width(1200)

*-----------------------------------------------------------------------------------------*
