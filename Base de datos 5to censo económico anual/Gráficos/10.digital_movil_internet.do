* Set working directory (adjust as needed)
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

clear
set more off

* Creating dataset with technology access data
input year telefonia_fija telefonia_movil television_cable computadora internet
2010 30.4 73.1 26.0 23.4 13.0
2011 29.8 75.2 29.8 25.4 16.4
2012 29.4 79.7 31.9 29.9 20.2
2013 28.6 82.0 33.6 32.0 22.1
2014 26.9 84.9 35.9 32.3 23.5
2015 24.5 87.2 36.7 32.6 23.2
2016 23.5 88.9 37.1 33.5 26.4
2017 21.9 90.2 37.4 33.2 28.2
2018 20.6 90.9 37.7 33.3 29.8
2019 19.1 92.1 38.0 32.7 35.9
2020 13.6 95.0 31.0 33.3 38.7
end

* Label variables
label variable year "Año"
label variable telefonia_fija "Telefonía fija (%)"
label variable telefonia_movil "Telefonía móvil (%)"
label variable television_cable "Televisión por cable (%)"
label variable computadora "Computadora (%)"
label variable internet "Internet (%)"

* Keep only year, internet, and telefonia_movil
keep year internet telefonia_movil

* Calculate trend lines
regress internet year
predict internet_trend, xb

regress telefonia_movil year
predict telefonia_movil_trend, xb

* Save the dataset
save "tech_access_peru_reduced_2010_2020.dta", replace

* Create the line graph
twoway (connected internet year, lcolor("70 146 207") mcolor("70 146 207") msymbol(circle) msize(medium)) ///
       (scatter internet year, msymbol(none) mlabel(internet) mlabposition(12) mlabgap(1.5) mlabcolor("3 81 143") mlabsize(small) mlabformat(%9.1f)) ///
       (line internet_trend year, lcolor("70 146 207*0.5") lwidth(vthin) lpattern(dash)) ///
       (connected telefonia_movil year, lcolor("235 148 19") mcolor("235 148 19") msymbol(circle) msize(medium)) ///
       (scatter telefonia_movil year, msymbol(none) mlabel(telefonia_movil) mlabposition(12) mlabgap(1.5) mlabcolor("192 115 0") mlabsize(small) mlabformat(%9.1f)) ///
       (line telefonia_movil_trend year, lcolor("235 148 19*0.5") lwidth(vthin) lpattern(dash)), ///
       title("Evolución del Acceso a Internet y Telefonía Móvil en Perú" "(2010-2020)", margin(b+5)) ///
       xtitle("Año") ///
       ytitle("Porcentaje de hogares (%)", margin(r+4)) ///
       xlabel(2010(1)2020, labsize(small)) ///
       ylabel(0(20)100, nogrid angle(horizontal) format(%9.0f)) ///
       legend(label(1 "Internet (%)") label(3 "Tendencia Internet") label(4 "Telefonía móvil (%)") label(6 "Tendencia Telefonía móvil") order(1 3 4 6) position(6) ring(1) span cols(2) size(small) bmargin(t+2)) ///
    //note("Fuente: Instituto Nacional de Estadística e Informática - Encuesta Nacional de Hogares. Elaboración Propia.") ///
       plotregion(margin(l=10 r=10 t=10 b=10)) ///
       scheme(s1color)

