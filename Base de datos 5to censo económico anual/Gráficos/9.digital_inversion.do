* Set working directory (adjust as needed)
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

clear
set more off

* Creating dataset with digital advertising investment data
input year investment_millions
2014 66
2015 74
2016 79
2017 105
2018 109
2019 132
2020 140
2021 249
2022 262
end

* Calculate year-to-year percentage change
gen pct_change = ((investment_millions - investment_millions[_n-1]) / investment_millions[_n-1]) * 100 if year > 2014

* Calculate trend lines
regress investment_millions year
predict investment_trend, xb

regress pct_change year if year > 2014
predict pct_trend, xb

* Guardar los datos
save "digital_ad_investment_2014_2022.dta", replace

* Crear el gráfico de línea
twoway (connected investment_millions year, lcolor("70 146 207") mcolor("70 146 207") msymbol(circle) msize(medium)) ///
       (scatter investment_millions year, msymbol(none) mlabel(investment_millions) mlabposition(12) mlabgap(1.5) mlabcolor("3 81 143") mlabsize(small) mlabformat(%9.0f)) ///
       (line investment_trend year, lcolor("70 146 207*0.35") lwidth(vthin) lpattern(dash)) ///
       (connected pct_change year if year > 2014, yaxis(2) lcolor("235 148 19") mcolor("235 148 19") msymbol(circle) msize(medium) lpattern(dash)) ///
       (scatter pct_change year if year > 2014, yaxis(2) msymbol(none) mlabel(pct_change) mlabposition(12) mlabgap(1.5) mlabcolor("192 115 0") mlabsize(small) mlabformat(%9.1f)) ///
       (line pct_trend year if year > 2014, yaxis(2) lcolor("235 148 19*0.35") lwidth(vthin) lpattern(dash)), ///
       title("Evolución de la Inversión en Publicidad Digital en Perú" "(2014-2022)", margin(b+5)) ///
       xtitle("Año") ///
       ytitle("Inversión (millones de dólares)", axis(1) margin(r+4)) ///
       ytitle("Cambio porcentual (%)", axis(2) margin(l+4)) ///
       xlabel(2014(1)2022, labsize(small)) ///
       ylabel(0(50)300, axis(1) nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(-10(20)80, axis(2) nogrid angle(horizontal) format(%9.0f)) ///
       legend(label(1 "Inversión (millones USD)") label(3 "Tendencia Inversión") label(4 "Cambio porcentual (%)") label(6 "Tendencia Cambio %") order(1 3 4 6) position(6) ring(1) span cols(2) size(small) bmargin(t+2)) ///
       //note("Fuente: IAB Perú y PwC. Elaboración Propia.") ///
       plotregion(margin(l=10 r=10 t=10 b=10)) ///
       scheme(s1color)

* Exportar el gráfico
graph export "digital_ad_investment_2014_2022_trend.png", replace width(1200)
