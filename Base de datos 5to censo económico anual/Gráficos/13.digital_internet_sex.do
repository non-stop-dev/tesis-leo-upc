* Set working directory (adjust as needed)
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

clear
set more off

* Creating dataset with internet usage by sex
input year internet_hombre internet_mujer
2010 38.9 30.5
2011 39.7 32.3
2012 41.6 34.6
2013 42.4 36.0
2014 43.0 37.3
2015 43.2 38.5
2016 48.3 42.6
2017 51.5 45.9
2018 55.2 49.9
2019 59.7 54.5
2020 66.8 62.3
end

* Label variables
label variable year "Año"
label variable internet_hombre "Internet - Hombre (%)"
label variable internet_mujer "Internet - Mujer (%)"

* Save the dataset
save "internet_usage_sex_peru_2010_2020.dta", replace

* Graph: Grouped bar graph for internet usage by sex with percentage labels
graph bar internet_hombre internet_mujer, over(year, label(labsize(small))) ///
    bar(1, color("160 208 200")) ///
    bar(2, color("236 196 77")) ///
    blabel(bar, format(%9.1f) color("60 60 60") size(vsmall) position(outside)) ///
    title("Uso de Internet por Sexo en Perú (2010-2020)", margin(b+5)) ///
    ytitle("Porcentaje de población (%)", margin(r+4)) ///
    ylabel(0(20)80, nogrid angle(horizontal) format(%9.0f)) ///
    legend(label(1 "Hombre") label(2 "Mujer") position(6) ring(1) span cols(2) size(small) bmargin(t+2)) ///
    //note("Fuente: Instituto Nacional de Estadística e Informática - Encuesta Nacional de Hogares. Elaboración Propia.") ///
    plotregion(margin(l=10 r=10 t=10 b=10)) ///
    scheme(s1color)

* Export the graph
graph export "internet_usage_sex_peru_2010_2020.png", replace width(1200)
