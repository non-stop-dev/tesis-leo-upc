* Set working directory (adjust as needed)
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

clear
set more off

* Creating dataset with mobile telephony and internet data by area
input year mobile_lima mobile_resto_urbano mobile_rural internet_lima internet_resto_urbano internet_rural
2010 83.3 81.3 46.2 25.7 11.4 0.3
2011 85.6 82.7 49.8 32.7 14.2 0.4
2012 88.8 85.7 58.2 38.7 18.3 0.8
2013 88.5 87.9 63.1 41.4 20.0 0.9
2014 91.1 89.4 68.9 44.5 21.1 1.2
2015 92.9 90.8 73.1 42.4 21.5 1.0
2016 93.3 92.5 76.4 48.2 24.2 1.5
2017 94.0 93.5 78.6 52.1 25.7 1.6
2018 95.1 93.6 79.9 54.2 27.1 2.1
2019 96.3 94.5 81.2 58.7 35.7 4.6
2020 97.5 96.4 88.1 58.7 38.9 8.8
end

* Label variables
label variable year "Año"
label variable mobile_lima "Telefonía móvil - Lima Metropolitana (%)"
label variable mobile_resto_urbano "Telefonía móvil - Resto urbano (%)"
label variable mobile_rural "Telefonía móvil - Área rural (%)"
label variable internet_lima "Internet - Lima Metropolitana (%)"
label variable internet_resto_urbano "Internet - Resto urbano (%)"
label variable internet_rural "Internet - Área rural (%)"

* Graph 1: Grouped bar graph for mobile telephony by region with percentage labels
graph bar mobile_lima mobile_resto_urbano mobile_rural, over(year, label(labsize(small))) ///
    bar(1, color("160 208 200")) ///
    bar(2, color("236 196 77")) ///
    bar(3, color("118 152 160")) ///
    blabel(bar, format(%9.0f) color("60 60 60") size(vsmall) position(outside)) ///
    title("Uso de Telefonía Móvil por Región en Perú (2010-2020)", margin(b+5)) ///
    ytitle("Porcentaje de hogares (%)", margin(r+4)) ///
    ylabel(0(20)100, nogrid angle(horizontal) format(%9.0f)) ///
    legend(label(1 "Lima Metropolitana") label(2 "Resto urbano") label(3 "Área rural") position(6) ring(1) span cols(3) size(small) bmargin(t+2)) ///
    //note("Fuente: Instituto Nacional de Estadística e Informática - Encuesta Nacional de Hogares. Elaboración Propia.") ///
    plotregion(margin(l=10 r=10 t=10 b=10)) ///
    scheme(s1color)

* Graph 2: Grouped bar graph for internet usage by region with percentage labels
graph bar internet_lima internet_resto_urbano internet_rural, over(year, label(labsize(small))) ///
    bar(1, color("160 208 200")) ///
    bar(2, color("236 196 77")) ///
    bar(3, color("118 152 160")) ///
    blabel(bar, format(%9.0f) color("60 60 60") size(vsmall) position(outside)) ///
    title("Uso de Internet por Región en Perú (2010-2020)", margin(b+5)) ///
    ytitle("Porcentaje de hogares (%)", margin(r+4)) ///
    ylabel(0(20)100, nogrid angle(horizontal) format(%9.0f)) ///
    legend(label(1 "Lima Metropolitana") label(2 "Resto urbano") label(3 "Área rural") position(6) ring(1) span cols(3) size(small) bmargin(t+2)) ///
    //note("Fuente: Instituto Nacional de Estadística e Informática - Encuesta Nacional de Hogares. Elaboración Propia.") ///
    plotregion(margin(l=10 r=10 t=10 b=10)) ///
    scheme(s1color)

