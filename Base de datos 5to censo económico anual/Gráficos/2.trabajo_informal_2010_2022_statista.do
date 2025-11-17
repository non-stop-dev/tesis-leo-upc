// Establecer la ruta de trabajo
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

// Limpiar cualquier dato previo
clear



// Ingresar los datos manualmente
input year informal_employment
2010 76.9
2011 75.5
2012 70.7
2013 69.8
2014 68.6
2015 69.2
2016 68.0
2017 68.1
2018 68.5
2019 68.4
2020 70.1
2021 68.4
2022 74.4
2023 71.6
end

// Calcular la línea de tendencia para informal_employment
regress informal_employment year
predict informal_trend, xb

// Crear el gráfico con el estilo definido
twoway (connected informal_employment year, lcolor("70 146 207") mcolor("70 146 207") msymbol(circle) msize(small) lpattern(solid)) ///
       (scatter informal_employment year, msymbol(none) mlabel(informal_employment) mlabposition(12) mlabgap(1) mlabcolor("3 81 143") mlabsize(small) mlabformat(%9.1f)) ///
       (line informal_trend year, lcolor("120 180 230") lwidth(vthin) lpattern(dash)), ///
       title("Porcentaje de Población con Empleos Informales en Perú, 2010-2023", margin(b+5 r+15)) ///
       xtitle("Año") ///
       ytitle("Porcentaje (%)", margin(r+4)) ///
       xlabel(2009(2)2024, nogrid angle(horizontal) format(%9.0f)) ///
       ylabel(65(3)80, nogrid angle(horizontal) format(%9.0f)) ///
       legend(label(1 "Empleos Informales (%)") label(3 "Tendencia") order(1 3) position(6) ring(1) span cols(2) size(small) bmargin(t+2)) ///
       // note("Fuente: Elaboración propia con datos de Statista") ///
       plotregion(margin(l=10 r=10 t=10 b=10)) ///
       scheme(s1color)

// Exportar el gráfico
// graph export "empleos_informales_2010_2023.png", replace width(1200)
