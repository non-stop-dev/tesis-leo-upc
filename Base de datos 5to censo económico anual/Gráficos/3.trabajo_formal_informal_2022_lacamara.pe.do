// Establecer la ruta de trabajo
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

// Limpiar cualquier dato previo
clear

// Ingresar los datos
input str10 size_category informal formal
"1-20"    79.4 13.7
"21-50"   0.5  1.5
"51-500"  0.2  1.4
">500"    0    3.3
end

// Combinar los gráficos con una leyenda compartida debajo
graph bar informal formal, over(size_category) ///
    asyvars /// Evita que Stata use "mean of"
    bargap(0) /// Barras pegaditas
    bar(1, color("235 148 19")) bar(2, color("70 146 207")) /// Colores suaves
    blabel(bar, position(outside) color("black") size(small) format(%9.1f)) /// Etiquetas fuera de las barras
    title("Distribución de la fuerza laboral formal/informal por" "empresas clasificadas según cantidad de trabajadores (2022)", margin(b+5)) ///
    ytitle("% del total de la PEA Ocupada", margin(r+4)) ///
    ylabel(0(20)80, nogrid angle(horizontal) format(%9.0f)) /// Escala ajustada
    legend(label(1 "Trabajadores informales") label(2 "Trabajadores formales") order(1 2) position(6) ring(1) span cols(2) size(small) bmargin(t+2)) ///
    // note("Fuente: ENAHO - INEI. IEDEP. Elaboración Propia") ///
    scheme(s1color)

// Exportar el gráfico combinado
graph export "trabajo_formal_informal_2022_lacamara.pe.png", replace width(1200)
