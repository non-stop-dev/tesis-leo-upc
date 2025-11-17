* Set working directory
cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Gráficos"

clear
set more off
clear
set more off

* Creating dataset with department-level data
input str20 departamento mipyme_2021 participacion
"Amazonas"      14523   0.7
"Ancash"        56753   2.7
"Apurímac"      21156   1.0
"Arequipa"      125741  5.9
"Ayacucho"      28661   1.4
"Cajamarca"     51384   2.4
"Callao"        67725   3.2
"Cusco"         80979   3.8
"Huancavelica"  9828    0.5
"Huánuco"       32525   1.5
"Ica"           56338   2.7
"Junín"         75324   3.6
"La Libertad"   111205  5.2
"Lambayeque"    70694   3.3
"Lima"          952923  45.0
"Loreto"        38306   1.8
"Madre de Dios" 16729   0.8
"Moquegua"      14136   0.7
"Pasco"         13092   0.6
"Piura"         87201   4.1
"Puno"          58505   2.8
"San Martín"    50022   2.4
"Tacna"         33109   1.6
"Tumbes"        15585   0.7
"Ucayali"       35849   1.7
"Total"         2118293 100.0
end

* Dropping the total row for regional aggregation
drop if departamento == "Total"

* Assigning regions
gen region = ""
replace region = "Costa" if inlist(departamento, "Callao", "Ica", "La Libertad", "Lambayeque", "Lima", "Moquegua", "Piura", "Tacna", "Tumbes")
replace region = "Sierra" if inlist(departamento, "Ancash", "Apurímac", "Arequipa", "Ayacucho")
replace region = "Sierra" if inlist(departamento, "Cajamarca", "Cusco", "Huancavelica", "Huánuco", "Junín", "Pasco", "Puno")
replace region = "Selva" if inlist(departamento, "Amazonas", "Loreto", "Madre de Dios", "San Martín", "Ucayali")

* Aggregating by region
collapse (sum) mipyme_2021 (sum) participacion, by(region)


graph pie mipyme_2021, over(region) ///
    plabel(_all sum, format(%12.0fc) color("67 69 68") size(small) gap(-5)) ///
    plabel(_all percent, format(%9.2f) color(black) size(small) gap(25)) ///
    pie(1, color("138 161 169")) ///
    pie(2, color("233 203 112")) ///
    pie(3, color("179 212 206")) ///
    title("Distribución de MYPEs por Región en miles (2021)", margin(b+5 t+5 r+5 l+5)) ///
    legend(label(1 "Costa") label(2 "Selva") label(3 "Sierra") position(3) ring(1) span cols(1) size(small) bmargin(t+2)) ///
    //note("Fuente: Datos proporcionados. Elaboración Propia.") ///
    plotregion(margin(l=10 r=10 t=10 b=10)) ///
    scheme(s1color)


* Exportar el gráfico
graph export "mipyme_region_2021.png", replace width(1200)
