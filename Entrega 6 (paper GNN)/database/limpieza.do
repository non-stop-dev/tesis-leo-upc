cd "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Entrega 6 (paper GNN)/database"

// Cargar base cruda desde la ubicación original
use "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Base de datos 5to censo económico anual/Base curada/1.v_censo_crudo.dta", clear

// ssc install distinct

//----- Filtrar solo empresas privadas -----// Nótese el espacio delante de Particular, la base tiene ese typo
keep if REGISTRO == 11
drop REGISTRO

//------------------------------------------//



//----- Crear variable region (Costa, Sierra, Selva) usando CCDD -----//
// Basado en la clasificación oficial del gobierno peruano
gen region = .
replace region = 0 if inlist(CCDD, "07", "11", "13", "14", "15", "18", "20", "23", "24")  // Costa
replace region = 1 if inlist(CCDD, "02", "03", "04", "05", "06")  // Sierra
replace region = 1 if inlist(CCDD, "08", "09", "10", "12", "19", "21")  // Sierra
replace region = 2 if inlist(CCDD, "01", "16", "17", "22", "25")  // Selva

// Etiquetar la variable
label define region_label 0 "Costa" 1 "Sierra" 2 "Selva"
label values region region_label
label variable region "Región geográfica del Perú"
//------------------------------------------------------------//

//----- Crear UBIGEO (Código Único de Ubicación Geográfica) -----//
// Formato: CCDD (2) + CCPP (2) + CCDI (2) = 6 dígitos
gen ubigeo = CCDD + CCPP + CCDI
label variable ubigeo "Ubigeo (Departamento + Provincia + Distrito)"
//------------------------------------------------------------//



//----- Operaciones durante 2021 -----//
// En la base de datos cruda:
// 1 - Operó
// 2 - No operó
// 3 - Operó parcialmente

// En la base de datos limpia:
// 0-No operó para nada durante 2021
// 1-Operó, al menos, en algún momento del 2021

// Generar 2 categorías, con 3 el logit no corre
gen op2021_original = .
replace op2021_original = 0 if C2P12A_2021 == 2
replace op2021_original = 1 if C2P12A_2021 == 3
replace op2021_original = 1 if C2P12A_2021 == 1

label define op2021_label 0 "Prácticamente no operó durante 2021" 1 "Operó, al menos, parcialmente durante 2021"
label values op2021 op2021_label
label variable op2021 "Operación del establecimiento en 2021"


drop C2P12A_2021

//------------------------------------------------------------//




//----- Operaciones durante 2021 -----//
// Generar la variable op2021_ajustado para corregir inconsistencias en la operatividad
// Explicación: La variable op2021_original presenta dos inconsistencias:
// 1. Marca como no operativas (0) algunas empresas que reportan ventas en 2021, incluso superiores a 100K soles, lo cual es ilógico.
// 2. Marca como operativas (1) algunas empresas con ventas igual a 0 soles, lo cual también es inconsistente.

// Para corregir esto, generamos op2021_ajustado, que:
// - Marca como operativas (1) las empresas con ventas superiores a 5000 soles, aunque estén como no operativas en op2021_original.
// - Marca como no operativas (0) las empresas con ventas igual a 0 soles, aunque estén como operativas en op2021_original.

gen op2021_ajustado = op2021_original
replace op2021_ajustado = 0 if op2021_original == 1 & VENTAS_2021 == 0 & !missing(VENTAS_2021)
replace op2021_ajustado = 1 if op2021_original == 0 & VENTAS_2021 > 5000 & !missing(VENTAS_2021)

label define op2021_ajustado_label 0 "Prácticamente no operó durante 2021" 1 "Operó, al menos, parcialmente durante 2021"
label values op2021_ajustado op2021_ajustado_label
label variable op2021_ajustado "Operación del establecimiento en 2021 (ajustado por ventas)"

//------------------------------------------------------------//




//----- Formalidad -----//
gen ruc = 0 if TENENCIA == 2
replace ruc = 1 if (TENENCIA == 1 | TENENCIA == 3)
label variable ruc "TENENCIA RUC"

label define ruc_label 0 "Informal (sin RUC)" 1 "Formal (con RUC)"
label values ruc ruc_label
label variable ruc "Tenencia de RUC (Formalidad)"

drop TENENCIA

//------------------------------------------//

drop ID_DET ID_REG_ORD
drop C2P2_NS C2P2_NT C2P1_12 C2P1_O C2P7_N C2P8_N ACT_EC_DESC C3P4_COD_PRODUCTO C3P4_LIT_PRODUCTO
drop ZONA_ORD MANZANA_ORD FRENTE_ORD

//----- Crear variable para regimen tributario -----//
// 0-Ninguno
// 1-Nuevo Régimen Único Simplificado (RUS)
// 2-Régimen Especial de Renta (RER)
// 3-Régimen MYPE Tributario (RMT)
// 4-Régimen General (RG)

gen regimen = .
replace regimen = 0 if C3P3 == 5
replace regimen = 1 if C3P3 == 1
replace regimen = 2 if C3P3 == 2
replace regimen = 3 if C3P3 == 3
replace regimen = 4 if C3P3 == 4

label define regimen_label 0 "Ninguno" 1 "Nuevo Régimen Único Simplificado (RUS)" 2 "Régimen Especial de Renta (RER)" 3 "Régimen MYPE Tributario (RMT)" 4 "Régimen General (RG)"
label values regimen regimen_label
label variable regimen "Régimen Tributario"

drop C3P3

//--------------------------------------------------//



//----- Tipo de local -----//
// 0-Local propio
// 1-Local alquilado
// 2-Otro

gen tipo_local = .
replace tipo_local = 0 if C3P8 == 1
replace tipo_local = 1 if C3P8 == 2
replace tipo_local = 2 if C3P8 == 3

label define tipo_local_label 0 "Propio" 1 "Alquilado" 2 "Otro"
label values tipo_local tipo_local_label
label variable tipo_local "Propiedad del local"


drop C3P8

//--------------------------------------------------//





//----- Género del gerente del local -----//
// 0-Mujer
// 1-Hombre

gen sexo_gerente = . 
replace sexo_gerente = 0 if C4_SEXO == 2
replace sexo_gerente = 1 if C4_SEXO == 1

label define sexo_gerente_label 0 "Mujer" 1 "Hombre"
label values sexo_gerente sexo_gerente_label
label variable sexo_gerente "Género del gerente del local"

drop C4_SEXO

//------------------------------------------------------------//

//-------------- Sector ----------------//
// 0-Comercial
// 1-Productivo
// 2-Servicios

gen sector = .
replace sector = 0 if C3P5 == 2
replace sector = 1 if C3P5 == 1
replace sector = 2 if C3P5 == 3

label define sector_label 0 "Comercial" 1 "Productivo" 2 "Servicios"
label values sector sector_label
label variable sector "Sector Económico"

drop C3P5

//------------------------------------------------------------//

//----- Crear variables CIIU a diferentes niveles de agregación -----//
// Clasificación Internacional Industrial Uniforme (CIIU Rev. 4)
// Niveles de desagregación según INEI-SUNAT:
// - 2 dígitos: División (ej: 47 = Comercio al por menor)
// - 3 dígitos: Grupo (ej: 471 = Comercio al por menor en establecimientos no especializados)
// - 4 dígitos: Clase (ej: 4711 = Comercio al por menor en establecimientos no especializados con surtido compuesto principalmente de alimentos, bebidas o tabaco)
// - 6 dígitos: Subclase (ej: 471100 = máxima desagregación sectorial)
//
// Uso econométrico (Cameron & Miller 2015, "A Practitioner's Guide to Cluster-Robust Inference"):
// - Mayor agregación (2 dig): Pocos clusters, errores estándar más conservadores
// - Mayor desagregación (6 dig): Muchos clusters, mayor precisión sectorial
// - Recomendación: Probar múltiples niveles para robustez

// Verificar longitud de códigos CIIU disponibles
quietly summarize C3P4_COD
gen ciiu_length = length(C3P4_COD) if C3P4_COD != ""

// CIIU 2 dígitos: División (agregación sectorial amplia)
gen ciiu_2dig = substr(C3P4_COD, 1, 2) if C3P4_COD != ""
label variable ciiu_2dig "CIIU División (2 dígitos)"

// CIIU 4 dígitos: Clase (desagregación sectorial estándar)
gen ciiu_4dig = substr(C3P4_COD, 1, 4) if C3P4_COD != "" & ciiu_length >= 4
label variable ciiu_4dig "CIIU Clase (4 dígitos)"

// Nota: Se eliminan niveles 3 y 6 dígitos por redundancia y escasez
// - 4 dígitos: Nivel óptimo para definir "competencia" (aristas)
// - 2 dígitos: Nivel óptimo para "contexto sectorial" (feature global)

// Diagnóstico: Verificar distribución de empresas por nivel de agregación
tab ciiu_2dig, missing
tab ciiu_4dig if !missing(ciiu_4dig)

// Resumen de clusters disponibles para análisis de robustez
quietly distinct ciiu_2dig
display "Clusters CIIU 2 dígitos: " r(ndistinct)
quietly distinct ciiu_4dig
display "Clusters CIIU 4 dígitos: " r(ndistinct)

drop C3P4_COD ciiu_length

//------------------------------------------------------------//

//----- Digitalización binaria -----//
// 0-no tiene
// 1-tiene

gen digital = .
replace digital = 0 if C2P6_N == 1 & C2P10_N == 1 & C2P11_N == 1 & !missing(C2P6_N, C2P10_N, C2P11_N)
replace digital = 1 if (C2P6_N == 0 | C2P10_N == 0 | C2P11_N == 0) & !missing(C2P6_N, C2P10_N, C2P11_N)
label define digital_label 0 "Ninguna herramienta digital" 1 "Al menos 1 herramienta digital"
label values digital digital_label
label variable digital "Herramientas digitales"

//------------------------------------------------------------//



//----- Score digitalización (0 a 4) -----//
// Asigna 1 punto por cada herramienta digital: página web, Facebook, otra red social
gen digital_score = 0
replace digital_score = digital_score + 1 if C2P6_N == 0 & !missing(C2P6_N)
replace digital_score = digital_score + 1 if C2P10_N == 0 & !missing(C2P10_N)
replace digital_score = digital_score + 1 if C2P11_N == 0 & !missing(C2P11_N)
replace digital_score = . if missing(C2P6_N, C2P10_N, C2P11_N)
label variable digital_score "Puntaje de digitalización (0 a 3)"
label define digital_score 0 "Sin herramientas digitales" 1 "Al menos 1 herramienta" 2 "Al menos 2 herramientas" 3 "Al menos 3 herramientas"
label values digital_score digital_score

drop C4P4_N
//------------------------------------------------------------//



drop TIPO_PADRE C2P1_PADRE RESFIN RESFIN_O
drop C2P6_N C2P9_N C2P10_N C2P11_N C2P11 C3P1 C3P1_1A C3P2 C3P2_7 C3P2_O C3P5_4 
drop ACT_ECONOMICA C3P5_AE_PRODUCTIVA C3P5_AE_COMERCIAL C3P5_AE_SERVICIOS C5P5_AE_OTRO
drop C3P8_O C4P1_CARGO C4P1_CARGO_O C4P2_N C4P2_A_N C4P3_N



//----- Convertir ventas netas a UIT (UIT 2021 = 4400 PEN) -----//
gen ventas_soles_2021 = VENTAS_2021
gen ventas_uit_2021 = VENTAS_2021 / 4400
label variable ventas_soles_2021 "Ventas netas 2021 en soles"
label variable ventas_uit_2021 "Ventas netas 2021 en UIT"
format ventas_uit_2021 %12.2f

drop VENTAS_2021  
//------------------------------------------------------------//



//----- Clasificar empresas por tamaño según ventas en UIT -----//
// Microempresas: <= 150 UIT
// Pequeñas empresas: > 150 UIT y <= 1700 UIT
// Medianas empresas: > 1700 UIT y <= 2300 UIT
// Grandes empresas:  > 2300 UIT
gen tamano_empresa = .
replace tamano_empresa = 0 if ventas_uit_2021 <= 150 & !missing(ventas_uit_2021)
replace tamano_empresa = 1 if ventas_uit_2021 > 150 & ventas_uit_2021 <= 1700 & !missing(ventas_uit_2021)
replace tamano_empresa = 2 if ventas_uit_2021 > 1700 & ventas_uit_2021 <= 2300 & !missing(ventas_uit_2021)
replace tamano_empresa = 3 if ventas_uit_2021 > 2300 & !missing(ventas_uit_2021)

label define tamano_empresa 0 "Microempresa" 1 "Pequeña empresa" 2 "Mediana empresa" 3 "Gran Empresa"
label values tamano_empresa tamano_empresa
label variable tamano_empresa "Tamaño de la empresa según ventas en UIT"
//------------------------------------------------------------//



//----- Filtrar solo microempresas (1) y pequeñas empresas (2) -----//
keep if inlist(tamano_empresa, 0, 1)
//------------------------------------------------------------//



//----- Capturar y etiquetar variables económicas y de personal -----//

// Capturar tributos
gen tributos = TRIBUTOS
label variable tributos "Tributos pagados por la empresa"
drop TRIBUTOS

// Capturar recuperación de impuestos
gen recup_tributos = RECUPERACION_IMPUESTOS
label variable recup_tributos "Recuperación de impuestos"
drop RECUPERACION_IMPUESTOS

// Capturar utilidad (resultado del ejercicio)
gen result_ejerc = RESULTADO_DEL_EJERCICIO
label variable result_ejerc "Resultado del ejercicio (utilidad)"
drop RESULTADO_DEL_EJERCICIO

// Capturar utilidad (resultado del ejercicio)
gen margen_comerc = MARGEN_COMERCIAL
label variable margen_comerc "Margen comercial"
drop MARGEN_COMERCIAL

// Capturar salarios (remuneraciones)
gen salarios = REMUNERACION
label variable salarios "Remuneraciones pagadas al personal"
drop REMUNERACION

		// Remuneraciones
		//Son todos los pagos brutos, en dinero y/o especie, así como las aportaciones de seguridad y previsión social realizados por el empleador durante el año de estudio (01 de enero al 31 de diciembre), destinados a retribuir el trabajo del personal remunerado.

// Capturar personal total ocupado
gen personal_total = PERSONAL_OCUPADO
label variable personal_total "Total de personal ocupado"
drop PERSONAL_OCUPADO

		// Personal Ocupado
		//Es toda persona que trabaja en un establecimiento, incluyendo asalariados (empleados u obreros) que perciben un ingreso por prestar sus servicios en un proceso productivo. Comprende tanto a hombres como mujeres, ya sean permanentes o contratados por un período determinado. También incluye al personal no remunerado.

// Capturar personal masculino
gen personal_masc = C3P7_TOT_H
label variable personal_masc "Personal masculino ocupado"
drop C3P7_TOT_H

// Capturar personal femenino
gen personal_fem = C3P7_TOT_M
label variable personal_fem "Personal femenino ocupado"
drop C3P7_TOT_M

// Capturar valor agregado
// NOTA: El V Censo 2022 (INEI) proporciona VALOR_AGREGADO ya calculado como:
// VAB = Ventas - Consumo Intermedio (costos de materias primas, insumos, servicios de terceros)
// Esta variable se usa para calcular productividad laboral según definición oficial INEI
gen valor_agreg = VALOR_AGREGADO
label variable valor_agreg "Valor agregado bruto (VAB) de la empresa"
drop VALOR_AGREGADO


// Crear variable de Productividad del trabajo
// DEFINICIÓN OFICIAL INEI (V Censo Nacional Económico 2022):
// "La productividad del trabajo mide el aporte de cada trabajador en la generación
// de valor agregado. Se calcula como el valor agregado anual promedio generado por
// cada trabajador (personal ocupado). Cuanto mayor sea este indicador, más deseable
// es el resultado."
//
// Fórmula: Productividad Laboral = Valor Agregado Bruto (VAB) / Trabajadores Ocupados
// donde VAB = Ventas - Consumo Intermedio
//
// Esta definición es consistente con:
// - Manual OECD "Measuring Productivity" (2001)
// - Cuenta Satélite de la Economía Informal (INEI)
// - Compendio Estadístico Perú 2014 (INEI)

gen productividad_x_trabajador = valor_agreg / personal_total if personal_total > 0
label variable productividad_x_trabajador "Productividad laboral: VAB promedio por trabajador (INEI 2022)"


//------------------------------------------------------------//

drop VENTAS_NETAS_MERCADERIA COSTO_VENTAS_MERCADERIAS _v1
drop PRESTACION_SERVICIOS PRODUCCION_ALMACENADA PRODUCCION_INMOVILIZADA
drop PRODUCCION_DEL_EJERCICIO PRODUCCION CONSUMO_INTERMEDIO RESULTADO_EXPLOTACION
drop COMPRA_MATERIAS_PRIMAS VARIACION_MATERIAS_PRIMAS SERVICIOS_PRESTADOS_TERCEROS
drop EXCEDENTE_EXPLOTACION VARIACION_MERCADERIA FLAG_ID FLAG_FASE
// drop CCDD DEPARTAMENTO (Conservados para visualización espacial)

//----- Guardar base limpia para GNN -----//
save "msme_gnn_preprocessed.dta", replace
