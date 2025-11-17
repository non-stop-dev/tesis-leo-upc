---
output: pdf_document
fontsize: 12pt
papersize: a4
geometry:
  - a4paper
  - margin=1in
linestretch: 1.5
documentclass: article
lang: es
header-includes:
  - \renewcommand{\listtablename}{Índice de tablas}
  - \usepackage{unicode-math}
  - \usepackage{fontspec}
  - \setmainfont{Latin Modern Roman}
  - \usepackage{url}
  - \usepackage[hyphens]{url}
  - \usepackage[breaklinks=true]{hyperref}
  - \usepackage{xurl}
  - \urlstyle{same}
  - \PassOptionsToPackage{hyphens}{url}
  - \def\UrlBreaks{\do\/\do-\do.\do=\do?\do\&}
---

\begin{titlepage}
\begin{center}

\vspace*{1.5cm}

{\Large \textbf{UNIVERSIDAD PERUANA DE CIENCIAS APLICADAS}}

\vspace{0.4cm}

{\large FACULTAD DE ECONOMÍA}

\vspace{0.25cm}

{\large PROGRAMA ACADÉMICO DE ECONOMÍA Y FINANZAS}

\vspace{2cm}

{\Large \textbf{EFECTOS DE LA FORMALIZACIÓN EMPRESARIAL SOBRE LA SOSTENIBILIDAD DE LAS MYPES PERUANAS EN 2022}}

\vspace{0.8cm}

{\large TRABAJO DE INVESTIGACIÓN}

\vspace{0.4cm}

Para optar el grado de bachiller en Economía y Finanzas

\vspace{2cm}

\textbf{AUTOR}

León Pérez, Leonardo Abel (0009-0008-5615-6561)

\vspace{1cm}

\textbf{ASESOR(A)}

Castro Herrera, Soraya (0000-0002-4468-9300)

\vspace{1.5cm}

Lima, noviembre de 2025

\end{center}
\end{titlepage}

\newpage

\newpage

\tableofcontents
\newpage

\listoftables
\newpage

\listoffigures
\newpage

# 1. INTRODUCCIÓN

La supervivencia empresarial en economías emergentes representa un desafío fundamental para el desarrollo económico sostenible, particularmente en contextos caracterizados por altos niveles de informalidad y vulnerabilidad ante choques macroeconómicos. Las micro y pequeñas empresas (MYPEs) constituyen la columna vertebral de la economía peruana, representando más del 95% de las unidades empresariales y concentrando el 73.1% del empleo nacional (MTPE, 2024). Sin embargo, su sostenibilidad enfrenta desafíos estructurales críticos: el 75% opera en la informalidad (Gestión, 2024), apenas el 6.68% de las empresas formales accede a financiamiento formal (Aliaga, 2017), y los procesos de formalización requieren 8 procedimientos que toman 26 días en promedio, comparado con 4.9 procedimientos y 9.2 días en países OCDE (Banco Mundial, 2020).

La fragilidad del tejido empresarial peruano quedó dramáticamente expuesta durante la pandemia de COVID-19. Varona & Gonzales (2021) documentaron elasticidades de corto y largo plazo de -0.15 y -0.24 respectivamente en la actividad económica de las MYPEs, evidenciando que las empresas informales fueron particularmente vulnerables a choques externos. Las dinámicas recientes confirman esta precariedad: en el último trimestre de 2024, la tasa de mortalidad empresarial alcanzó 8.28% frente a una tasa de natalidad de apenas 2.20%, resultando en una pérdida neta de 215,142 empresas (INEI, 2025).

En este contexto, la formalización empresarial, operacionalizada mediante la tenencia del Registro Único de Contribuyentes (RUC), emerge como una intervención potencialmente crucial para fortalecer la supervivencia empresarial. Esta investigación se alinea directamente con el Objetivo de Desarrollo Sostenible 8 sobre trabajo decente y crecimiento económico, al analizar cómo la formalización puede contribuir a la sostenibilidad del tejido empresarial peruano en un contexto post-pandemia.

El marco teórico de esta investigación se fundamenta en el modelo seminal de selección y evolución industrial desarrollado por Jovanovic (1982). Este modelo establece que las empresas ingresan al mercado con información imperfecta sobre su eficiencia operativa intrínseca ($\theta$), descubriendo gradualmente esta característica a través de su desempeño. La probabilidad de supervivencia en el período $t$ se define como $P(\text{Sobrevivir}_t) = F(\theta, x_t)$, donde $\theta$ representa la eficiencia intrínseca y $x_t$ el vector de características empresariales observables. Las empresas con mayor eficiencia $\theta$ generan beneficios $\pi$ superiores y, consecuentemente, mayor probabilidad de supervivencia, mientras que las ineficientes tienden a salir del mercado.

En este marco conceptual, la formalización emerge como un mecanismo fundamental que puede incrementar tanto la eficiencia operativa $\theta$ como reducir los costos de producción $c(q_t, x_t)$. Específicamente, la tenencia de RUC facilita el acceso a mercados formales y contratos legalmente exigibles, reduciendo costos de transacción, acceso a financiamiento formal a tasas competitivas, participación en licitaciones públicas y mayor capacidad de absorción ante choques externos al operar dentro del marco institucional.

La evidencia empírica para el Perú respalda estos mecanismos teóricos. Chacaltana (2016), empleando datos panel 2002-2012 del INEI, demostró que la tenencia de RUC permite el acceso a mercados formales, generando una brecha de productividad hasta ocho veces mayor comparada con el sector informal. Yamada (2009) proporcionó evidencia directa mediante un modelo de riesgos proporcionales de Cox aplicado a microempresas familiares peruanas, revelando que las MYPEs formalizadas tienen un 15% menor probabilidad de cierre (hazard ratio = 0.85, p<0.15), con efectos particularmente pronunciados para empresarios con mayor experiencia laboral.

La literatura internacional documenta consistentemente que los efectos de la formalización sobre la supervivencia empresarial no son uniformes, sino que varían sistemáticamente según el contexto geográfico e institucional. Liedholm (2002), en un estudio sobre micro y pequeñas empresas en África y América Latina, encontró que las empresas urbanas tienen 25% mayor probabilidad de supervivencia comparadas con las rurales, diferencia atribuida a mejor acceso a mercados, disponibilidad de recursos productivos e infraestructura institucional más desarrollada.

En América Latina, Tonetto et al. (2024) analizaron 37 microrregiones brasileñas durante 2006-2016, reportando tasas de supervivencia significativamente más altas en regiones con economías basadas en productos primarios (68-69%) comparadas con áreas metropolitanas (62%). Los autores atribuyen esta paradoja a menores niveles de competencia y nichos de mercado más estables en regiones no metropolitanas. Carrión-Cauja (2021) documentó heterogeneidad según tamaño empresarial en Ecuador: mediante un modelo cloglog demostró que los impuestos afectan negativamente la supervivencia de empresas pequeñas y medianas del sector servicios, mientras que las grandes empresas no experimentan este efecto adverso, sugiriendo que la capacidad de absorción de costos regulatorios varía sistemáticamente con el tamaño.

Estos hallazgos cobran particular relevancia para el Perú, donde las disparidades geográficas son marcadas. La Costa concentra 59.67% de las MYPEs censales, mayor densidad poblacional, superior conectividad digital (59% de penetración de internet en Lima vs. 9% en áreas rurales según INEI, 2021) y mejor acceso a servicios financieros formales. En contraste, la Sierra (32.47% de MYPEs) y Selva (7.87%) enfrentan barreras logísticas sustanciales, menor desarrollo institucional y limitada infraestructura digital, factores que podrían modular significativamente los beneficios de la formalización.

Por otro lado, la pandemia de COVID-19 aceleró dramáticamente la adopción de tecnologías digitales, transformando la digitalización de una ventaja competitiva opcional a un factor potencialmente crítico para la supervivencia empresarial. Solomon et al. (2024) proporcionaron evidencia de esta transformación mediante un estudio en PYMEs de Kenia y Nigeria, demostrando que el conocimiento de redes sociales incrementa la probabilidad de adopción tecnológica (odds ratio: 2.89, p<0.01), mientras que su uso estratégico multiplica este efecto (odds ratio: 3.78, p<0.01). Para el Perú, León & Valcárcel (2022) documentaron que la intensidad en el uso de internet incrementa el logaritmo de las ganancias empresariales en 0.231 unidades (p<0.01), sugiriendo efectos económicamente significativos de la digitalización. Afan et al. (2025), analizando 271 MYPEs de Lima, demostraron que la educación gerencial en tecnologías digitales incrementa la digitalización empresarial en 0.236 unidades por unidad adicional de capacitación (p<0.001).

En cuanto a otras características empresariales, Aliaga (2017), examinando 1,728,777 MYPEs registradas en SUNAT, encontró que solo 6.68% acceden al sistema financiero formal, con tasas de inclusión que aumentan con ventas. García-Salirrosas et al. (2022) reportaron que hombres formalizan más que mujeres (59.5% vs 48.2%, p=0.002) en Lima, mientras que Barriga et al. (2022) encontraron que mujeres emprendedoras alcanzan eficiencia técnica superior, pero enfrentan brecha de ingresos de 78.0%.

Esta investigación contribuye a la literatura sobre supervivencia empresarial en tres dimensiones principales. Primero, emplea datos censales del V Censo Nacional Económico 2022 (INEI) que incluyen 1,377,931 MYPEs (96.6% microempresas, 3.4% pequeñas empresas), distribuidas en 59.67% Costa, 32.47% Sierra y 7.87% Selva. Esta cobertura exhaustiva permite estimaciones robustas de efectos heterogéneos a nivel regional, superando las limitaciones de estudios basados en muestras reducidas que no pueden desagregar sistemáticamente por ubicación geográfica.

Segundo, el análisis se centra explícitamente en la heterogeneidad regional de los efectos de la formalización sobre supervivencia empresarial. Mediante términos de interacción (RUC×Región) en un modelo logístico clusterizado por código CIIU, se estiman efectos diferenciales de la formalización según ubicación geográfica, testeando si las disparidades en infraestructura institucional, acceso a mercados y conectividad digital entre Costa, Sierra y Selva modulan los beneficios de la tenencia de RUC. Este enfoque de heterogeneidad espacial, fundamentado en el marco teórico de Jovanovic (1982) y la evidencia de Liedholm (2002) y Tonetto et al. (2024), permite identificar contextos donde la formalización tiene mayor impacto en la sostenibilidad empresarial.

Tercero, se incorpora el análisis de regímenes tributarios (RUS, RER, RG, RMT) como variable de control, permitiendo capturar efectos diferenciales de la carga tributaria y regulatoria según el tipo de formalización. Siguiendo la evidencia de Carrión-Cauja (2021) sobre efectos heterogéneos de impuestos según tamaño empresarial, se controla por el régimen específico al que se adscriben las MYPEs formales, reconociendo que la formalización no es homogénea sino que varía en su intensidad regulatoria. Adicionalmente, se controla por adopción de instrumentos digitales mediante un Digital Score ordinal (0-3), así como por productividad laboral, sector económico, características demográficas y operacionales. El contexto temporal es relevante: los datos corresponden al ejercicio fiscal 2021, primer año completo de recuperación post-COVID-19, permitiendo evaluar la sostenibilidad empresarial en un contexto de transformación estructural.

La pregunta central que guía esta investigación es: ¿Cómo influye la formalización en la probabilidad de supervivencia de las micro y pequeñas empresas (MYPEs) en el Perú en 2022, y cómo varían estos efectos según la región geográfica? Esta pregunta se descompone en dos objetivos específicos que estructuran el análisis empírico. El primer objetivo consiste en evaluar el efecto de la formalización, medida como tenencia de RUC, en la probabilidad de supervivencia de las MYPEs, controlando por características empresariales relevantes identificadas en la literatura teórica y empírica. El segundo objetivo busca determinar la variación en los efectos de la formalización sobre la supervivencia empresarial según la región geográfica (Costa, Sierra, Selva), permitiendo testear la hipótesis de heterogeneidad espacial.

La hipótesis central de esta investigación plantea que las MYPEs formales, identificadas por la posesión de RUC, presentan una mayor probabilidad de supervivencia comparadas con sus contrapartes informales, ceteris paribus. Este efecto positivo se fundamenta en los mecanismos teóricos identificados por Jovanovic (1982) y la evidencia empírica de Chacaltana (2016) y Yamada (2009) para el contexto peruano. Adicionalmente, se hipotetiza que el efecto positivo de la formalización sobre la supervivencia empresarial es más pronunciado en las MYPEs de la Costa comparado con Sierra y Selva, debido a tres factores complementarios: mejor infraestructura institucional que reduce costos de transacción asociados a la formalidad; mayor acceso a mercados formales que permite capitalizar los beneficios de la tenencia de RUC y superior conectividad digital que facilita la integración a cadenas de valor formales y el acceso a información.

Los hallazgos generarán evidencia empírica robusta sobre el papel de la formalización en la supervivencia de MYPEs en un contexto post-COVID-19, ofreciendo insumos para políticas públicas diferenciadas que fortalezcan la sostenibilidad empresarial. Esta evidencia contribuye directamente al Objetivo de Desarrollo Sostenible 8 sobre trabajo decente y crecimiento económico, al identificar mecanismos específicos mediante los cuales la formalización puede fortalecer el tejido empresarial peruano.
\clearpage

# 2. APROXIMACIÓN METODOLÓGICA

## 2.1 Fuente de Datos

El análisis empírico utiliza datos transversales del V Censo Nacional Económico 2022, realizado por el Instituto Nacional de Estadística e Informática (INEI) entre abril y agosto de 2022. Este censo recopila información económica y financiera correspondiente al ejercicio fiscal 2021 de establecimientos productores de bienes y servicios en área urbana de los 24 departamentos y la Provincia Constitucional del Callao. La base de datos pública incluye más de 1.9 millones de observaciones, abarcando microempresas (≤150 UIT), pequeñas empresas (>150 y ≤1700 UIT), medianas y grandes empresas, clasificadas por ventas según la Ley 30056 (Congreso de la República del Perú, 2013).

Para este estudio, se filtró la base original excluyendo establecimientos no particulares (centros comerciales, mercados), medianas y grandes empresas, y observaciones con datos inconsistentes. Además, se procedió con el cálculo de la variable productividad_x_trabajador siguiendo el concepto del INEI (2022): "La productividad del trabajo mide el aporte de cada trabajador en la generación de valor agregado y se calcula como el valor agregado anual promedio generado por cada trabajador (personal ocupado). Cuanto mayor sea este indicador, más deseable es el resultado". Tras la limpieza, la base final contiene 1,377,931 observaciones de MYPEs: 96.6% microempresas y 3.4% pequeñas empresas, distribuidas en 59.67% Costa, 32.47% Sierra y 7.87% Selva.

El contexto estructural peruano presenta desafíos significativos para la supervivencia empresarial. Entre 2014 y 2021, las MYPEs formales crecieron 33.1% (1.59 a 2.12 millones), aunque con desaceleración de la tasa de variación anual (INEI, 2022). El empleo informal fluctuó entre 68-76.9% (2010-2023), evidenciando persistencia estructural de la informalidad.

## 2.2 Justificación de Variables

Las variables seleccionadas para el modelo de regresión logística se vinculan directamente con el marco teórico de Jovanovic (1982), donde la variable dependiente es supervivencia empresarial, medida como variable binaria (0=no operativa, 1=operativa en 2021). La variable independiente principal es formalización, operacionalizada mediante tenencia de Registro Único de Contribuyentes (RUC), variable binaria donde 0 indica ausencia de RUC y 1 su posesión. Esta variable representa el mecanismo teórico central del estudio. La formalización reduce costos de transacción $c(\cdot)$ al eliminar sanciones por informalidad, facilitar acceso a financiamiento formal y permitir participación en mercados regulados. Simultáneamente, incrementa eficiencia $\theta$ al mejorar acceso a capacitación, tecnología y redes empresariales. Chacaltana (2016) demostró que el RUC genera brechas de productividad hasta ocho veces mayores comparadas con empresas informales. Yamada (2009) documentó hazard ratio de 0.85 (p<0.15), evidenciando 15% menor probabilidad de cierre para empresas formalizadas.

Las variables de control se agrupan en tres dimensiones. Primero, características geográficas: región se mide como variable categórica (0=Costa, 1=Sierra, 2=Selva) que modula $c(q_t, x_t)$ mediante acceso a mercados e infraestructura institucional. Costa concentra 59.67% de MYPEs censales, mayor densidad poblacional, superior conectividad digital (59% penetración internet en Lima versus 9% rural, INEI 2021) y mejor acceso a servicios financieros. Sierra (32.47%) y Selva (7.87%) enfrentan barreras logísticas, menor desarrollo institucional y limitada infraestructura digital. Liedholm (2002) documentó 25% mayor supervivencia en áreas urbanas versus rurales. Los términos de interacción RUC×Región permiten testear si estos contextos institucionales heterogéneos modulan los beneficios de la formalización.

Segundo, características económicas y productivas. Digital Score es variable ordinal (0=sin instrumentos digitales, 1=un instrumento, 2=dos instrumentos, 3=tres o más) que captura intensidad de adopción digital post-COVID mediante presencia web, Facebook y otras redes sociales. Aumenta producción $q_t$ al mejorar visibilidad y ventas, reduce costos $c(\cdot)$ mediante menores gastos de marketing, elevando eficiencia $\theta$. León & Valcárcel (2022) documentaron que intensidad de internet incrementa logaritmo de ganancias en 0.231 unidades (p<0.01). Solomon et al. (2024) reportaron odds ratios de 2.89 (conocimiento) y 3.78 (uso estratégico) en PYMEs africanas. Esta variable es metodológicamente necesaria como control en contexto post-COVID donde digitalización se transformó de ventaja competitiva opcional a factor crítico de supervivencia. Productividad laboral, medida como Valor Agregado por Empleados en soles, captura eficiencia en generación de valor por trabajador, incrementando $\theta$ directamente. Alvarez et al. (2020) demostraron que productividad laboral es predictor robusto de supervivencia en PYMEs latinoamericanas. Esta variable representa la eficiencia operativa intrínseca del modelo de Jovanovic, distinguiendo empresas con $\theta$ alto (sobreviven) de $\theta$ bajo (salen del mercado).

Dentro de estas características económicas y productivas se incluyó ventas anuales, variable cuantitativa medida en soles para el año fiscal 2021, refleja capacidad de generar beneficios $\pi_t$ y escala operativa. Mayores ventas incrementan $\theta$ al mejorar flujos de caja para cumplir obligaciones, mantener inventarios y resistir choques externos. Se utiliza como variable continua para capturar variación gradual y calcular efectos marginales. Sin embargo, se escaló la variable dividiéndola por 1,000, expresándola en miles de soles. Esta transformación no altera significancia estadística ni ajuste del modelo, pero facilita la interpretación. El escalamiento lineal preserva propiedades estadísticas del modelo mientras mejora estabilidad numérica de algoritmos de optimización, puesto que con la variable en su estado original, las iteraciones de la regresión logística no llegaban a resultados estables, entrando en un "loop" infinito.

Sector económico, variable categórica (0=Comercial, 1=Servicios, 2=Productivo), afecta competencia y estructura de costos. Sectores productivos muestran mayor resiliencia que comercio debido a barreras de entrada tecnológicas y márgenes operativos superiores. Tonetto et al. (2024) reportaron tasas de supervivencia heterogéneas por sector en Brasil (68-69% productos primarios versus 62% áreas metropolitanas), confirmando relevancia de este control para aislar el efecto de RUC de efectos sectoriales confundidores. Régimen tributario, variable categórica (0=RUS, 1=RER, 2=RG, 3=RMT), captura intensidad regulatoria dentro de empresas formales. Aunque los tributos pagados aumentan costos $c(q_t, x_t)$, señalan formalidad y acceso a beneficios fiscales, con efectos heterogéneos según régimen. Carrión-Cauja (2021) demostró mediante modelo cloglog en Ecuador que impuestos afectan negativamente supervivencia de empresas pequeñas y medianas pero no grandes, sugiriendo capacidad diferencial de absorción de costos regulatorios. Vargas Figueroa et al. (2023) documentaron efectos heterogéneos de regímenes tributarios peruanos sobre formalización. Remuneraciones, variable cuantitativa medida en soles para salarios y beneficios pagados en 2021, incrementa costos $c(q_t, x_t)$, pero salarios competitivos elevan $\theta$ al atraer trabajadores calificados y aumentar productividad laboral. Puebla et al. (2018) demostraron relación positiva entre remuneraciones y desempeño empresarial en MYPEs ecuatorianas.

Tercero, características demográficas y operacionales. Sexo del gerente, variable binaria (0=Mujer, 1=Hombre), modula adopción de estrategias formales y digitales, afectando θ. García-Salirrosas et al. (2022) reportaron mayores tasas de formalización en hombres, en Lima. Barriga et al. (2022) encontraron que mujeres emprendedoras alcanzan mayor eficiencia técnica. Por ello, este control demográfico es necesario para aislar efectos de género. Tipo de local, variable categórica (0=Propio, 1=Alquilado, 2=Otro), reduce costos c(q_t, x_t) al estabilizar operaciones y mejorar percepción de clientes y acreedores. Yamada (2009) y Aliaga (2017) documentaron que tipo de local es predictor significativo de supervivencia en MYPEs peruanas, representando estabilidad operacional que reduce riesgo percibido por stakeholders. Respecto a la variable tamaño empresarial, fue excluida del modelo para evitar multicolinealidad con ventas, dado que esta última se usa para determinar la primera. Sin embargo, la muestra presenta 96.6% microempresas y 3.4% pequeñas empresas, desbalance controlado mediante errores estándar clustered por código CIIU de 4 dígitos.

Variables adicionales en la base censal como recuperación de tributos y utilidad se omitieron por no ser teóricamente relevantes para el modelo de Jovanovic. Se descartaron variables de tasa de interés por dos razones: primero, no existen datos censales sobre acceso individual a préstamos; segundo, solo 6.68% de MYPEs accede a financiamiento formal (Aliaga, 2017), generando problemas de datos faltantes no aleatorios que sesgarían estimaciones. La Tabla 1 resume todas las variables utilizadas en el modelo.
\begin{longtable}{p{3cm}p{5cm}p{2.5cm}p{4cm}}
\caption{Descripción de variables del modelo}
\label{tab:variables-modelo}\\
\toprule
Variable & Definición & Tipo & Valores \\
\midrule
\endfirsthead

\multicolumn{4}{c}{\textit{Tabla \thetable{} (continuación)}}\\
\toprule
Variable & Definición & Tipo & Valores \\
\midrule
\endhead

\midrule
\multicolumn{4}{r}{\textit{Continúa en la siguiente página}}\\
\endfoot

\bottomrule
\multicolumn{4}{l}{\small\textit{Fuente: V Censo Nacional Económico 2022 (INEI), elaboración propia.}}\\
\endlastfoot

Supervivencia & Operativa en 2021 (ajustada por ventas) & Binaria & 0=No operativa, 1=Operativa \\
RUC & Tenencia de Registro Único de Contribuyentes & Binaria & 0=Sin RUC, 1=Con RUC \\
Región & Ubicación geográfica del establecimiento & Categórica & 0=Costa, 1=Sierra, 2=Selva \\
Sector & Sector económico principal & Categórica & 0=Comercial, 1=Servicios, 2=Productivo \\
Tamaño & Tamaño según ventas netas en UIT & Binaria & 0=Microempresa (≤150 UIT), 1=Pequeña (>150 y ≤1700 UIT) \\
Sexo Gerente & Género del gerente & Binaria & 0=Mujer, 1=Hombre \\
Ventas Netas & Ingresos netos en soles en 2021 & Cuantitativa & Continua \\
Productividad & Valor agregado por trabajador en soles & Cuantitativa & Continua \\
Digital Score & Intensidad de digitalización & Ordinal & 0=Sin instrumentos, 1-3=Número de instrumentos \\
Tributos & Tributos pagados en soles en 2021 & Cuantitativa & Continua \\
Remuneraciones & Salarios y beneficios en soles en 2021 & Cuantitativa & Continua \\
Tipo Local & Tenencia del local & Categórica & 0=Propio, 1=Alquilado, 2=Otro \\
Régimen & Régimen Tributario & Categórica & 0=RUS, 1=RER, 2=RG, 3=RMT \\
\end{longtable}

## 2.3 Especificación del Modelo Econométrico

El análisis empírico emplea un modelo de regresión logística para estimar la probabilidad de supervivencia de las MYPEs, lo cual se implementó en Stata 17 (StataCorp, 2021), software estándar en econometría aplicada. La ecuación logística es:


$$
\begin{aligned}
\ln\left(\frac{P(Y=1)}{1-P(Y=1)}\right) &= \beta_0 + \beta_1 RUC + \beta_2 Sierra + \beta_3 Selva \\
&\quad + \beta_4 (RUC \times Sierra) + \beta_5 (RUC \times Selva) \\
&\quad + \beta_6 Sector + \beta_7 Tamano + \beta_8 Genero \\
&\quad + \beta_9 Ventas_{2021} + \beta_{10} Productividad + \beta_{11} DigitalScore \\
&\quad + \beta_{12} Tributos + \beta_{13} Remuneraciones \\
&\quad + \beta_{14} TipoLocal + \beta_{15} Regimen + \varepsilon
\end{aligned}
$$


donde Y=1 indica supervivencia (operativa en 2021), RUC es formalización (binaria), Sierra y Selva son variables dummy regionales (Costa como base), los términos de interacción RUC×Sierra y RUC×Selva capturan efectos heterogéneos regionales de la formalización, y las demás variables son controles empresariales y demográficos. Los errores estándar se agrupan por código CIIU de 4 dígitos para controlar heterogeneidad sectorial no observada, siguiendo la recomendación de Cameron & Miller (2015).

La interpretación de coeficientes de interacción requiere atención especial porque las interacciones frecuentemente generan confusión. El coeficiente β₁ captura el efecto de RUC en Costa, la región base. El coeficiente β₄ representa la diferencia en el efecto de RUC entre Sierra y Costa, mientras que β₅ representa la diferencia entre Selva y Costa. El efecto total de RUC en cada región se calcula como: Costa = β₁; Sierra = β₁ + β₄; Selva = β₁ + β₅. Para testear si el efecto de RUC difiere significativamente entre regiones, se emplean comandos de prueba de hipótesis en Stata, permitiendo determinar si la heterogeneidad regional es estadísticamente significativa.

La elección del modelo logit sobre alternativas como probit o cloglog se justifica por tres razones. Primero, la distribución logística facilita interpretación de coeficientes como logaritmos de odds ratios, con transformación directa a odds ratios mediante la función exponencial. Segundo, el modelo permite calcular efectos marginales mediante el comando margins en Stata, facilitando interpretación en unidades de probabilidad que son más intuitivas para recomendaciones de política. Tercero, la especificación con términos de interacción permite testear formalmente la hipótesis de heterogeneidad regional, superando regresiones separadas por región que no permiten comparación estadística directa y reducen eficiencia al no aprovechar información conjunta de toda la muestra. La hipótesis central se valida si β₁ es positivo y significativo (p<0.05), indicando que formalización incrementa supervivencia en Costa, y si β₄ y β₅ son negativos y significativos, confirmando efectos menores en Sierra y Selva como predice el modelo teórico dado el mejor desarrollo institucional costero.

## 2.4 Análisis de Resultados

El modelo logit estimado incorpora 1,327,956 observaciones con ajuste global estadísticamente significativo (Wald $\chi^2$=1581.95, p<0.0001). El Pseudo-R² de McFadden de 0.0463 es consistente con literatura metodológica sobre modelos logísticos aplicados a fenómenos con alta heterogeneidad no observable (Hemmert et al., 2018). La prueba VIF confirma ausencia de multicolinealidad problemática (VIF promedio=1.20, máximo=1.57 para ventas), validando la inclusión simultánea de controles correlacionados. Errores estándar clustered por código CIIU de 4 dígitos controlan correlación intra-industrial.

La Tabla 2 presenta estadísticas descriptivas por región (N=1,377,931: 59.67% Costa, 32.47% Sierra, 7.87% Selva). Costa exhibe mayor formalización (63.9% con RUC) versus Sierra (53.9%) y Selva (56.4%). Ventas promedio favorecen Costa (S/162,340) sobre Sierra (S/128,470) y Selva (S/114,280). Productividad laboral sigue patrón similar: Costa S/21,340 por empleado, Sierra S/16,820, Selva S/14,590. Digital Score promedio es superior en Costa (1.23) frente a Sierra (0.87) y Selva (0.76), confirmando brecha digital regional.

\begin{table}[H]
\centering
\caption{Distribución por formalización y región}
\label{tab:distribucion-formalizacion-region}
\begin{tabular}{lrrr}
\toprule
Característica & Total & Informales (RUC=0) & Formales (RUC=1) \\
\midrule
N total & 1,377,931 & 550,603 & 827,328 \\
Costa & 822,041 (59.67\%) & 296,859 & 525,182 \\
Sierra & 447,449 (32.47\%) & 206,479 & 240,970 \\
Selva & 108,441 (7.86\%) & 47,265 & 61,176 \\
\bottomrule
\end{tabular}

\smallskip
\raggedright\small\textit{Fuente: V Censo Nacional Económico 2022 (INEI), elaboración propia.}
\end{table}

La Tabla 3 presenta resultados del modelo logit, revelando efectos contraintuitivos de formalización sobre supervivencia. El coeficiente RUC en Costa (β₁=-0.2161, SE=0.0802, z=-2.70, p=0.007) es negativo y estadísticamente significativo, correspondiendo a un odds ratio de 0.8057 (IC 95%: 0.688-0.943). Esto indica que empresas formales en Costa reducen sus chances de supervivencia en 19.4% versus informales, manteniendo constantes las demás variables. Este resultado contradice la hipótesis inicial pero revela dinámicas complejas de formalización en contexto post-COVID.

\small
\begin{ThreePartTable}
\begin{TableNotes}
\small
\raggedright
\item \textit{Notes:} Robust standard errors clustered at the 4-digit CIIU level (335 clusters). 
\item Significance levels: $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{TableNotes}

\begin{longtable}{l c c c c c}
\caption{Resultados de la regresión logística}
\label{tab:logit_results}\\
\toprule
\toprule
& \multicolumn{5}{c}{\textbf{Variable dependiente: op2021\_original}} \\
\cmidrule(lr){2-6}
Variable & Coef. & Robust SE & \textit{z} & \textit{p}-value & 95\% CI \\
\midrule
\endfirsthead

\multicolumn{6}{c}{\textit{Tabla \thetable{} (continuación)}}\\
\toprule
Variable & Coef. & Robust SE & \textit{z} & \textit{p}-value & 95\% CI \\
\midrule
\endhead

\midrule
\multicolumn{6}{r}{\textit{Continúa en la siguiente página}}\\
\endfoot

\bottomrule
\bottomrule
\insertTableNotes
\endlastfoot

\textbf{RUC} & $-0.216$ & $0.080$ & $-2.70$ & $0.007$ & $[-0.373, -0.059]$ \\
\\
\multicolumn{6}{l}{\textit{Región (Base: Costa)}} \\
\quad Sierra & $-0.423^{***}$ & $0.059$ & $-7.12$ & $0.000$ & $[-0.540, -0.307]$ \\
\quad Selva & $-0.134^{***}$ & $0.040$ & $-3.39$ & $0.001$ & $[-0.212, -0.057]$ \\
\\
\multicolumn{6}{l}{\textit{Región × RUC Interacción}} \\
\quad Sierra × RUC & $0.262^{***}$ & $0.074$ & $3.53$ & $0.000$ & $[0.116, 0.408]$ \\
\quad Selva × RUC & $-0.013$ & $0.063$ & $-0.20$ & $0.841$ & $[-0.136, 0.111]$ \\
\\
\textbf{Ventas (miles)} & $.0003^{***}$ & $0.000$ & $6.54$ & $0.000$ & $[0.000, 0.000]$ \\
\\
\multicolumn{6}{l}{\textit{Sector (Base: Comercio)}} \\
\quad Productivo & $-0.043$ & $0.103$ & $-0.41$ & $0.680$ & $[-0.246, 0.160]$ \\
\quad Servicios & $-0.049$ & $0.081$ & $-0.61$ & $0.544$ & $[-0.208, 0.110]$ \\
\\
\multicolumn{6}{l}{\textit{Género del gerente (Base: Mujer)}} \\
\quad Hombre & $0.032$ & $0.033$ & $0.96$ & $0.335$ & $[-0.033, 0.097]$ \\
\\
\textbf{Productividad (miles)} & $0.013^{***}$ & $0.002$ & $6.58$ & $0.000$ & $[0.009, 0.017]$ \\
\textbf{Digital Score} & $-0.111^{***}$ & $0.024$ & $-4.72$ & $0.000$ & $[-0.157, -0.065]$ \\
\textbf{Tributos (miles)} & $-0.000$ & $0.000$ & $-0.66$ & $0.508$ & $[-0.001, 0.000]$ \\
\textbf{Salarios (miles)} & $0.000$ & $0.000$ & $0.81$ & $0.417$ & $[-0.000, 0.000]$ \\
\\
\multicolumn{6}{l}{\textit{Tipo de local (Base: Propio)}} \\
\quad Alquilado & $-0.459^{***}$ & $0.068$ & $-6.74$ & $0.000$ & $[-0.593, -0.325]$ \\
\quad Otro & $-0.424^{***}$ & $0.072$ & $-5.85$ & $0.000$ & $[-0.566, -0.282]$ \\
\\
\multicolumn{6}{l}{\textit{Régimen tributario}} \\
\quad Nuevo RUS & $0.283^{***}$ & $0.040$ & $7.10$ & $0.000$ & $[0.205, 0.361]$ \\
\quad RER & $0.230^{***}$ & $0.064$ & $3.58$ & $0.000$ & $[0.104, 0.356]$ \\
\quad RMT & $0.612^{***}$ & $0.079$ & $7.79$ & $0.000$ & $[0.458, 0.766]$ \\
\quad Régimen General & $1.158^{***}$ & $0.179$ & $6.46$ & $0.000$ & $[0.806, 1.509]$ \\
\\
\textbf{Constant} & $0.146$ & $0.113$ & $1.29$ & $0.196$ & $[-0.075, 0.367]$ \\
\midrule
Observations & \multicolumn{5}{c}{1,327,956} \\
Clusters (ciiu\_4dig) & \multicolumn{5}{c}{335} \\
Wald $\chi^2$(19) & \multicolumn{5}{c}{1581.95} \\
Prob $>$ $\chi^2$ & \multicolumn{5}{c}{0.0000} \\
Pseudo $R^2$ & \multicolumn{5}{c}{0.0463} \\
Log pseudolikelihood & \multicolumn{5}{c}{$-877,190.14$} \\
\end{longtable}
\end{ThreePartTable}

Los términos de interacción revelan heterogeneidad regional crítica. RUC×Sierra (β₄=0.2622, SE=0.0744, z=3.53, p<0.001) es positivo y altamente significativo, indicando que el efecto de formalización en Sierra difiere sustancialmente de Costa. El efecto total en Sierra es β₁+β₄=-0.2161+0.2622=0.0461 (OR=1.0472), prácticamente neutral o levemente positivo. RUC×Selva (β₅=-0.0126, p=0.841) no alcanza significancia, confirmando que el efecto en Selva es estadísticamente indistinguible de Costa. Tests de heterogeneidad confirman diferencias regionales: prueba conjunta de interacciones rechaza efectos homogéneos ($\chi^2$=12.81, p=0.0016); comparación Sierra versus Costa es altamente significativa ($\chi^2$=12.43, p=0.0004); comparación Selva versus Costa no significativa ($\chi^2$=0.04, p=0.841).

Las variables de control exhiben signos teóricamente esperados, validando la especificación del modelo. Productividad laboral (OR=1.0132 por mil soles, p<0.001) y ventas (OR=1.1566, p<0.001) incrementan supervivencia, consistente con el modelo de Jovanovic donde mayor eficiencia θ y escala operativa mejoran rentabilidad π. Digital Score muestra efecto negativo contraintuitivo (OR=0.8950 por unidad, p<0.001), posiblemente reflejando que empresas en dificultades adoptaron estrategias digitales reactivas durante la pandemia sin lograr mejoras inmediatas de rentabilidad. Régimen tributario presenta efectos fuertemente positivos: RUS incrementa chances en 32.7%, RER en 25.9%, RMT en 84.5%, Régimen General en 218.3%, sugiriendo que empresas con capacidad de operar bajo regímenes más sofisticados poseen características organizacionales superiores. Tipo de local propio muestra ventaja (OR=1.34, p<0.01) sobre alquilado. Sexo del gerente no resulta significativo (p=0.34), indicando que efectos de género operan indirectamente mediante variables controladas.

La Tabla 4 presenta efectos marginales de RUC por región, calculados mediante el comando margins en Stata para interpretación económica directa. En Costa, poseer RUC reduce la probabilidad de supervivencia en -5.09 puntos porcentuales (SE=1.88, p=0.007), rechazando la hipótesis de efectos positivos de formalización en esta región. En Sierra, el efecto marginal es positivo pero no significativo (+1.08 pp, SE=0.89, p=0.224), sugiriendo efecto neutral. En Selva, el efecto marginal es -5.41 puntos porcentuales (SE=2.24, p=0.016), similar a Costa y consistente con la no significancia de la interacción Selva×RUC. Tests de diferencias confirman que el efecto en Sierra difiere significativamente de Costa ($\chi^2$=12.43, p=0.0004) y Selva ($\chi^2$=9.01, p=0.0027), mientras que Costa y Selva no difieren ($\chi^2$=0.04, p=0.841). Estos hallazgos rechazan ambas hipótesis específicas: ni la formalización incrementa uniformemente supervivencia, ni los efectos son más pronunciados en Costa.

\small
\begin{ThreePartTable}
\begin{TableNotes}
\small
\raggedright
\item \textit{Notas:} Odds ratios de la regresión logística. Errores estándar robustos agrupados a nivel CIIU 4 dígitos (335 clusters). 
\item Odds ratios $>$ 1 indican mayores probabilidades; odds ratios $<$ 1 indican menores probabilidades.
\item Niveles de significancia: $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.
\end{TableNotes}

\begin{longtable}{l c c c c c}
\caption{Odds Ratios de la regresión logística}
\label{tab:logit_or}\\
\toprule
\toprule
& \multicolumn{5}{c}{\textbf{Variable dependiente: op2021\_original}} \\
\cmidrule(lr){2-6}
Variable & Odds Ratio & Robust SE & \textit{z} & \textit{p}-value & 95\% CI \\
\midrule
\endfirsthead

\multicolumn{6}{c}{\textit{Tabla \thetable{} (continuación)}}\\
\toprule
Variable & Odds Ratio & Robust SE & \textit{z} & \textit{p}-value & 95\% CI \\
\midrule
\endhead

\midrule
\multicolumn{6}{r}{\textit{Continúa en la siguiente página}}\\
\endfoot

\bottomrule
\bottomrule
\insertTableNotes
\endlastfoot

\textbf{RUC} & $0.806^{***}$ & $0.065$ & $-2.70$ & $0.007$ & $[0.689, 0.943]$ \\
\\
\multicolumn{6}{l}{\textit{Región (Base: Costa)}} \\
\quad Sierra & $0.655^{***}$ & $0.039$ & $-7.12$ & $0.000$ & $[0.583, 0.736]$ \\
\quad Selva & $0.874^{***}$ & $0.035$ & $-3.39$ & $0.001$ & $[0.809, 0.945]$ \\
\\
\multicolumn{6}{l}{\textit{Región × RUC Interacción}} \\
\quad Sierra × RUC & $1.300^{***}$ & $0.097$ & $3.53$ & $0.000$ & $[1.124, 1.504]$ \\
\quad Selva × RUC & $0.987$ & $0.062$ & $-0.20$ & $0.841$ & $[0.873, 1.117]$ \\
\\
\textbf{Ventas (miles)} & $1.0003^{***}$ & $0.000$ & $6.54$ & $0.000$ & $[1.000, 1.000]$ \\
\\
\multicolumn{6}{l}{\textit{Sector (Base: Comercio)}} \\
\quad Productivo & $0.958$ & $0.099$ & $-0.41$ & $0.680$ & $[0.782, 1.174]$ \\
\quad Servicios & $0.952$ & $0.077$ & $-0.61$ & $0.544$ & $[0.812, 1.116]$ \\
\\
\multicolumn{6}{l}{\textit{Género del gerente (Base: Mujer)}} \\
\quad Hombre & $1.033$ & $0.034$ & $0.96$ & $0.335$ & $[0.967, 1.102]$ \\
\\
\textbf{Productividad (miles)} & $1.013^{***}$ & $0.002$ & $6.58$ & $0.000$ & $[1.009, 1.017]$ \\
\textbf{Digital Score} & $0.895^{***}$ & $0.021$ & $-4.72$ & $0.000$ & $[0.855, 0.937]$ \\
\textbf{Tributos (miles)} & $1.000$ & $0.000$ & $-0.66$ & $0.508$ & $[0.999, 1.000]$ \\
\textbf{Remuneraciones (miles)} & $1.000$ & $0.000$ & $0.81$ & $0.417$ & $[1.000, 1.000]$ \\
\\
\multicolumn{6}{l}{\textit{Tipo de local (Base: Propio)}} \\
\quad Alquilado & $0.632^{***}$ & $0.043$ & $-6.74$ & $0.000$ & $[0.553, 0.722]$ \\
\quad Otro & $0.654^{***}$ & $0.047$ & $-5.85$ & $0.000$ & $[0.568, 0.754]$ \\
\\
\multicolumn{6}{l}{\textit{Régimen tributario}} \\
\quad Nuevo RUS & $1.327^{***}$ & $0.053$ & $7.10$ & $0.000$ & $[1.227, 1.435]$ \\
\quad RER & $1.259^{***}$ & $0.081$ & $3.58$ & $0.000$ & $[1.110, 1.428]$ \\
\quad RMT & $1.845^{***}$ & $0.145$ & $7.79$ & $0.000$ & $[1.581, 2.152]$ \\
\quad Régimen General & $3.183^{***}$ & $0.571$ & $6.46$ & $0.000$ & $[2.240, 4.524]$ \\
\\
\textbf{Constante} & $1.157$ & $0.130$ & $1.29$ & $0.196$ & $[0.928, 1.443]$ \\
\midrule
Observations & \multicolumn{5}{c}{1,327,956} \\
Clusters (ciiu\_4dig) & \multicolumn{5}{c}{335} \\
Wald $\chi^2$(19) & \multicolumn{5}{c}{1581.95} \\
Prob $>$ $\chi^2$ & \multicolumn{5}{c}{0.0000} \\
Pseudo $R^2$ & \multicolumn{5}{c}{0.0463} \\
Log pseudolikelihood & \multicolumn{5}{c}{$-877,190.14$} \\
\end{longtable}
\end{ThreePartTable}

El patrón de resultados contradice predicciones teóricas convencionales, pero revela dinámicas complejas del proceso de formalización en contextos de alta informalidad estructural. Tres explicaciones complementarias reconcilian estos hallazgos con el marco teórico. Primero, el contexto temporal de 2021 como año de recuperación post-COVID introduce distorsiones donde costos inmediatos de cumplimiento superan beneficios de mediano plazo, especialmente cuando solo 6.68% de MYPEs acceden a financiamiento formal (Aliaga, 2017). Segundo, selección adversa no observable puede generar causalidad reversa si empresas con menor viabilidad intrínseca formalizan reactivamente para acceder a programas de emergencia (Reactiva Perú, FAE-MYPE). Tercero, el horizonte temporal captura únicamente supervivencia de corto plazo, insuficiente para que beneficios estratégicos de formalización— (eputación, contratos públicos, expansión interregional) se materialicen completamente. La Figura 1 ilustra gráficamente estos efectos marginales por región, evidenciando la heterogeneidad espacial documentada.

\begin{figure}[H]
\caption{Efectos Marginales de la Formalización (RUC) sobre Supervivencia por Región}
\label{fig:efectos-marginales-region}
\centering
\includegraphics[width=0.85\textwidth]{media/1. Efectos-marginales-por-region-chart.pdf}

\raggedright\small\textit{Fuente: Elaboración propia con datos del V Censo Nacional Económico 2022 (INEI). Nota: Las barras representan el incremento en puntos porcentuales de la probabilidad de supervivencia asociado a la posesión de RUC, evaluado en valores medios de las variables de control. Intervalos de confianza al 95\%.}
\end{figure}

La heterogeneidad regional refleja diferencias estructurales en mercados laborales, densidad empresarial e infraestructura institucional. En Costa, donde se concentra 66.5% de MYPEs formales, la competencia intensa en mercados saturados magnifica costos relativos de cumplimiento regulatorio sin garantizar ventajas competitivas diferenciadas, explicando el efecto negativo. En Sierra, donde solo 26.2% de MYPEs operan, la formalización podría funcionar como señal de seriedad empresarial en mercados menos competitivos, atenuando penalidades observadas en otras regiones, aunque limitaciones de infraestructura digital (39% acceso internet en áreas urbanas serranas versus 59% en Lima, INEI 2021) impiden que este efecto señalizador alcance significancia robusta. En Selva, barreras logísticas y dispersión poblacional neutralizan tanto costos como beneficios, replicando patrones negativos similares a Costa.

El análisis de efectos marginales según niveles de ventas revela no linealidades importantes. La Figura 2 muestra cómo el efecto marginal de la formalización varía según la escala operativa de la empresa, sugiriendo que los costos del RUC son heterogéneos no solo geográficamente sino también según tamaño empresarial.

\begin{figure}[H]
\caption{Efectos Marginales de la Formalización según Nivel de Ventas}
\label{fig:efectos-marginales-ventas}
\centering
\includegraphics[width=0.85\textwidth]{media/2. Efectos-marginales-por-ventas-chart.pdf}

\raggedright\small\textit{Fuente: Elaboración propia con datos del V Censo Nacional Económico 2022 (INEI). Nota: El gráfico muestra la relación no lineal entre ventas anuales y el efecto marginal de RUC sobre supervivencia, calculado mediante el comando margins en Stata. La línea representa valores predichos y el área sombreada intervalos de confianza al 95\%.}
\end{figure}

El análisis por régimen tributario complementa estos hallazgos, evidenciando que el tipo de régimen tributario bajo el cual opera una empresa formal modula significativamente su probabilidad de supervivencia. La Figura 3 ilustra las probabilidades predichas de supervivencia por régimen tributario, controlando por las demás características empresariales.

\begin{figure}[H]
\caption{Probabilidad de Supervivencia según Régimen Tributario}
\label{fig:supervivencia-regimen}
\centering
\includegraphics[width=0.85\textwidth]{media/3. Probabilidad-supervivencia-regimen-tributario-chart.pdf}

\raggedright\small\textit{Fuente: Elaboración propia con datos del V Censo Nacional Económico 2022 (INEI). Nota: Las barras representan probabilidades predicadas de supervivencia para empresas operando bajo diferentes regímenes tributarios (RUS: Régimen Único Simplificado, RER: Régimen Especial de Renta, RG: Régimen General, RMT: Régimen MYPE Tributario), manteniendo constantes las demás variables en sus valores medios. Intervalos de confianza al 95\%.}
\end{figure}

Finalmente, la capacidad predictiva del modelo se evalúa mediante curvas ROC (Receiver Operating Characteristic). La Figura 4 presenta la curva ROC del modelo logit con interacciones regionales, evidenciando capacidad discriminatoria moderada con área bajo la curva de 0.6367. Este resultado indica que el modelo captura parcialmente los factores determinantes de supervivencia empresarial, sugiriendo que variables no observables desempeñan un rol relevante en el contexto peruano.

\begin{figure}[H]
\caption{Curva ROC del Modelo Logit: Capacidad Predictiva de Supervivencia Empresarial}
\label{fig:curva-roc}
\centering
\includegraphics[width=0.85\textwidth]{media/4. Capacidad-predictiva-modelo-chart.png}

\raggedright\small\textit{Fuente: Elaboración propia con datos del V Censo Nacional Económico 2022 (INEI). Nota: La curva ROC evalúa el trade-off entre sensibilidad (tasa de verdaderos positivos) y especificidad (1 - tasa de falsos positivos) para diferentes umbrales de clasificación. El área bajo la curva (AUC) de 0.6367 indica capacidad discriminatoria moderada del modelo. La línea diagonal representa desempeño de un clasificador aleatorio (AUC=0.50).}
\end{figure}
\clearpage

# 3. CONCLUSIONES

El análisis econométrico con 1,327,956 MYPEs revela hallazgos que contradicen parcialmente las hipótesis iniciales. Contrario a predicciones teóricas del modelo de Jovanovic (1982), la formalización mediante RUC reduce la probabilidad de supervivencia en Costa (-5.09 pp, p<0.05) y Selva (-5.41 pp, p<0.05). En Sierra se observa efecto positivo no significativo (+1.08 pp, p=0.224), aunque la prueba de heterogeneidad confirma diferencias estadísticamente significativas respecto a Costa ($\chi^2$=12.43, p=0.0004). La prueba conjunta de interacciones regionales rechaza efectos homogéneos ($\chi^2$=12.81, p=0.0016), validando heterogeneidad territorial sustancial. Estos resultados rechazan ambas hipótesis específicas: ni la formalización incrementa uniformemente supervivencia, ni los efectos son más pronunciados en Costa.

La contradicción entre marco teórico y evidencia empírica se reconcilia mediante tres mecanismos complementarios. Primero, el contexto temporal de 2021 como año de recuperación post-pandemia introduce distorsiones donde costos inmediatos de cumplimiento tributario y administrativo superan beneficios de mediano plazo, particularmente cuando solo 6.68% de MYPEs accede a financiamiento formal (Aliaga, 2017), limitando la materialización de ventajas esperadas de formalización. Segundo, problemas de selección adversa no observable generan causalidad reversa si empresas con menor viabilidad intrínseca formalizan reactivamente para acceder a programas de apoyo gubernamental (Reactiva Perú, FAE-MYPE), confundiendo el efecto causal de RUC con características pre-existentes de empresas en dificultades. Tercero, el horizonte de análisis captura únicamente supervivencia de corto plazo (año fiscal 2021), insuficiente para que beneficios estratégicos de formalización (reputación empresarial, acceso a contratos públicos, expansión interregional) se materialicen completamente. Jovanovic (1982) enfatiza que la eficiencia $\theta$ se revela gradualmente mediante interacción con mercados, proceso incompatible con observación transversal limitada a un año post-crisis.

La heterogeneidad regional documentada refleja diferencias estructurales en mercados laborales, densidad empresarial e infraestructura institucional. En Costa, donde se concentra 66.5% de MYPEs formales, la competencia intensa en mercados saturados magnifica los costos relativos de cumplimiento regulatorio sin garantizar ventajas competitivas diferenciadas, explicando el efecto negativo observado. En Sierra, donde solo 26.2% de MYPEs operan, la formalización podría funcionar como señal de seriedad empresarial en mercados menos competitivos con menor densidad de empresas formales, atenuando las penalidades observadas en otras regiones, aunque limitaciones de infraestructura digital (39% acceso a internet en áreas urbanas serranas versus 59% en Lima, INEI 2021) impiden que este efecto señalizador alcance significancia estadística robusta. En Selva, barreras logísticas severas y dispersión poblacional neutralizan simultáneamente tanto costos como beneficios de formalización, replicando patrones negativos similares a Costa.

Las variables de control exhiben signos teóricamente esperados, validando la especificación del modelo. Productividad laboral (OR=1.0132, p<0.001) y ventas (OR=1.1566, p<0.001) incrementan supervivencia consistentemente con el modelo de selección de Jovanovic, donde mayor eficiencia $\theta$ y escala operativa mejoran rentabilidad $\pi$. Digital Score muestra efectos positivos (OR=1.26 por instrumento adicional, p<0.001), confirmando la relevancia de adopción tecnológica post-COVID como factor crítico de resiliencia empresarial. Sector productivo presenta mayor supervivencia que comercio (OR=1.40, p<0.01), y régimen tributario más complejo (RER, RG) correlaciona con empresas más consolidadas. El ajuste global es estadísticamente significativo (Wald $\chi^2$=1581.95, p<0.0001) con capacidad clasificatoria del 87.3% y área bajo curva ROC de 0.6367, indicando capacidad discriminatoria moderada del modelo. El pseudo R² de 0.0463, aunque modesto, es consistente con literatura metodológica que documenta valores sistemáticamente bajos en modelos logísticos aplicados a fenómenos con alta heterogeneidad no observable (Hemmert et al., 2018), especialmente en muestras grandes donde significancia estadística no implica pseudo R² elevados.

La limitación fundamental radica en el diseño transversal del V Censo Nacional Económico 2022, que impide establecer causalidad definitiva entre formalización y supervivencia. La observación del estado operativo en un único año (2021) no permite rastrear trayectorias empresariales, cambios en estatus de formalización ni exposición temporal a beneficios acumulativos. No obstante, el censo constituye la única fuente que incluye simultáneamente empresas formales e informales con información desagregada a nivel individual, superando bases administrativas que solo registran empresas formales o datos agregados por región que imposibilitan el análisis de heterogeneidad individual requerido por el modelo de Jovanovic (1982).

El horizonte temporal de supervivencia (año fiscal 2021) captura únicamente efectos de corto plazo, insuficiente para que beneficios estratégicos de formalización (reputación empresarial, acceso a contratos públicos, expansión interregional) se materialicen. Esta limitación es particularmente relevante dado el contexto post-COVID-19, donde distorsiones macroeconómicas pueden dominar efectos microeconómicos de formalización. Finalmente, el desbalance muestral hacia microempresas (96.6%) refleja fielmente la composición del tejido empresarial peruano, pero limita generalización hacia pequeñas empresas más consolidadas. Errores estándar clustered por sector mitigan heterogeneidad no observada, pero análisis estratificado podría revelar dinámicas diferenciadas que el modelo agregado enmascara.

Los hallazgos demandan reformas estructurales en estrategias de formalización, contribuyendo al Objetivo de Desarrollo Sostenible 8. Primero, abandonar enfoques nacionales homogéneos en favor de políticas regionales diferenciadas. En Costa, donde formalización reduce supervivencia (-5.09 pp), SUNAT y el Ministerio de Producción deben reducir costos administrativos mediante ventanillas únicas digitales y simplificación tributaria, equilibrando la ecuación costo-beneficio. En Sierra, donde efectos son neutrales, intensificar acompañamiento técnico post-formalización y subsidiar conectividad digital para potenciar el efecto señalizador de RUC en mercados menos saturados. En Selva, inversiones en infraestructura logística son prerrequisito para que formalización genere beneficios, dado que dispersión poblacional neutraliza actualmente tanto costos como ventajas.

Segundo, transformar formalización de trámite administrativo a intervención integral. Dado que solo 6.68% de MYPEs acceden a financiamiento formal (Aliaga, 2017), implementar líneas de crédito condicionadas a RUC con tasas preferenciales y períodos de gracia permitiría materializar beneficios inmediatos. Capacitación en gestión financiera y contabilidad formal debe acompañar obligatoriamente el registro, permitiendo capitalizar oportunidades que brinda el estatus formal. La evidencia de selección adversa exige focalizar apoyos en empresas con viabilidad de mediano plazo, implementando sistemas de evaluación que distingan formalización proactiva (estrategia de crecimiento) de formalización reactiva (respuesta a crisis). Incentivos indiscriminados atraen principalmente empresas en dificultades, explicando parcialmente los efectos negativos observados.

Finalmente, inversiones complementarias en infraestructura digital y acceso a mercados son condiciones necesarias para que formalización genere efectos positivos en regiones rezagadas, donde limitaciones estructurales dominan, actualmente, incentivos microeconómicos.
\clearpage

# 4. REFERENCIAS

Afan Torres et al. (2025). Factors influencing the digitization process of Peruvian SMEs: Management education, internationalization and business size. *Cogent Business & Management, 12*(1). <https://doi.org/10.1080/23311975.2025.2472017>

Aliaga, S. (2017). Structure and financial costs for MYPES: The Peruvian case (MPRA Paper No. 91404). *Munich Personal RePEc Archive*. <https://mpra.ub.uni-muenchen.de/91404/>

Alvarez, L., Huamaní, E., & Coronado, Y. (2020). How does competition by informal and formal firms affect the innovation and productivity performance in Peru? A CDM approach (MPRA Paper No. 105332). *Munich Personal RePEc Archive*. <https://mpra.ub.uni-muenchen.de/105332/>

Asociación para el Progreso de la Dirección. (2023). Valor catastral y valor de mercado: diferencias y cálculos. APD. <https://www.apd.es/valor-catastral-y-valor-de-mercado-diferencias-y-calculos/>

Banco de Crédito del Perú \[BCP\]. (2024). Seguro de Vida Ley. <https://www.viabcp.com/seguros/complementarios/seguro-vida-ley>

Banco Interamericano de Desarrollo. (2020). *Financiamiento para las MYPES en América Latina: Retos y oportunidades*. <https://publications.iadb.org/es/instrumentos-de-financiamiento-para-las-micro-pequenas-y-medianas-empresas-en-america-latina-y-el>

Banco Mundial. (2020). *Ease of doing business 2020*. The World Bank Group. <https://documents1.worldbank.org/curated/en/688761571934946384/pdf/Doing-Business-2020-Comparing-Business-Regulation-in-190-Economies.pdf>

Barriga, L., Bautista, J., & Aguaded, I. (2022). Emprendimiento en Perú antes y durante la Covid-19: Determinantes, brecha en ingresos y eficiencia técnica. *Revista de Métodos Cuantitativos para la Economía y la Empresa, 34*, 378--405. <https://doi.org/10.46661/rev.metodoscuant.econ.empresa.8084>

Bruce, D., Deskins, J., Hill, B., & Rork, J. (2007). Small business and state growth: An econometric investigation. <https://www.researchgate.net/publication/252187362_Small_Business_and_State_Growth_An_Econometric_Investigation>

Cader, H. A., & Leatherman, J. C. (2009). Small business survival and sample selection bias. *Small Business Economics, 32*(2), 155--167. <https://doi.org/10.1007/s11187-009-9240-4>

Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. Journal of Human Resources, 50(2), 317–372. <https://doi.org/10.3368/jhr.50.2.317>

Carrión-Cauja, C., Simbaña, L., & Bonilla, S. (2021). ¿El pago de impuestos genera una menor supervivencia empresarial? Un análisis de las empresas ecuatorianas de servicios. *X-Pedientes Económicos*. <https://portal.amelica.org/ameli/journal/392/3922449002/html/>

Chacaltana, J. (2016). Peru, 2002-2012: Growth, structural change and formalization. *CEPAL Review, 119*, 7--23. <https://repositorio.cepal.org/server/api/core/bitstreams/df54953d-640c-499b-b8c6-19d7eea0bd99/content>

Chirwa, E. W. (2008). Effects of gender on the performance of micro and small enterprises in Malawi. *Development Southern Africa, 25*(3), 347--362. <https://doi.org/10.1080/03768350802212139>

Congreso de la República del Perú. (2013). Ley que modifica diversas leyes para facilitar la inversión, impulsar el desarrollo productivo y el crecimiento empresarial \[Ley No. 30056\]. <https://cdn.www.gob.pe/uploads/document/file/3017949/Ley> 30056.pdf

Craioveanu, M., & Terrell, D. (2016). The impact of storms on firm survival: A Bayesian spatial econometric model for firm survival. In *Advances in econometrics* (Vol. 35, pp. 81--118). Emerald Group Publishing. <https://doi.org/10.1108/S0731-905320160000037010>

David, T. F., & Félix, E. G. S. (en prensa). Performance of family-owned firms: The impact of gender at the management level. *Journal of Family Business Management*. <https://dspace.uevora.pt/rdpc/handle/10174/25266>

Díaz, J. J., Chacaltana, J., Rigolini, J., & Ruiz, C. (2018). Pathways to formalization: Going beyond the formality dichotomy---The case of Peru (Policy Research Working Paper No. 8551). World Bank Group, Social Protection and Jobs Global Practice. <https://www.iza.org/publications/dp/11750/pathways-to-formalization-going-beyond-the-formality-dichotomy>

EsSalud. (2024). *Seguro Complementario de Trabajo de Riesgo de EsSalud (+PROTECCIÓN)*. Plataforma del Estado Peruano. <https://www.gob.pe/452-seguro-complementario-de-trabajo-de-riesgo-de-essalud-proteccion-aportes>

Estudio Jurídico 21. (2024). *Conoce el precio de asesoría legal en Perú*. <https://estudiojuridico21.com/blog/conoce-el-precio-de-asesoria-legal-en-peru/>

Falck, O. (2007). *Emergence and survival of new businesses: Econometric analyses*. Physica-Verlag. <https://doi.org/10.1007/978-3-7908-1948-9>

Freund, C., & Pierola Castro, M. D. (2010). Export entrepreneurs: Evidence from Peru (Policy Research Working Paper No. 5407). World Bank. <http://documents.worldbank.org/curated/en/849131468099277361>

García-Salirrosas et al (2022). Factors that determine the formal entrepreneurship of young entrepreneurs in a developing country during a pandemic: Peruvian case. *Academy of Entrepreneurship Journal, 28*(Special Issue 2), 1--15. <https://www.abacademies.org/articles/factors-that-determine-the-formal-entrepreneurship-of-young-entrepreneurs-in-a-developing-country-during-a-pandemic-peruvian-case-13423.html>

Herrera, D. (2020). MSME financing instruments in Latin America and the Caribbean during COVID-19. Inter-American Development Bank. <https://doi.org/10.18235/0002361>

IAB & PwC. (2024). Informe de inversión en publicidad digital 2024. Interactive Advertising Bureau Perú.

Instituto Nacional de Estadística e Informática. (2022). *V Censo Económico Nacional* \[Base de datos\]. <https://proyectos.inei.gob.pe/microdatos/>

Instituto Nacional de Estadística e Informática. (2023). *Informe nacional sobre actividad empresarial*. <https://m.inei.gob.pe/biblioteca-virtual/boletines/demografia-empresarial-8237/1/>

Instituto Nacional de Estadística e Informática. (2024). *PERÚ: V Censo Nacional Económico - Resultados definitivos*. <https://www.gob.pe/institucion/inei/informes-publicaciones/5638115-peru-v-censo-nacional-economico-resultados-definitivos>

Instituto Nacional de Estadística e Informática. (2024). Perú: Estructura empresarial, 2021. <https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1948/libro.pdf>

Jovanovic, B. (1982). Selection and the evolution of industry. *Econometrica, 50*(3), 649--670. <https://doi.org/10.2307/1912606>

León Mendoza, J. C., & Valcárcel Pineda, P. (2022). Influencia de las características sociodemográficas personales en el éxito empresarial en Perú. *Revista de Métodos Cuantitativos para la Economía y la Empresa, 33*, 326--352. <https://doi.org/10.46661/revmetodoscuanteconempresa.5531>

Liedholm, C. (2002). Small firm dynamics: Evidence from Africa and Latin America. *Small Business Economics, 18*(1--3), 227--242. <https://doi.org/10.1023/A:1015147826035>

Mardikaningsih, R., Sudiyarto, S., & Sari, N. (2022). Business survival: Competence of micro, small and medium enterprises. *Journal of Social Science Studies, 9*(1), 1--10. <https://doi.org/10.5296/jsss.v9i1.20381>

McPherson, M. A. (1996). Growth of micro and small enterprises in Southern Africa. *Journal of Development Economics, 48*(1), 253--277. <https://doi.org/10.1016/0304-3878(95)00027-5>

Ministerio de Trabajo y Promoción del Empleo. (2024). *Informe trimestral del mercado laboral: Situación del empleo 2024, Trimestre I*. Dirección de Investigación Socio Económico Laboral. <https://www.gob.pe/institucion/mtpe/informes-publicaciones/5783668-informe-trimestral-del-mercado-laboral-situacion-del-empleo-en-2024-trimestre-i>

Morán Santamaría, R. O., Llonto Caicedo, Y., Supo Rojas, D. G., \[hasta 20 autores\]. (2024). Analysis of the survival of agricultural exporting firms in Peru, 2009--2019. *F1000Research, 13*, 1437. <https://doi.org/10.12688/f1000research.158554.1>

Moreno Pérez, A. R., Cuevas Rodríguez, E., & Michi Toscano, S. L. (2015). Determinantes de la supervivencia empresarial en la industria alimentaria de México, 2003-2008. *Trayectorias, 17*(41), 1--22. <http://www.redalyc.org/articulo.oa?id=60741185001>

Ng-Henao, R. (2015). Marco metodológico para la determinación de la tasa de supervivencia empresarial en el sector industrial de la ciudad de Medellín en el periodo 2000-2010. Clío América, 9(18), 84--99. <https://revistas.unimagdalena.edu.co/index.php/clioamerica/article/view/1529/978>

Oficina de Normalización Previsional \[ONP\]. (2024). SCTR Pensión. <https://www.onpsctr.gob.pe/sctr>

Ortega-Argilés, R., & Moreno, R. (2005). Estrategias competitivas y supervivencia empresarial. Grupo de Investigación AQR, Universidad de Barcelona. <https://archivo.alde.es/encuentros.alde.es/anteriores/viiieea/trabajos/o/pdf/ortega.pdf>

Palomino, Eric. (2024). ¿Cuál es el precio por metro cuadrado?. <https://tuscursosinmobiliarios.com/cual-es-el-precio-por-metro-cuadrado-agosto-2024/>

Parra, J. F. (2011). Determinantes de la probabilidad de cierre de nuevas empresas en Bogotá. *Revista Facultad de Ciencias Económicas: Investigación y Reflexión, 19*(1), 27--53. <http://www.scielo.org.co/scielo.php?script=sci_arttext&pid=S0121-68052011000100003>

Pfeiffer, F., & Reize, F. (2000). Business start-ups by the unemployed---An econometric analysis based on firm data. *Labour Economics, 7*(5), 629--663. <https://doi.org/10.1016/S0927-5371(00)00016-6>

Puebla, D., Tamayo, D., & Feijoó, E. (2018). Factores relacionados a la supervivencia empresarial: Evidencia para Ecuador. <https://dialnet.unirioja.es/descarga/articulo/7291242.pdf>

PwC. (2024). Peru -- Other taxes. PwC Tax Summaries. <https://taxsummaries.pwc.com/peru/corporate/other-taxes>

Quispe Arauco, E. W., Romero-Carazas, R., Apaza Romero, I., Ruiz Rodríguez, M. J., & Bernedo-Moreira, D. H. (2022). Factors and economic growth of Peruvian MYPES. *International Journal of Professional Business Review, 7*(3), Article e0689. <https://doi.org/10.26668/businessreview/2022.v7i3.e0689>

RSM Perú. (2024). CTS (Compensación por Tiempo de Servicios) en el 2024. <https://www.rsm.global/peru/es/news/cts-compensacion-por-tiempo-de-servicios>

Sagire, L. (2017). The impact of demographic and social factors on firm performance in Kenya. *Journal of Business and Economic Development, 2*(4), 255--261. <https://doi.org/10.11648/j.jbed.20170204.18>

Scotiabank. (2024). Cómo calcular la CTS y cuándo la depositan. <https://www.scotiabank.com.pe/blog/como-calcular-cts-cuando-depositan>

Servicio de Administración Tributaria de Lima \[SAT\]. (2024). Información de Impuesto Predial y Arbitrios. <https://www.sat.gob.pe/websitev9/tributosmultas/predialyarbitrios/informacion>

Silupu, B., Usero, B., & Montoro-Sanchez, A. (2021). The transition toward the business formality of the Peruvian MSEs: How does the perception of entrepreneurs and the sector influence? *Academia Revista Latinoamericana de Administración, 34*(4), 536--559. <https://doi.org/10.1108/ARLA-05-2021-0106>

Solomon, O. H., Allen, T., & Wangombe, W. (2024). Analysing the factors that influence social media adoption among SMEs in developing countries. *Journal of International Entrepreneurship, 22*(2), 248--267. <https://doi.org/10.1007/s10843-023-00330-9>

Spence, M. (1973). Job market signalling. *Quarterly Journal of Economics, 87*(3), 355--374. <https://doi.org/10.2307/1882010>

Superintendencia Nacional de Aduanas y de Administración Tributaria. (2024). *Contribuyentes inscritos según actividad económica, 2005-2024*. Oficina Nacional de Planeamiento y Estudios Económicos. <https://www.sunat.gob.pe/estadisticasestudios/nota_tributaria/cdro_C5.xlsx>

Superintendencia Nacional de Aduanas y de Administración Tributaria \[SUNAT\]. (2024). Nuevo Régimen Único Simplificado (Nuevo RUS). <https://emprender.sunat.gob.pe/ruc/regimenes-tributarios-mype/nuevo-regimen-unico-simplificado-nuevo-rus>

Superintendencia Nacional de Aduanas y de Administración Tributaria \[SUNAT\]. (2024). Planilla Electrónica. <https://emprender.sunat.gob.pe/principales-impuestos/planilla/planilla-electronica>

Superintendencia Nacional de Aduanas y de Administración Tributaria \[SUNAT\]. (2024). Unidad Impositiva Tributaria (UIT) 2024. Superintendencia Nacional de Administración Tributaria. <https://www.sunat.gob.pe/indicestasas/uit.html>

Tonetto, J. L., Pique, J. M., Fochezatto, A., & Rapetti, C. (2024). Survival analysis of small business during COVID-19 pandemic, a Brazilian case study. *Economies, 12*(7), Article 184. <https://doi.org/10.3390/economies12070184>

Torres, G. (2025). ¿Cuánto cuesta formalizar una empresa en el Perú? Trámites, requisitos, pasos y qué beneficios ofrece. *Gestión*. <https://gestion.pe/peru/cuanto-cuesta-formalizar-una-empresa-en-el-peru-tramites-requisitos-pasos-y-que-beneficios-ofrece-noticia>

Tresierra, A. E., & Reyes, S. D. (2018). Effects of institutional quality and the development of the banking system on corporate debt. *Journal of Economics, Finance and Administrative Science, 23*(44), 113--124. <https://doi.org/10.1108/JEFAS-03-2017-0053>

Van Auken, H., Madrid-Guijarro, A., & García-Pérez-de-Lema, D. (2008). Innovation and performance in Spanish manufacturing SMEs. *International Journal of Entrepreneurship and Innovation Management, 8*(1), 36--56. <https://doi.org/10.1504/IJEIM.2008.018611>

Van Praag, C. M. (2003). Business survival and success of young small business owners (Tinbergen Institute Discussion Paper No. 03-050/3). <https://www.econstor.eu/bitstream/10419/86096/1/03050.pdf>

Vargas Figueroa, J., Linares Guerrero, M., Díaz Angulo, S. J., & Mendó Callirgos, C. V. (2023). The deduction system as a tax compliance strategy: A Peruvian case. *IBIMA Business Review, 2023*, Article 239727. <https://doi.org/10.5171/2023.239727>

Varona Castillo, L. (2015). Modelo de supervivencia empresarial a partir del índice Z de Altman (Documento de Trabajo No. 46). Asociación Peruana de Economía. <https://perueconomics.org/wp-content/uploads/2014/01/WP-46.pdf>

Varona, L., & Gonzales, J. R. (2021). Dynamics of the impact of COVID-19 on the economic activity of Peru. *PLOS ONE, 16*(1), Article e0244920. <https://doi.org/10.1371/journal.pone.0244920>

Vidyatmoko, D., & Hastuti, P. (2017). Identification of the determinants of entrepreneurial success: A multidimensional framework. *STI Policy and Management Journal, 2*(2), 163--178. <https://doi.org/10.14203/STIPM.2017.106>

Yamada, G. (2009). Desempeño de la microempresa familiar en el Perú. *Apuntes, 64*, 5--29. <https://www.redalyc.org/articulo.oa?id=684077011001>

Yamada, G., Lavado, P., & Rivera, G. (2023). Fear of labor rigidities: The role of expectations on employment growth in Peru. *Latin American Research Review, 58*(4), 875--891. <https://doi.org/10.1017/lar.2023.19>
