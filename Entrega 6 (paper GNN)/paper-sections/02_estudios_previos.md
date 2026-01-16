# Estudios Previos y Marco Teórico

## 1. Análisis Previo: Modelo Logístico con Heterogeneidad Regional

El presente estudio extiende el análisis de supervivencia empresarial de las MYPES peruanas desarrollado en León (2025), el cual empleó un modelo de regresión logística para estimar la probabilidad de supervivencia operativa durante el año fiscal 2021, utilizando datos del V Censo Nacional Económico (INEI, 2022). El modelo original adoptó la siguiente especificación:

$$
\text{logit}(P[\text{Supervivencia}=1]) = \beta_0 + \beta_1 \text{RUC} + \beta_2 \text{RUC} \times \text{Región} + \mathbf{X}'\boldsymbol{\gamma} + \epsilon
$$

donde $\text{RUC}$ representa la formalización tributaria, las interacciones $\text{RUC} \times \text{Región}$ capturan heterogeneidad geográfica (Costa, Sierra, Selva), y $\mathbf{X}$ incluye controles como productividad laboral, ventas netas, régimen tributario, sector económico y características del establecimiento. Los errores estándar fueron robustos a heterocedasticidad mediante clustering por código CIIU a 4 dígitos (Cameron & Miller, 2015).

Los hallazgos revelaron un efecto contra-intuitivo: la formalización (tenencia de RUC) reduce la probabilidad de supervivencia en la Costa (-5.09 puntos porcentuales) y Selva (-5.41 pp), mientras que el efecto es neutral o ligeramente positivo en la Sierra (+1.08 pp). Estos resultados contradicen las predicciones teóricas del modelo de selección de Jovanovic (1982), así como la evidencia empírica previa que documentaba beneficios sistemáticos de la formalización: Chacaltana (2016) reporta brechas de productividad de hasta ocho veces a favor de las empresas formales, mientras que Yamada (2009) encuentra una probabilidad de cierre 15% menor para firmas con RUC en el contexto peruano.

## 2. Limitaciones del Enfoque Tradicional

A pesar de la robustez econométrica del modelo logístico, el análisis previo presenta limitaciones metodológicas que restringen la profundidad de las conclusiones:

### 2.1 Aproximación Gruesa de Efectos de Vecindario

El uso de variables *dummy* regionales (Costa/Sierra/Selva) captura únicamente efectos promedio a nivel macro-regional, ignorando la heterogeneidad *intra-regional*. Como señala Gibbons y Overman (2012), los efectos de aglomeración y competencia operan típicamente a nivel de *micro-geografías* —distritos, zonas económicas locales— que las clasificaciones administrativas gruesas no pueden distinguir.

### 2.2 Supuesto de Independencia entre Observaciones

Los modelos de regresión tradicionales asumen independencia condicional entre unidades observadas. Sin embargo, las empresas en una misma localidad geográfica o sector económico comparten entornos institucionales, mercados laborales, cadenas de suministro e infraestructura (Glaeser et al., 2010). Esta interdependencia estructural viola el supuesto de observaciones i.i.d. y puede sesgar las inferencias cuando existen efectos de derrame (*spillover effects*).

### 2.3 Incapacidad para Modelar Efectos de Red

La economía de redes (Jackson, 2008) ha demostrado que la posición de un agente dentro de una estructura relacional afecta sus resultados económicos. En el contexto empresarial, la supervivencia de una MYPE puede depender no solo de sus características intrínsecas, sino también de las características de sus competidores sectoriales y vecinos geográficos. El modelo logístico tradicional no puede capturar esta dependencia topológica.

## 3. Redes Neuronales de Grafos: Fundamentos Teóricos

### 3.1 Representación Empresarial como Grafo

Una solución a las limitaciones anteriores consiste en representar explícitamente las relaciones entre empresas mediante una estructura de grafo $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, donde:

- **Nodos** ($\mathcal{V}$): Cada empresa $i$ constituye un nodo con vector de características $\mathbf{x}_i \in \mathbb{R}^d$ (ventas, productividad, régimen tributario, etc.).
- **Aristas** ($\mathcal{E}$): Las conexiones entre nodos representan relaciones económicas relevantes:
  - *Aristas geográficas*: Empresas ubicadas en el mismo distrito.
  - *Aristas sectoriales*: Empresas con el mismo código CIIU a 4 dígitos (competencia directa).

Esta representación permite modelar formalmente la estructura de interacciones que el enfoque logístico tradicional solo aproxima mediante efectos fijos regionales.

### 3.2 Paso de Mensajes (*Message Passing*)

Las Redes Neuronales de Grafos (GNNs) operan mediante un mecanismo denominado *message passing* (Gilmer et al., 2017), que actualiza iterativamente la representación de cada nodo agregando información de sus vecinos:

$$
\mathbf{h}_i^{(k)} = \text{UPDATE}\left( \mathbf{h}_i^{(k-1)}, \text{AGGREGATE}\left( \{ \mathbf{h}_j^{(k-1)} : j \in \mathcal{N}(i) \} \right) \right)
$$

donde $\mathbf{h}_i^{(k)}$ es la representación del nodo $i$ en la capa $k$, $\mathcal{N}(i)$ denota los vecinos del nodo $i$, y las funciones UPDATE y AGGREGATE son parametrizadas y aprendidas durante el entrenamiento.

### 3.3 Convolución en Grafos

Kipf y Welling (2017) propusieron la *Graph Convolutional Network* (GCN), que implementa el paso de mensajes mediante una regla de propagación con normalización simétrica:

$$
\mathbf{H}^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)
$$

donde $\tilde{A} = A + I$ es la matriz de adyacencia con auto-conexiones, $\tilde{D}$ es la matriz diagonal de grados, $\mathbf{W}^{(l)}$ son parámetros aprendibles, y $\sigma$ es una no-linealidad. Esta normalización evita que nodos con muchos vecinos dominen la agregación, una consideración crítica cuando la distribución de empresas está desbalanceada (microempresas representan >90% del censo).

### 3.4 GraphSAGE: Aprendizaje Inductivo

Hamilton et al. (2017) propusieron GraphSAGE (*Sample and Aggregate*), que extiende las GCNs permitiendo:

1. **Muestreo de vecinos**: En lugar de agregar todos los vecinos, se muestrea un subconjunto fijo, habilitando el escalamiento a grafos grandes.
2. **Agregadores flexibles**: Funciones de agregación personalizables (media, suma, LSTM, pooling).
3. **Aprendizaje inductivo**: Capacidad de generar embeddings para nodos no vistos durante el entrenamiento.

Para el presente análisis con 1.3 millones de nodos, GraphSAGE resulta particularmente apropiado por su capacidad de escalar mediante *neighbor sampling* (Hamilton et al., 2017).

## 4. Potencial Analítico de las GNNs para Supervivencia Empresarial

Las GNNs permiten abordar las preguntas que el modelo logístico tradicional no puede responder:

| Pregunta de Investigación | Limitación del Logit | Solución GNN |
|---------------------------|---------------------|--------------|
| ¿El efecto negativo de la formalización varía según la densidad competitiva local? | Solo captura efectos promedio regionales | Modela interacciones empresa-empresa a nivel de distrito/CIIU |
| ¿Existen clusters de empresas con dinámicas de supervivencia similares que no corresponden a divisiones administrativas? | Usa clasificación exógena (Costa/Sierra/Selva) | Descubre clusters endógenos mediante *embeddings* latentes |
| ¿Qué tipo de relación (geográfica vs. sectorial) es más determinante para la supervivencia? | No distingue canales de influencia | Análisis de sensibilidad por tipo de arista |

## 5. Limitaciones del Enfoque GNN Puro

A pesar de su poder expresivo, los modelos GNN puros presentan limitaciones para el análisis económico riguroso:

### 5.1 Opacidad Interpretativa

Mientras que los coeficientes logísticos tienen interpretación directa en términos de *log-odds* y efectos marginales, los pesos de una GNN no mapean directamente a relaciones causales interpretables (Molnar, 2020). Esto dificulta la comunicación de hallazgos a tomadores de decisiones de política pública.

### 5.2 Naturaleza Correlacional

Las GNNs son modelos predictivos que capturan asociaciones, no causalidad. La identificación causal requiere estrategias adicionales —variables instrumentales, discontinuidades, diferencias en diferencias— que las GNNs por sí solas no proveen (Angrist & Pischke, 2009).

### 5.3 Sensibilidad a la Definición del Grafo

El rendimiento y las interpretaciones de una GNN dependen críticamente de cómo se definen las aristas. Si las conexiones no reflejan relaciones económicas genuinas, los embeddings capturarán ruido en lugar de señal estructural (Ying et al., 2018).

## 6. Enfoque Híbrido: Embeddings GNN como Covariables

Para combinar las fortalezas de ambos paradigmas, proponemos un enfoque híbrido en dos etapas:

### Etapa 1: Extracción de Embeddings

Entrenar una GNN para la tarea de clasificación de supervivencia y extraer las representaciones latentes $\mathbf{z}_i$ de la penúltima capa. Estos embeddings codifican tanto las características propias de la empresa como la información estructural de su vecindario.

### Etapa 2: Modelo Econométrico Aumentado

Incorporar los embeddings como covariables adicionales en el modelo logístico:

$$
\text{logit}(P[\text{Supervivencia}=1]) = \beta_0 + \beta_1 \text{RUC} + \beta_2 \text{RUC} \times \text{Región} + \mathbf{X}'\boldsymbol{\gamma} + \boldsymbol{\delta}' \mathbf{z}_i + \epsilon
$$

### Ventajas del Enfoque Híbrido

1. **Interpretabilidad preservada**: Los coeficientes $\beta$ y $\gamma$ mantienen su interpretación económica estándar.
2. **Contexto estructural incorporado**: Los embeddings $\mathbf{z}_i$ capturan información de vecindario que el modelo tradicional ignora.
3. **Comparabilidad directa**: Se puede testear si $\boldsymbol{\delta} \neq 0$ para determinar si la estructura de red aporta poder explicativo incremental.
4. **Compatibilidad con inferencia causal**: El marco logístico permite aplicar técnicas de identificación causal sobre los efectos de interés.

Este enfoque ha sido empleado exitosamente en múltiples dominios, incluyendo predicción de propiedades moleculares (Gilmer et al., 2017), sistemas de recomendación (Ying et al., 2018), y análisis de redes sociales (Hamilton et al., 2017).

---

En las siguientes secciones se describe la metodología específica de construcción del grafo, la arquitectura de la GNN empleada, y los resultados comparativos entre el modelo base, el GNN puro y el enfoque híbrido.
