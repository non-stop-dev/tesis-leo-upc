# 3. Análisis Metodológico: Construcción del Grafo Heterogéneo y Aprendizaje Inductivo

Este capítulo detalla la transformación de la base de datos censal de 1.3 millones de MIPYMES peruanas en una estructura de grafos adecuada para el aprendizaje profundo. Dada la naturaleza dinámica del tejido empresarial peruano y la escala de los datos, la metodología se fundamenta en un enfoque inductivo (GraphSAGE) sobre un grafo heterogéneo (HeteroData), permitiendo capturar interacciones complejas de aglomeración y competencia sin incurrir en costos computacionales prohibitivos.

## 3.1. Justificación del Enfoque Inductivo

A diferencia de los enfoques transductivos tradicionales, como GCN o Node2Vec, que aprenden embeddings fijos para cada nodo y requieren reentrenar el modelo ante la llegada de nuevas empresas, esta investigación adopta el marco GraphSAGE (Graph Sample and Aggregate) propuesto por Hamilton et al. (2017). En el contexto de supervivencia empresarial peruana, este enfoque es crítico porque permite predecir la supervivencia de nuevas empresas (nodos no vistos durante el entrenamiento) basándose en sus interacciones locales y atributos, sin necesidad de reentrenamiento. Asimismo, en lugar de procesar la matriz de adyacencia completa, GraphSAGE aprende funciones de agregación que muestrean vecindarios locales, haciendo viable el procesamiento de 1.3 millones de nodos.

## 3.2. Construcción del Grafo Heterogéneo

Para modelar eficientemente las relaciones de aglomeración geográfica y competencia sectorial sin generar una explosión combinatoria de aristas, se diseñó un esquema de grafo heterogéneo inspirado en las Redes de Atención de Grafos Heterogéneos (HAN) de Wang et al. (2019) y la agregación de meta-caminos de MAGNN descrita por Fu et al. (2020).

El grafo se define con tres tipos de nodos. Las MIPYMES, que constituyen la unidad de análisis principal, poseen atributos que incluyen variables financieras (ventas netas normalizadas, productividad laboral) y características estructurales (régimen tributario, tipo de local, score digital). Adicionalmente, se incorporan nodos de entidad que representan la ubicación geográfica (Distrito) y la clasificación industrial (Sector).

En lugar de conectar cada empresa con todas las demás en su mismo distrito, lo cual generaría un número cuadrático de aristas, se utilizan nodos intermedios para definir las relaciones. Primero, se establece una relación de ubicación donde una empresa se conecta con su distrito correspondiente. Esta relación permite que una empresa agregue mensajes de su entorno geográfico, induciendo un meta-camino MIPYME-Distrito-MIPYME que captura el efecto de aglomeración local validado en la literatura de economía urbana (Glaeser et al., 2010). Segundo, se define una relación de competencia donde cada empresa se conecta con su sector industrial a cuatro dígitos. El meta-camino resultante, MIPYME-Sector-MIPYME, modela los efectos de competencia intra-industrial y shocks sectoriales comunes.

Este diseño reduce la complejidad espacial de las aristas de cuadrática a lineal, permitiendo la viabilidad técnica del modelo con 1.3 millones de observaciones.

## 3.3. Tratamiento del Desbalance de Datos

El censo presenta un desbalance significativo: de los 1,377,931 establecimientos analizados, aproximadamente el 96.2% sobrevivió al período de estudio, mientras que solo el 3.8% cesó operaciones. Este desbalance extremo representa un desafío metodológico crítico, ya que un modelo no ajustado tendería a predecir supervivencia para todos los casos, obteniendo una exactitud aparentemente alta (96.2%) pero con nula capacidad predictiva para la clase de interés: el cierre empresarial.

### 3.3.1. Ponderación por Frecuencia Inversa

Para abordar este desbalance, se adopta la técnica de ponderación de clases por frecuencia inversa en la función de pérdida Cross-Entropy. Este enfoque, ampliamente validado en la literatura de aprendizaje automático para clasificación desbalanceada (He y García, 2009), asigna pesos inversamente proporcionales a la frecuencia de cada clase:

$$w_c = \frac{N}{C \times n_c}$$

donde $N$ representa el número total de observaciones, $C$ el número de clases, y $n_c$ la cantidad de observaciones en la clase $c$. En el contexto del presente estudio, esto implica que los errores en la clasificación de empresas que cesan operaciones se penalizan aproximadamente 25 veces más que los errores en empresas sobrevivientes, reflejando la proporción inversa de sus frecuencias.

La función de pérdida ponderada se define entonces como:

$$\mathcal{L} = -\sum_{i=1}^{N} w_{y_i} \log(\hat{p}_{y_i})$$

donde $y_i$ es la etiqueta verdadera del nodo $i$ y $\hat{p}_{y_i}$ es la probabilidad predicha para dicha clase.

### 3.3.2. Justificación frente a Alternativas

Se consideraron enfoques alternativos como la pérdida focal (Focal Loss) propuesta por Lin et al. (2017), que añade un factor modulador $(1-\hat{p})^\gamma$ para concentrar el aprendizaje en ejemplos difíciles. Sin embargo, la ponderación por frecuencia inversa fue seleccionada por tres razones: primero, minimiza la introducción de hiperparámetros adicionales que requerirían validación; segundo, mantiene la interpretabilidad directa de los pesos como corrección por desbalance poblacional; y tercero, ha demostrado desempeño competitivo en problemas de clasificación de nodos en grafos con desbalance severo, según Park et al. (2022) en su análisis sistemático de técnicas para GNNs desbalanceados.

Adicionalmente, se implementa muestreo estratificado en la partición entrenamiento-validación-prueba (70%-15%-15%) para garantizar que cada subconjunto preserve la distribución original de clases, evitando sesgos en la evaluación del modelo.
