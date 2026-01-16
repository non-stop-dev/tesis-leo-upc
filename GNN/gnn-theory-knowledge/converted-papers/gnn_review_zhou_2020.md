# Gnn Review Zhou 2020

> **Note**: This is an auto-converted document from PDF. Some formatting may be imperfect.

---

                                                     i An update to this article is included at the end


                                                                                AI Open 1 (2020) 57–81


                                                                      Contents lists available at ScienceDirect


                                                                                      AI Open
                                                  journal homepage: www.keaipublishing.com/en/journals/ai-open


Graph neural networks: A review of methods and applications
Jie Zhou a, 1, Ganqu Cui a, 1, Shengding Hu a, Zhengyan Zhang a, Cheng Yang b, Zhiyuan Liu a, *,
Lifeng Wang c, Changcheng Li c, Maosong Sun a
a
  Department of Computer Science and Technology, Tsinghua University, Beijing, China
b
  School of Computer Science, Beijing University of Posts and Telecommunications, China
c
  Tencent Incorporation, Shenzhen, China


A R T I C L E I N F O                                       A B S T R A C T

Keywords:                                                   Lots of learning tasks require dealing with graph data which contains rich relation information among elements.
Deep learning                                               Modeling physics systems, learning molecular ﬁngerprints, predicting protein interface, and classifying diseases
Graph neural network                                        demand a model to learn from graph inputs. In other domains such as learning from non-structural data like texts
                                                            and images, reasoning on extracted structures (like the dependency trees of sentences and the scene graphs of
                                                            images) is an important research topic which also needs graph reasoning models. Graph neural networks (GNNs)
                                                            are neural models that capture the dependence of graphs via message passing between the nodes of graphs. In
                                                            recent years, variants of GNNs such as graph convolutional network (GCN), graph attention network (GAT), graph
                                                            recurrent network (GRN) have demonstrated ground-breaking performances on many deep learning tasks. In this
                                                            survey, we propose a general design pipeline for GNN models and discuss the variants of each component, sys-
                                                            tematically categorize the applications, and propose four open problems for future research.


1. Introduction                                                                              neural networks for graphs. In the nineties, Recursive Neural Networks
                                                                                             are ﬁrst utilized on directed acyclic graphs (Sperduti and Starita, 1997;
    Graphs are a kind of data structure which models a set of objects                        Frasconi et al., 1998). Afterwards, Recurrent Neural Networks and
(nodes) and their relationships (edges). Recently, researches on                             Feedforward Neural Networks are introduced into this literature
analyzing graphs with machine learning have been receiving more and                          respectively in (Scarselli et al., 2009) and (Micheli, 2009) to tackle cy-
more attention because of the great expressive power of graphs, i.e.                         cles. Although being successful, the universal idea behind these methods
graphs can be used as denotation of a large number of systems across                         is building state transition systems on graphs and iterate until conver-
various areas including social science (social networks (Wu et al., 2020),                   gence, which constrained the extendability and representation ability.
natural science (physical systems (Sanchez et al., 2018; Battaglia et al.,                   Recent advancement of deep neural networks, especially convolutional
2016) and protein-protein interaction networks (Fout et al., 2017)),                         neural networks (CNNs) (LeCun et al., 1998) result in the rediscovery of
knowledge graphs (Hamaguchi et al., 2017) and many other research                            GNNs. CNNs have the ability to extract multi-scale localized spatial
areas (Khalil et al., 2017). As a unique non-Euclidean data structure for                    features and compose them to construct highly expressive representa-
machine learning, graph analysis focuses on tasks such as node classiﬁ-                      tions, which led to breakthroughs in almost all machine learning areas
cation, link prediction, and clustering. Graph neural networks (GNNs) are                    and started the new era of deep learning (LeCun et al., 2015). The keys of
deep learning based methods that operate on graph domain. Due to its                         CNNs are local connection, shared weights and the use of multiple layers
convincing performance, GNN has become a widely applied graph                                (LeCun et al., 2015). These are also of great importance in solving
analysis method recently. In the following paragraphs, we will illustrate                    problems on graphs. However, CNNs can only operate on regular
the fundamental motivations of graph neural networks.                                        Euclidean data like images (2D grids) and texts (1D sequences) while
    The ﬁrst motivation of GNNs roots in the long-standing history of                        these data structures can be regarded as instances of graphs. Therefore, it


  * Corresponding author.
    E-mail addresses: zhoujie18@mails.tsinghua.edu.cn (J. Zhou), cgq19@mails.tsinghua.edu.cn (G. Cui), hsd20@mails.tsinghua.edu.cn (S. Hu), zy-z19@mails.
tsinghua.edu.cn (Z. Zhang), albertyang33@gmail.com (C. Yang), liuzy@tsinghua.edu.cn (Z. Liu), fandywang@tencent.com (L. Wang), harrychli@tencent.com
(C. Li), sms@tsinghua.edu.cn (M. Sun).
  1
    indicates equal contribution.

https://doi.org/10.1016/j.aiopen.2021.01.001
Received 16 September 2020; Received in revised form 15 December 2020; Accepted 27 January 2021
Available online 8 April 2021
2666-6510/© 2021 The Author(s). Published by Elsevier B.V. on behalf of KeAi Communications Co., Ltd. This is an open access article under the CC BY license
(http://creativecommons.org/licenses/by/4.0/).
J. Zhou et al.                                                                                                                              AI Open 1 (2020) 57–81


is straightforward to generalize CNNs on graphs. As shown in Fig. 1, it is           different graph types and also provide a detailed summary of GNNs’
hard to deﬁne localized convolutional ﬁlters and pooling operators,                  applications in different domains.
which hinders the transformation of CNN from Euclidean domain to                         There have also been several surveys focusing on some speciﬁc graph
non-Euclidean domain. Extending deep neural models to non-Euclidean                  learning ﬁelds. Sun et al. (2018) and Chen et al. (2020a) give detailed
domains, which is generally referred to as geometric deep learning, has              overviews for adversarial learning methods on graphs, including graph
been an emerging research area (Bronstein et al., 2017). Under this                  data attack and defense. Lee et al. (2018a) provide a review over graph
umbrella term, deep learning on graphs receives enormous attention.                  attention models. The paper proposed by Yang et al. (2020) focuses on
    The other motivation comes from graph representation learning (Cui               heterogeneous graph representation learning, where nodes or edges are
et al., 2018a; Hamilton et al., 2017b; Zhang et al., 2018a; Cai et al., 2018;        of multiple types. Huang et al. (2020) review over existing GNN models
Goyal and Ferrara, 2018), which learns to represent graph nodes, edges               for dynamic graphs. Peng et al. (2020) summarize graph embeddings
or subgraphs by low-dimensional vectors. In the ﬁeld of graph analysis,              methods for combinatorial optimization. We conclude GNNs for het-
traditional machine learning approaches usually rely on hand engineered              erogeneous graphs, dynamic graphs and combinatorial optimization in
features and are limited by its inﬂexibility and high cost. Following the            Section 4.2, Section 4.3, and Section 8.1.6 respectively.
idea of representation learning and the success of word embedding                        In this paper, we provide a thorough review of different graph neural
(Mikolov et al., 2013), DeepWalk (Perozzi et al., 2014), regarded as the             network models as well as a systematic taxonomy of the applications. To
ﬁrst graph embedding method based on representation learning, applies                summarize, our contributions are:
SkipGram model (Mikolov et al., 2013) on the generated random walks.
Similar approaches such as node2vec (Grover and Leskovec, 2016), LINE                  We provide a detailed review over existing graph neural network
(Tang et al., 2015) and TADW (Yang et al., 2015) also achieved break-                   models. We present a general design pipeline and discuss the variants
throughs. However, these methods suffer from two severe drawbacks                       of each module. We also introduce researches on theoretical and
(Hamilton et al., 2017b). First, no parameters are shared between nodes                 empirical analyses of GNN models.
in the encoder, which leads to computationally inefﬁciency, since it                   We systematically categorize the applications and divide the appli-
means the number of parameters grows linearly with the number of                        cations into structural scenarios and non-structural scenarios. We
nodes. Second, the direct embedding methods lack the ability of gener-                  present several major applications and their corresponding methods
alization, which means they cannot deal with dynamic graphs or                          for each scenario.
generalize to new graphs.                                                              We propose four open problems for future research. We provide a
    Based on CNNs and graph embedding, variants of graph neural net-                    thorough analysis of each problem and propose future research
works (GNNs) are proposed to collectively aggregate information from                    directions.
graph structure. Thus they can model input and/or output consisting of
elements and their dependency.                                                           The rest of this survey is organized as follows. In Section 2, we present
    There exists several comprehensive reviews on graph neural net-                  a general GNN design pipeline. Following the pipeline, we discuss each
works. Bronstein et al. (2017) provide a thorough review of geometric                step in detail to review GNN model variants. The details are included in
deep learning, which presents its problems, difﬁculties, solutions, ap-              Section 3 to Section 6. In Section 7, we revisit research works over
plications and future directions. Zhang et al. (2019a) propose another               theoretical and empirical analyses of GNNs. In Section 8, we introduce
comprehensive overview of graph convolutional networks. However,                     several major applications of graph neural networks applied to structural
they mainly focus on convolution operators deﬁned on graphs while we                 scenarios, non-structural scenarios and other scenarios. In Section 9, we
investigate other computation modules in GNNs such as skip connections               propose four open problems of graph neural networks as well as several
and pooling operators.                                                               future research directions. And ﬁnally, we conclude the survey in Section
    Papers by Zhang et al. (2018b), Wu et al. (2019a), Chami et al. (2020)           10.
are the most up-to-date survey papers on GNNs and they mainly focus on
models of GNN. Wu et al. (2019a) categorize GNNs into four groups:                   2. General design pipeline of GNNs
recurrent graph neural networks, convolutional graph neural networks,
graph autoencoders, and spatial-temporal graph neural networks. Zhang                    In this paper, we introduce models of GNNs in a designer view. We ﬁrst
et al. (2018b) give a systematic overview of different graph deep learning           present the general design pipeline for designing a GNN model in this
methods and Chami et al. (2020) propose a Graph Encoder Decoder                      section. Then we give details of each step such as selecting computational
Model to unify network embedding and graph neural network models.                    modules, considering graph type and scale, and designing loss function in
Our paper provides a different taxonomy with them and we mainly focus                Section 3, 4, and 5, respectively. And ﬁnally, we use an example to illus-
on classic GNN models. Besides, we summarize variants of GNNs for                    trate the design process of GNN for a speciﬁc task in Section 6.


                                          Fig. 1. Left: image in Euclidean space. Right: graph in non-Euclidean space.

                                                                                58
J. Zhou et al.                                                                                                                           AI Open 1 (2020) 57–81


    In later sections, we denote a graph as G ¼ ðV;EÞ, where jVj ¼ N is the           Static/Dynamic Graphs. When input features or the topology of the
number of nodes in the graph and jEj ¼ N e is the number of edges. A 2                 graph vary with time, the graph is regarded as a dynamic graph. The
RNN is the adjacency matrix. For graph representation learning, we use                time information should be carefully considered in dynamic graphs.
hv and ov as the hidden state and output vector of node v. The detailed
descriptions of the notations could be found in Table 1.                                Note these categories are orthogonal, which means these types can be
    In this section, we present the general design pipeline of a GNN model          combined, e.g. one can deal with a dynamic directed heterogeneous
for a speciﬁc task on a speciﬁc graph type. Generally, the pipeline con-            graph. There are also several other graph types designed for different
tains four steps: (1) ﬁnd graph structure, (2) specify graph type and scale,        tasks such as hypergraphs and signed graphs. We will not enumerate all
(3) design loss function and (4) build model using computational mod-               types here but the most important idea is to consider the additional in-
ules. We give general design principles and some background knowledge               formation provided by these graphs. Once we specify the graph type, the
in this section. The design details of these steps are discussed in later           additional information provided by these graph types should be further
sections.                                                                           considered in the design process.
                                                                                        As for the graph scale, there is no clear classiﬁcation criterion for
                                                                                    “small” and “large” graphs. The criterion is still changing with the
2.1. Find graph structure
                                                                                    development of computation devices (e.g. the speed and memory of
                                                                                    GPUs). In this paper, when the adjacency matrix or the graph Laplacian of
   At ﬁrst, we have to ﬁnd out the graph structure in the application.
                                                                                    a graph (the space complexity is Oðn2 Þ) cannot be stored and processed
There are usually two scenarios: structural scenarios and non-structural
                                                                                    by the device, then we regard the graph as a large-scale graph and then
scenarios. In structural scenarios, the graph structure is explicit in the
                                                                                    some sampling methods should be considered.
applications, such as applications on molecules, physical systems,
knowledge graphs and so on. In non-structural scenarios, graphs are
implicit so that we have to ﬁrst build the graph from the task, such as
                                                                                    2.3. Design loss function
building a fully-connected “word” graph for text or building a scene
graph for an image. After we get the graph, the later design process at-
                                                                                       In this step we should design the loss function based on our task type
tempts to ﬁnd an optimal GNN model on this speciﬁc graph.
                                                                                    and the training setting.
                                                                                       For graph learning tasks, there are usually three kinds of tasks:
2.2. Specify graph type and scale
                                                                                      Node-level tasks focus on nodes, which include node classiﬁcation,
   After we get the graph in the application, we then have to ﬁnd out the              node regression, node clustering, etc. Node classiﬁcation tries to
graph type and its scale.                                                              categorize nodes into several classes, and node regression predicts a
   Graphs with complex types could provide more information on nodes                   continuous value for each node. Node clustering aims to partition the
and their connections. Graphs are usually categorized as:                              nodes into several disjoint groups, where similar nodes should be in
                                                                                       the same group.
  Directed/Undirected Graphs. Edges in directed graphs are all                       Edge-level tasks are edge classiﬁcation and link prediction, which
   directed from one node to another, which provide more information                   require the model to classify edge types or predict whether there is an
   than undirected graphs. Each edge in undirected graphs can also be                  edge existing between two given nodes.
   regarded as two directed edges.                                                    Graph-level tasks include graph classiﬁcation, graph regression, and
  Homogeneous/Heterogeneous Graphs. Nodes and edges in ho-                            graph matching, all of which need the model to learn graph
   mogeneous graphs have same types, while nodes and edges have                        representations.
   different types in heterogeneous graphs. Types for nodes and edges
   play important roles in heterogeneous graphs and should be further                   From the perspective of supervision, we can also categorize graph
   considered.                                                                      learning tasks into three different training settings:

  Table 1                                                                             Supervised setting provides labeled data for training.
  Notations used in this paper.                                                       Semi-supervised setting gives a small amount of labeled nodes and a
     Notations                         Descriptions
                                                                                       large amount of unlabeled nodes for training. In the test phase, the
                                                                                       transductive setting requires the model to predict the labels of the
    R    m
                                       m-dimensional Euclidean space
                                                                                       given unlabeled nodes, while the inductive setting provides new
    a; a; A                            Scalar, vector and matrix
    AT                                 Matrix transpose                                unlabeled nodes from the same distribution to infer. Most node and
    IN                                 Identity matrix of dimension N                  edge classiﬁcation tasks are semi-supervised. Most recently, a mixed
    gw ⋆ x                             Convolution of gw and x                         transductive-inductive scheme is undertaken by Wang and Leskovec
    N, N v                             Number of nodes in the graph                    (2020) and Rossi et al. (2018), craving a new path towards the mixed
    Ne                                 Number of edges in the graph
                                                                                       setting.
    Nv                                 Neighborhood set of node v
    atv                                Vector a of node v at time step t              Unsupervised setting only offers unlabeled data for the model to
    hv                                 Hidden state of node v                          ﬁnd patterns. Node clustering is a typical unsupervised learning task.
    htv                                Hidden state of node v at time step t
    otv                                Output of node v at time step t                  With the task type and the training setting, we can design a speciﬁc
    evw                                Features of edge from node v to w
                                                                                    loss function for the task. For example, for a node-level semi-supervised
    ek                                 Features of edge with label k
    W i ; Ui ;                         Matrices for computing i; o:                 classiﬁcation task, the cross-entropy loss can be used for the labeled
    Wo ; Uo                                                                         nodes in the training set.
    bi ; bo                            Vectors for computing i; o
     ρ                                 An alternative non-linear function
     σ                                 The logistic sigmoid function
                                                                                    2.4. Build model using computational modules
     tanh                              The hyperbolic tangent function
     LeakyReLU                         The LeakyReLU function
                                      Element-wise multiplication operation          Finally, we can start building the model using the computational
     k                                 Vector concatenation                         modules. Some commonly used computational modules are:

                                                                               59
J. Zhou et al.                                                                                                                              AI Open 1 (2020) 57–81


  Propagation Module. The propagation module is used to propagate                    modules. We introduce three sub-components of propagation modules:
   information between nodes so that the aggregated information could                 convolution operator, recurrent operator and skip connection in Section
   capture both feature and topological information. In propagation                   3.1, 3.2, and 3.3 respectively. Then we introduce sampling modules and
   modules, the convolution operator and recurrent operator are                       pooling modules in Section 3.4 and 3.5. An overview of computational
   usually used to aggregate information from neighbors while the skip                modules is shown in Fig. 3.
   connection operation is used to gather information from historical
   representations of nodes and mitigate the over-smoothing problem.                  3.1. Propagation modules - convolution operator
  Sampling Module. When graphs are large, sampling modules are
   usually needed to conduct propagation on graphs. The sampling                          Convolution operators that we introduce in this section are the mostly
   module is usually combined with the propagation module.                            used propagation operators for GNN models. The main idea of convolu-
  Pooling Module. When we need the representations of high-level                     tion operators is to generalize convolutions from other domain to the
   subgraphs or graphs, pooling modules are needed to extract infor-                  graph domain. Advances in this direction are often categorized as spec-
   mation from nodes.                                                                 tral approaches and spatial approaches.

    With these computation modules, a typical GNN model is usually                    3.1.1. Spectral approaches
built by combining them. A typical architecture of the GNN model is                       Spectral approaches work with a spectral representation of the
illustrated in the middle part of Fig. 2 where the convolutional operator,            graphs. These methods are theoretically based on graph signal processing
recurrent operator, sampling module and skip connection are used to                   (Shuman et al., 2013) and deﬁne the convolution operator in the spectral
propagate information in each layer and then the pooling module is                    domain.
added to extract high-level information. These layers are usually stacked                 In spectral methods, a graph signal x is ﬁrstly transformed to the
to obtain better representations. Note this architecture can generalize               spectral domain by the graph Fourier transform F , then the convolution
most GNN models while there are also exceptions, for example, NDCN                    operation is conducted. After the convolution, the resulted signal is
(Zang and Wang, 2020) combines ordinary differential equation systems
                                                                                      transformed back using the inverse graph Fourier transform F 1 . These
(ODEs) and GNNs. It can be regarded as a continuous-time GNN model
                                                                                      transforms are deﬁned as:
which integrates GNN layers over continuous time without propagating
through a discrete number of layers.                                                  F ðxÞ ¼ UT x;
                                                                                                                                                              (1)
    An illustration of the general design pipeline is shown in Fig. 2. In             F 1 ðxÞ ¼ Ux:
later sections, we ﬁrst give the existing instantiations of computational
modules in Section 3, then introduce existing variants which consider                 Here U is the matrix of eigenvectors of the normalized graph Laplacian
                                                                                      L ¼ IN  D2 AD2 (D is the degree matrix and A is the adjacency matrix
                                                                                                  1     1
different graph types and scale in Section 4. Then we survey on variants
designed for different training settings in Section 5. These sections                 of the graph). The normalized graph Laplacian is real symmetric positive
correspond to details of step (4), step (2), and step (3) in the pipeline. And        semideﬁnite, so it can be factorized as L ¼ UΛUT (where Λ is a diagonal
ﬁnally, we give a concrete design example in Section 6.                               matrix of the eigenvalues). Based on the convolution theorem (Mallat,
                                                                                      1999), the convolution operation is deﬁned as:
3. Instantiations of computational modules
                                                                                      g ⋆ x ¼ F 1 ðF ðgÞ  F ðxÞÞ
                                                                                                                                                              (2)
    In this section we introduce existing instantiations of three compu-                    ¼ UðUT g  UT xÞ;
tational modules: propagation modules, sampling modules and pooling
                                                                                      where UT g is the ﬁlter in the spectral domain. If we simplify the ﬁlter by


                                                      Fig. 2. The general design pipeline for a GNN model.

                                                                                 60
J. Zhou et al.                                                                                                                              AI Open 1 (2020) 57–81


                                                      Fig. 3. An overview of computational modules.


using a learnable diagonal matrix gw , then we have the basic function of         of the eigenvalues in L~ is [-1, 1]. w 2 RK is now a vector of Chebyshev
the spectral methods:                                                             coefﬁcients. The Chebyshev polynomials are deﬁned as Tk ðxÞ ¼
                                                                                  2xTk1 ðxÞ  Tk2 ðxÞ, with T0 ðxÞ ¼ 1 and T1 ðxÞ ¼ x. It can be observed
gw ⋆ x ¼ Ugw UT x:                                                    (3)
                                                                                  that the operation is K-localized since it is a K th -order polynomial in the
     Next we introduce several typical spectral methods which design              Laplacian. Defferrard et al. (2016) use this K-localized convolution to
different ﬁlters gw .                                                             deﬁne a convolutional neural network which could remove the need to
     Spectral Network. Spectral network (Bruna et al., 2014) uses a learn-        compute the eigenvectors of the Laplacian.
able diagonal matrix as the ﬁlter, that is gw ¼ diagðwÞ, where w 2 RN is              GCN. Kipf and Welling (2017) simplify the convolution operation in
the parameter. However, this operation is computationally inefﬁcient              Eq. (4) with K ¼ 1 to alleviate the problem of overﬁtting. They further
and the ﬁlter is non-spatially localized. Henaff et al. (2015) attempt to         assume λmax  2 and simplify the equation to
make the spectral ﬁlters spatially localized by introducing a parameter-                                                      1     1
ization with smooth coefﬁcients.                                                  gw ⋆ x  w0 x þ w1 ðL  IN Þx ¼ w0 x  w1 D2 AD2 x                        (5)
     ChebNet. Hammond et al. (2011) suggest that gw can be approximated
by a truncated expansion in terms of Chebyshev polynomials Tk ðxÞ up to           with two free parameters w0 and w1 . With parameter constraint w ¼
                                                                                  w0 ¼  w1 , we can obtain the following expression:
K th order. Defferrard et al. (2016) propose the ChebNet based on this
theory. Thus the operation can be written as:                                                0                1

            X
            K                                                                   gw ⋆ x  w@IN þ D AD12   12 A
                                                                                                                x:                                            (6)
gw ⋆ x                 ~ x;
                  wk Tk L                                             (4)
            k¼0
                                                                                     GCN further introduces a renormalization trick to solve the exploding/
      ~ ¼ 2 L  IN , λmax denotes the largest eigenvalue of L. The range
where L                                                                                                                                     ~D
                                                                                                                                         ~ 2A
                                                                                  vanishing gradient problem in Eq. (6): IN þ D2 AD2 → D
                                                                                                                                        1   1      1
                                                                                                                                              ~ 2 , with 1
         λmax


                                                                             61
J. Zhou et al.                                                                                                                                             AI Open 1 (2020) 57–81

~ ¼ A þ IN and D     P~
A              ~ ii ¼ A ij . Finally, the compact form of GCN is deﬁned                              for nodes. For node classiﬁcation, the diffusion representations of each
                                    j                                                                node in the graph can be expressed as:
as:
          1      1
                                                                                                     H ¼ f ðWc  P* XÞ 2 RNKF ;                                           (10)
  ~ 2 A
H¼D    ~D~ 2 XW;                                                                         (7)
                                                                                                     where X 2 RNF is the matrix of input features (F is the dimension). P* is
                                                                 0
where X 2 R          NF
                              is the input matrix, W 2 R   FF
                                                                     is the parameter and H 2        an N  K  N tensor which contains the power series {P; P2 , …, PK } of
      0                                             0
RNF is the convolved matrix. F and F are the dimensions of the input                                matrix P. And P is the degree-normalized transition matrix from the
and the output, respectively. Note that GCN can also be regarded as a                                graphs adjacency matrix A. Each entity is transformed to a diffusion
spatial method that we will discuss later.                                                           convolutional representation which is a K  F matrix deﬁned by K hops
    AGCN. All of these models use the original graph structure to denote                             of graph diffusion over F features. And then it will be deﬁned by a K  F
relations between nodes. However, there may have implicit relations                                  weight matrix and a non-linear activation function f.
between different nodes. The Adaptive Graph Convolution Network                                          PATCHY-SAN. The PATCHY-SAN model (Niepert et al., 2016) ex-
(AGCN) is proposed to learn the underlying relations (Li et al., 2018a).                             tracts and normalizes a neighborhood of exactly k nodes for each node.
AGCN learns a “residual” graph Laplacian and add it to the original                                  The normalized neighborhood serves as the receptive ﬁeld in the tradi-
Laplacian matrix. As a result, it is proven to be effective in several                               tional convolutional operation.
graph-structured datasets.                                                                               LGCN. The learnable graph convolutional network (LGCN) (Gao et al.,
    DGCN. The dual graph convolutional network (DGCN) (Zhuang and                                    2018a) also exploits CNNs as aggregators. It performs max pooling on
Ma, 2018) is proposed to jointly consider the local consistency and global                           neighborhood matrices of nodes to get top-k feature elements and then
consistency on graphs. It uses two convolutional networks to capture the                             applies 1-D CNN to compute hidden representations.
local and global consistency and adopts an unsupervised loss to ensemble                                 GraphSAGE. GraphSAGE (Hamilton et al., 2017a) is a general
them. The ﬁrst convolutional network is the same as Eq. (7), and the                                 inductive framework which generates embeddings by sampling and
second network replaces the adjacency matrix with positive pointwise                                 aggregating features from a node’s local neighborhood:
mutual information (PPMI) matrix:                                                                                   t             
                                                                                                     htþ1
                                                                                                      N v ¼ AGGtþ1    hu ; 8u 2 N v ;
          0                        1                                                                               h           i                                          (11)
                                                                                                     htþ1
                                                                                                      v   ¼ σ Wtþ1  htv k htþ1
                                                                                                                             Nv   :
  ’  B 1      1    C
H ¼ ρ@DP 2 AP DP 2 HWA;                                                                   (8)
                                                                                                        Instead of using the full neighbor set, GraphSAGE uniformly samples
                                                                                                     a ﬁxed-size set of neighbors to aggregate information. AGGtþ1 is the ag-
where AP is the PPMI matrix and DP is the diagonal degree matrix of AP .                             gregation function and GraphSAGE suggests three aggregators: mean
    GWNN. Graph wavelet neural network (GWNN) (Xu et al., 2019a)                                     aggregator, LSTM aggregator, and pooling aggregator. GraphSAGE with
uses the graph wavelet transform to replace the graph Fourier transform.                             a mean aggregator can be regarded as an inductive version of GCN while
It has several advantages: (1) graph wavelets can be fastly obtained                                 the LSTM aggregator is not permutation invariant, which requires a
without matrix decomposition; (2) graph wavelets are sparse and local-                               speciﬁed order of the nodes.
ized thus the results are better and more explainable. GWNN outperforms
several spectral methods on the semi-supervised node classiﬁcation task.                             3.1.3. Attention-based spatial approaches
    AGCN and DGCN try to improve spectral methods from the perspec-                                      The attention mechanism has been successfully used in many
tive of augmenting graph Laplacian while GWNN replaces the Fourier                                   sequence-based tasks such as machine translation (Bahdanau et al., 2015;
transform. In conclusion, spectral approaches are well theoretically based                           Gehring et al., 2017; Vaswani et al., 2017), machine reading (Cheng
and there are also several theoretical analyses proposed recently (see                               et al., 2016) and so on. There are also several models which try to
Section 7.1.1). However, in almost all of the spectral approaches                                    generalize the attention operator on graphs (Velickovic et al., 2018;
mentioned above, the learned ﬁlters depend on graph structure. That is to                            Zhang et al., 2018c). Compared with the operators we mentioned before,
say, the ﬁlters cannot be applied to a graph with a different structure and                          attention-based operators assign different weights for neighbors, so that
those models can only be applied under the “transductive” setting of                                 they could alleviate noises and achieve better results.
graph tasks.                                                                                             GAT. The graph attention network (GAT) (Velickovic et al., 2018)
                                                                                                     incorporates the attention mechanism into the propagation step. It
3.1.2. Basic spatial approaches                                                                      computes the hidden states of each node by attending to its neighbors,
    Spatial approaches deﬁne convolutions directly on the graph based on                             following a self-attention strategy. The hidden state of node v can be ob-
the graph topology. The major challenge of spatial approaches is deﬁning                             tained by:
the convolution operation with differently sized neighborhoods and                                                                          !
                                                                                                                             X
maintaining the local invariance of CNNs.                                                                          htþ1 ¼ρ           αvu Whtu ;
                                                                                                                    v
    Neural FPs. Neural FPs (Duvenaud et al., 2015) uses different weight                                                     u2N v
matrices for nodes with different degrees:                                                                                                                                  (12)
                 X                                                                                           expðLeakyReLUðaT ½Whv k Whu ÞÞ
                                                                                                     αvu ¼ X                                      ;
  t ¼ htv þ           htu ;                                                                                        expðLeakyReLUðaT ½Whv k Whk ÞÞ
              u2N v                                                                                        k2N v
                                                                                        (9)
htþ1 ¼    σ tWtþ1
              jN v j ;
 v                                                                                                   where W is the weight matrix associated with the linear transformation
                                                                                                     which is applied to each node, and a is the weight vector of a single-layer
where Wtþ1
       jN v j is the weight matrix for nodes with degree jN v j at layer t þ                         MLP.
1. The main drawback of the method is that it cannot be applied to large-                                Moreover, GAT utilizes the multi-head attention used by Vaswani et al.
scale graphs with more node degrees.                                                                 (2017) to stabilize the learning process. It applies K independent atten-
    DCNN. The diffusion convolutional neural network (DCNN) (Atwood                                  tion head matrices to compute the hidden states and then concatenates
and Towsley, 2016) uses transition matrices to deﬁne the neighborhood                                their features (or computes the average), resulting in the following two


                                                                                                62
J. Zhou et al.                                                                                                                                    AI Open 1 (2020) 57–81

                                                                                                  X                       
output representations:                                                                 mtþ1
                                                                                         v   ¼           Mt htv ; htu ; evu ;
                                          !                                                      u2N v                                                             (15)
                         X                                                                                  
htþ1 ¼ kKk¼1 σ                  αkvu Wk htu ;                                           htþ1
                                                                                         v   ¼ Ut htv ; mtþ1
                                                                                                         v     :
 v
                        u2N v

                          !                                                (13)         Here evu represents features of undirected edge ðv; uÞ. The readout phase
 tþ1   1 XK X
                   k    t                                                               computes a feature vector of the whole graph using the readout function
hv ¼ σ            α Wk hu :
       K k¼1 u2N v vu                                                                   R:
                                                                                                T      
    Here αkij is the normalized attention coefﬁcient computed by the k-th               b
                                                                                        y ¼R     hv v 2 G ;                                                        (16)
attention head. The attention architecture has several properties: (1) the
computation of the node-neighbor pairs is parallelizable thus the oper-                 where T denotes the total time steps. The message function Mt , vertex
ation is efﬁcient; (2) it can be applied to graph nodes with different de-              update function Ut and readout function R may have different settings.
grees by specifying arbitrary weights to neighbors; (3) it can be applied to            Hence the MPNN framework could instantiate several different models
the inductive learning problems easily.                                                 via different function settings. Speciﬁc settings for different models could
    GaAN. The gated attention network (GaAN) (Zhang et al., 2018c) also                 be found in (Gilmer et al., 2017).
uses the multi-head attention mechanism. However, it uses a                                 NLNN. The non-local neural network (NLNN) generalizes and extends
self-attention mechanism to gather information from different heads to                  the classic non-local mean operation (Buades et al., 2005) in computer
replace the average operation of GAT.                                                   vision. The non-local operation computes the hidden state at a position as
                                                                                        a weighted sum of features at all possible positions. The potential posi-
3.1.4. General frameworks for spatial approaches                                        tions can be in space, time or spacetime. Thus the NLNN can be viewed as
    Apart from different variants of spatial approaches, several general                a uniﬁcation of different “self-attention”-style methods (Hoshen, 2017;
frameworks are proposed aiming to integrate different models into one                   Vaswani et al., 2017; Velickovic et al., 2018).
single framework. Monti et al. (2017) propose the mixture model                             Following the non-local mean operation (Buades et al., 2005), the
network (MoNet), which is a general spatial framework for several                       generic non-local operation is deﬁned as
methods deﬁned on graphs or manifolds. Gilmer et al. (2017) propose the
                                                                                                 1 X  t t  t
message passing neural network (MPNN), which uses message passing                       htþ1
                                                                                         v ¼              f hv ; hu g hu ;                                         (17)
                                                                                               C ðht Þ 8u
functions to unify several variants. Wang et al. (2018a) propose the
non-local neural network (NLNN) which uniﬁes several “self--
                                                                                        where u is the index of all possible positions for position v, f ðhtv ; htu Þ
attention”-style methods (Hoshen, 2017; Vaswani et al., 2017; Velickovic
                                                                                        computes a scalar between v and u representing the relation between
et al., 2018). Battaglia et al. (2018) propose the graph network (GN). It
deﬁnes a more general framework for learning node-level, edge-level and                 them, gðhtu Þ denotes a transformation of the input htu and C ðht Þ is a
graph-level representations.                                                            normalization factor. Different variants of NLNN can be deﬁned by
    MoNet. Mixture model network (MoNet) (Monti et al., 2017) is a                      different f and g settings and more details can be found in the original
spatial framework that try to uniﬁes models for non-euclidean do-                       paper (Buades et al., 2005).
mains, including CNNs for manifold and GNNs. The Geodesic CNN                               Graph Network. The graph network (GN) (Battaglia et al., 2018) is a
(GCNN) (Masci et al., 2015) and Anisotropic CNN (ACNN) (Boscaini                        more general framework compared to others by learning node-level,
et al., 2016) on manifolds or GCN (Kipf and Welling, 2017) and DCNN                     edge-level and graph level representations. It can unify many variants
(Atwood and Towsley, 2016) on graphs can be formulated as partic-                       like MPNN, NLNN, Interaction Networks (Battaglia et al., 2016; Watters
ular instances of MoNet. In MoNet, each point on a manifold or each                     et al., 2017), Neural Physics Engine (Chang et al., 2017), CommNet
vertex on a graph, denoted by v, is regarded as the origin of a                         (Sukhbaatar Ferguset al., 2016), structure2vec (Dai et al., 2016; Khalil
pseudo-coordinate system. The neighbors u 2 N v are associated with                     et al., 2017), GGNN (Li et al., 2016), Relation Network (Raposo et al.,
pseudo-coordinates uðv; uÞ. Given two functions f ; g deﬁned on the                     2017; Santoro et al., 2017), Deep Sets (Zaheer et al., 2017), Point Net (Qi
vertices of a graph (or points on a manifold), the convolution operator                 et al., 2017a) and so on.
in MoNet is deﬁned as:                                                                      The core computation unit of GN is called the GN block. A GN block
                                                                                        deﬁnes three update functions and three aggregation functions:
             X
             J
                                                                                                                                         
ðf ⋆ gÞ ¼              gj Dj ðvÞf ;                                                     etþ1 ¼ φe etk ; htrk ; htsk ; ut ; etþ1 ¼ ρe→h Etþ1   ;
                                                                                         k                                  v           v
                 j¼1                                                       (14)
             X                                                                                                                    
Dj ðvÞf ¼              wj ðuðv; uÞÞf ðuÞ:                                               htþ1 ¼ φh etþ1     t   t
                                                                                                                 ; etþ1 ¼ ρe→u Etþ1 ;
                                                                                         v          v ; hv ; u                                                     (18)
            u2N v
                                                                                                         tþ1
                                                                                                                  tþ1               
                                                                                        utþ1 ¼ φu etþ1 ; h ; ut ; h ¼ ρh→u Htþ1 :
Here w1 ðuÞ; …; wJ ðuÞ are the functions assigning weights for neighbors
according to their pseudo-coordinates. Thus the Dj ðvÞf is the aggregated
values of the neighbors’ functions. By deﬁning different u and w, MoNet                 Here rk is the receiver node and sk is the sender node of edge k. Etþ1 and
can instantiate several methods. For GCN, the function f ; g map the nodes              Htþ1 are the matrices of stacked edge vectors and node vectors at time
to their features, the pseudo-coordinate for ðv; uÞ is uðv;uÞ ¼ ðjN v j;jN u jÞ,        step tþ1, respectively. Etþ1
                                                                                                                   v   collects edge vectors with receiver node v. u
J ¼ 1 and w1 ðuðv;uÞÞ ¼ pﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃﬃ
                             1
                                       . In MoNet’s own model, the parameters           is the global attribute for graph representation. The φ and ρ functions can
                                         jN v jjN u j
                                                                                        have various settings and the ρ functions must be invariant to input or-
wj are learnable.                                                                       ders and should take variable lengths of arguments.
   MPNN. The message passing neural network (MPNN) (Gilmer et al.,
2017) extracts the general characteristics among several classic models.
                                                                                        3.2. Propagation modules - recurrent operator
The model contains two phases: a message passing phase and a readout
phase. In the message passing phase, the model ﬁrst uses the message
                                                                                            Recurrent methods are pioneers in this research line. The major dif-
function Mt to aggregate the “message” mtv from neighbors and then uses
                                                                                        ference between recurrent operators and convolution operators is that
the update function Ut to update the hidden state htv :
                                                                                        layers in convolution operators use different weights while layers in


                                                                                   63
J. Zhou et al.                                                                                                                                                             AI Open 1 (2020) 57–81


Table 2                                                                                                            where F, the global transition function, and G, the global output function are
Different variants of recurrent operators.
                                                                                                                   stacked versions of f and g for all nodes in a graph, respectively. The value
  Variant                 Aggregator                         Updater                                               of H is the ﬁxed point of Eq. (20) and is uniquely deﬁned with the
                                  P t1                                                                            assumption that F is a contraction map.
  GGNN                    htN v ¼    hk þ b                  ztv ¼ σ ðWz htN v þ Uz ht1
                                                                                     v Þ
                                     k2N v
                                                             rtv ¼ σ ðWr htN v þ Ur ht1
                                                                                     v Þ
                                                                                                                       With the suggestion of Banach’s ﬁxed point theorem (Khamsi and
                                                             h~t ¼ tanhðWht þ Uðrt ht1 ÞÞ
                                                              v                N v         v       v
                                                                                                                   Kirk, 2011), GNN uses the following classic iterative scheme to compute
                                                             htv ¼ ð1  ztv Þ  ht1 þ ztv  h~tv                  the state:
                                                                                 v
                                       P
  Tree LSTM               htiN v ¼           Ui ht1
                                                 k           itv ¼ σ ðWi xtv þ htiN v þ bi Þ                       Htþ1 ¼ FðHt ; XÞ;                                                        (21)
    (Child sum)                      k2N v
                                                             f tvk ¼ σ ðWf xtv þ htfN v k þ bf Þ
                          htfN v k ¼ Uf ht1
                                          k
                                     P o t1                 otv ¼ σ ðWo xtv þ hto
                                                                                Nv þ b Þ
                                                                                      o
                                                                                                                   where Ht denotes the t-th iteration of H. The dynamical system Eq. (21)
                          hto
                           Nv ¼           U hk
                                    k2N v                    utv ¼ tanhðWu xtv þ htuN þ b Þ
                                                                                           u
                                                                                                                   converges exponentially fast to the solution for any initial value.
                                     P u t1                                   P t v t1
                          htu
                           Nv ¼           U hk               ctv ¼ itv  utv þ   f vk  ck                            Though experimental results have shown that GNN is a powerful
                                    k2N v                                      k2N v
                                                             htv ¼ otv  tanhðctv Þ                                architecture for modeling structural data, there are still several
  Tree LSTM (N-ary)                  P
                                     K                                                                             limitations:
                          htiN v ¼         Uil ht1
                                                vl
                                     l¼1

                          htfN v k ¼
                                      PK
                                            Ufkl ht1
                                                                                                                     GNN requires f to be a contraction map which limits the model’s
                                                  vl
                                     l¼1                                                                              ability. And it is inefﬁcient to update the hidden states of nodes
                                     P
                                     K
                          hto
                           Nv ¼            Uol ht1
                                                vl
                                                                                                                      iteratively towards the ﬁxed point.
                                     l¼1
                                                                                                                     It is unsuitable to use the ﬁxed points if we focus on the representa-
                                     PK
                          htu
                           Nv ¼            Uul ht1
                                                vl                                                                    tion of nodes instead of graphs because the distribution of represen-
                                     l¼1
                                       P                                                                              tation in the ﬁxed point will be much smoother in value and less
  Graph LSTM in           htiN v ¼           Uimðv;kÞ ht1
                                                       k
  (Peng et al., 2017)                k2N v                                                                            informative for distinguishing each node.
                          htfN v k ¼ Ufmðv;kÞ ht1
                                               k
                                     P o
                          hto
                           Nv ¼          Umðv;kÞ ht1
                                                   k                                                                   GraphESN. Graph echo state network (GraphESN) (Gallicchio and
                                     k2N v
                                       P                                                                           Micheli, 2010) generalizes the echo state network (ESN) (Jaeger, 2001)
                          htu
                           Nv ¼              Uumðv;kÞ ht1
                                                       k
                                     k2N v                                                                         on graphs. It uses a ﬁxed contractive encoding function, and only trains a
                                                                                                                   readout function. The convergence is ensured by the contractivity of
                                                                                                                   reservoir dynamics. As a consequence, GraphESN is more efﬁcient than
recurrent operators share same weights. Early methods based on recur-                                              GNN.
sive neural networks focus on dealing with directed acyclic graphs                                                     SSE. Stochastic Steady-state Embedding (SSE) (Dai et al., 2018a) is
(Sperduti and Starita, 1997; Frasconi et al., 1998; Micheli et al., 2004;                                          also proposed to improve the efﬁciency of GNN. SSE proposes a learning
Hammer et al., 2004). Later, the concept of graph neural network (GNN)                                             framework which contains two steps. Embeddings of each node are
was ﬁrst proposed in (Scarselli et al., 2009; Gori et al., 2005), which                                            updated by a parameterized operator in the update step and these em-
extended existing neural networks to process more graph types. We name                                             beddings are projected to the steady state constraint space to meet the
the model as GNN in this paper to distinguish it with the general name.                                            steady-state conditions.
We ﬁrst introduce GNN and its later variants which require convergence                                                 LP-GNN. Lagrangian Propagation GNN (LP-GNN) (Tiezzi et al., 2020)
of the hidden states and then we talk about methods based on the gate                                              formalizes the learning task as a constraint optimization problem in the
mechanism.                                                                                                         Lagrangian framework and avoids the iterative computations for the
                                                                                                                   ﬁxed point. The convergence procedure is implicitly expressed by a
3.2.1. Convergence-based methods                                                                                   constraint satisfaction mechanism.
   In a graph, each node is naturally deﬁned by its features and the
                                                                                                                   3.2.2. Gate-based methods
related nodes. The target of GNN is to learn a state embedding hv 2 Rs
                                                                                                                       There are several works attempting to use the gate mechanism like
which contains the information of the neighborhood and itself for each
                                                                                                                   GRU (Cho et al., 2014) or LSTM (Hochreiter and Schmidhuber, 1997) in
node. The state embedding hv is an s-dimension vector of node v and can
                                                                                                                   the propagation step to diminish the computational limitations in GNN
be used to produce an output ov such as the distribution of the predicted
                                                                                                                   and improve the long-term propagation of information across the graph
node label. Then the computation steps of hv and ov are deﬁned as:
                                                                                                                   structure. They run a ﬁxed number of training steps without the guar-
                                                                                                                 antee of convergence.
hv ¼ f xv ; xco½v ; hN v ; xN v ;
                                                                                                       (19)            GGNN. The gated graph neural network (GGNN) (Li et al., 2016) is
ov ¼ gðhv ; xv Þ;
                                                                                                                   proposed to release the limitations of GNN. It releases the requirement of
where xv ; xco½v ; hN v ; xN v are the features of v, the features of its edges, the                              function f to be a contraction map and uses the Gate Recurrent Units
states and the features of the nodes in the neighborhood of v, respec-                                             (GRU) in the propagation step. It also uses back-propagation through
tively. f here is a parametric function called the local transition function. It                                   time (BPTT) to compute gradients. The computation step of GGNN can be
is shared among all nodes and updates the node state according to the                                              found in Table 2.
input neighborhood. g is the local output function that describes how the                                              The node v ﬁrst aggregates messages from its neighbors. Then the
output is produced. Note that both f and g can be interpreted as the                                               GRU-like update functions incorporate information from the other nodes
feedforward neural networks.                                                                                       and from the previous timestep to update each node’s hidden state. hN v
    Let H, O, X, and XN be the matrices constructed by stacking all the                                            gathers the neighborhood information of node v, while z and r are the
states, all the outputs, all the features, and all the node features,                                              update and reset gates.
respectively. Then we have a compact form as:                                                                          LSTMs are also used in a similar way as GRU through the propagation
                                                                                                                   process based on a tree or a graph.
H ¼ FðH; XÞ;                                                                                                           Tree LSTM. Tai et al. (2015) propose two extensions on the tree
                                                                                                       (20)
O ¼ GðH; XN Þ;                                                                                                     structure to the basic LSTM architecture: the Child-Sum Tree-LSTM and


                                                                                                              64
J. Zhou et al.                                                                                                                                 AI Open 1 (2020) 57–81


the N-ary Tree-LSTM. They are also extensions to the recursive neural                    LSTM-attention in the experiments to aggregate information. The JKN
network based models as we mentioned before. Tree is a special case of                   performs well on the experiments in social, bioinformatics and citation
graph and each node in Tree-LSTM aggregates information from its                         networks. It can also be combined with models like GCN, GraphSAGE and
children. Instead of a single forget gate in traditional LSTM, the                       GAT to improve their performance.
Tree-LSTM unit for node v contains one forget gate f vk for each child k.                   DeepGCNs. Li et al. (2019a) borrow ideas from ResNet (He et al.,
The computation step of the Child-Sum Tree-LSTM is displayed in                          2016a, 2016b) and DenseNet (Huang et al., 2017). ResGCN and Den-
Table 2. itv , otv , and ctv are the input gate, output gate and memory cell             seGCN are proposed by incorporating residual connections and dense
respectively. xtv is the input vector at time t. The N-ary Tree-LSTM is                  connections to solve the problems of vanishing gradient and over
further designed for a special kind of tree where each node has at most K                smoothing. In detail, the hidden state of a node in ResGCN and Den-
children and the children are ordered. The equations for computing htiN v ;              seGCN can be computed as:

htfN v k ; hto
            N v ; hN v in Table 2 introduce separate parameters for each child k.
                   tu
                                                                                         htþ1
                                                                                          Res ¼ h
                                                                                                  tþ1
                                                                                                      þ ht ;
                                                                                                                                                                (23)
These parameters allow the model to learn more ﬁne-grained represen-                      htþ1      tþ1 i
                                                                                           Dense ¼ ki¼0 h :
tations conditioning on the states of a unit’s children than the Child-Sum
Tree-LSTM.                                                                                  The experiments of DeepGCNs are conducted on the point cloud se-
    Graph LSTM. The two types of Tree-LSTMs can be easily adapted to                     mantic segmentation task and the best results are achieved with a 56-
the graph. The graph-structured LSTM in (Zayats and Ostendorf, 2018) is                  layer model.
an example of the N-ary Tree-LSTM applied to the graph. However, it is a
simpliﬁed version since each node in the graph has at most 2 incoming                    3.4. Sampling modules
edges (from its parent and sibling predecessor). Peng et al. (2017) pro-
pose another variant of the Graph LSTM based on the relation extraction                      GNN models aggregate messages for each node from its neighborhood
task. The edges of graphs in (Peng et al., 2017) have various labels so that             in the previous layer. Intuitively, if we track back multiple GNN layers,
Peng et al. (2017) utilize different weight matrices to represent different              the size of supporting neighbors will grow exponentially with the depth.
labels. In Table 2, mðv; kÞ denotes the edge label between node v and k.                 To alleviate this “neighbor explosion” issue, an efﬁcient and efﬁcacious
    Liang et al. (2016) propose a Graph LSTM network to address the                      way is sampling. Besides, when we deal with large graphs, we cannot
semantic object parsing task. It uses the conﬁdence-driven scheme to                     always store and process all neighborhood information for each node,
adaptively select the starting node and determine the node updating                      thus the sampling module is needed to conduct the propagation. In this
sequence. It follows the same idea of generalizing the existing LSTMs into               section, we introduce three kinds of graph sampling modules: node
the graph-structured data but has a speciﬁc updating sequence while                      sampling, layer sampling, and subgraph sampling.
methods mentioned above are agnostic to the order of nodes.
    S-LSTM. Zhang et al. (2018d) propose Sentence LSTM (S-LSTM) for                      3.4.1. Node sampling
improving text encoding. It converts text into a graph and utilizes the                      A straightforward way to reduce the size of neighboring nodes would be
Graph LSTM to learn the representation. The S-LSTM shows strong rep-                     selecting a subset from each node’s neighborhood. GraphSAGE (Hamilton
resentation power in many NLP problems.                                                  et al., 2017a) samples a ﬁxed small number of neighbors, ensuring a 2 to 50
                                                                                         neighborhood size for each node. To reduce sampling variance, Chen et al.
3.3. Propagation modules - skip connection                                               (2018a) introduce a control-variate based stochastic approximation algo-
                                                                                         rithm for GCN by utilizing the historical activations of nodes as a control
    Many applications unroll or stack the graph neural network layer                     variate. This method limits the receptive ﬁeld in the 1-hop neighborhood,
aiming to achieve better results as more layers (i.e k layers) make each                 and uses the historical hidden state as an affordable approximation.
node aggregate more information from neighbors k hops away. However,                         PinSage (Ying et al., 2018a) proposes importance-based sampling
it has been observed in many experiments that deeper models could not                    method. By simulating random walks starting from target nodes, this
improve the performance and deeper models could even perform worse.                      approach chooses the top T nodes with the highest normalized visit
This is mainly because more layers could also propagate the noisy in-                    counts.
formation from an exponentially increasing number of expanded neigh-
borhood members. It also causes the over smoothing problem because                       3.4.2. Layer sampling
nodes tend to have similar representations after the aggregation opera-                      Instead of sampling neighbors for each node, layer sampling retains a
tion when models go deeper. So that many methods try to add “skip                        small set of nodes for aggregation in each layer to control the expansion
connections” to make GNN models deeper. In this subsection we intro-                     factor. FastGCN (Chen et al., 2018b) directly samples the receptive ﬁeld
duce three kinds of instantiations of skip connections.                                  for each layer. It uses importance sampling, where the important nodes
    Highway GCN. Rahimi et al. (2018) propose a Highway GCN which                        are more likely to be sampled.
uses layer-wise gates similar to highway networks (Zilly et al., 2016). The                  In contrast to ﬁxed sampling methods above, Huang et al. (2018)
output of a layer is summed with its input with gating weights:                          introduce a parameterized and trainable sampler to perform layer-wise
                                                                                         sampling conditioned on the former layer. Furthermore, this adaptive
         Tðht Þ ¼ σ ðWt ht þ bt Þ;                                                       sampler could optimize the sampling importance and reduce variance
                                                                            (22)
htþ1 ¼ htþ1  Tðht Þ þ ht  ð1  Tðht Þ Þ:                                               simultaneously. LADIES (Zou et al., 2019) intends to alleviate the sparsity
                                                                                         issue in layer-wise sampling by generating samples from the union of
    By adding the highway gates, the performance peaks at 4 layers in a
                                                                                         neighbors of the nodes.
speciﬁc problem discussed in (Rahimi et al., 2018). The column network
(CLN) (Pham et al., 2017) also utilizes the highway network. But it has
                                                                                         3.4.3. Subgraph sampling
different functions to compute the gating weights.
                                                                                             Rather than sampling nodes and edges which builds upon the full
    JKN. Xu et al. (2018) study properties and limitations of neighbor-
                                                                                         graph, a fundamentally different way is to sample multiple subgraphs
hood aggregation schemes. They propose the jump knowledge network
                                                                                         and restrict the neighborhood search within these subgraphs. Clus-
(JKN) which could learn adaptive and structure-aware representations.
                                                                                         terGCN (Chiang et al., 2019) samples subgraphs by graph clustering al-
JKN selects from all of the intermediate representations (which “jump” to
                                                                                         gorithms, while GraphSAINT (Zeng et al., 2020) directly samples nodes
the last layer) for each node at the last layer, which makes the model
                                                                                         or edges to generate a subgraph.
adapt the effective neighborhood size for each node as needed. Xu et al.
(2018) use three approaches of concatenation, max-pooling and

                                                                                    65
J. Zhou et al.                                                                                                                             AI Open 1 (2020) 57–81


                                                Fig. 4. An overview of variants considering graph type and scale.


3.5. Pooling modules                                                                  coarsening algorithms. Spectral clustering algorithms are ﬁrstly used but
                                                                                      they are inefﬁcient because of the eigendecomposition step. Graclus
    In the area of computer vision, a convolutional layer is usually followed         (Dhillon et al., 2007) provides a faster way to cluster nodes and it is
by a pooling layer to get more general features. Complicated and large-scale          applied as a pooling module. For example, ChebNet and MoNet use
graphs usually carry rich hierarchical structures which are of great impor-           Graclus to merge node pairs and further add additional nodes to make
tance for node-level and graph-level classiﬁcation tasks. Similar to these            sure the pooling procedure forms a balanced binary tree.
pooling layers, a lot of work focuses on designing hierarchical pooling layers           ECC. Edge-Conditioned Convolution (ECC) (Simonovsky and Komo-
on graphs. In this section, we introduce two kinds of pooling modules: direct         dakis, 2017) designs its pooling module with recursively downsampling
pooling modules and hierarchical pooling modules.                                     operation. The downsampling method is based on splitting the graph into
                                                                                      two components by the sign of the largest eigenvector of the Laplacian.
3.5.1. Direct pooling modules                                                            DiffPool. DiffPool (Ying et al., 2018b) uses a learnable hierarchical
    Direct pooling modules learn graph-level representations directly                 clustering module by training an assignment matrix St in each layer:
from nodes with different node selection strategies. These modules are                                                
also called readout functions in some variants.                                       St ¼ softmax GNNt;pool ðAt ; Ht Þ ;
                                                                                               tþ1   t T t t                                                (24)
    Simple Node Pooling. Simple node pooling methods are used by several                     A ¼ ðS Þ A S ;
models. In these models, node-wise max/mean/sum/attention opera-
tions are applied on node features to get a global graph representation.              where Ht is the node feature matrix and At is coarsened adjacency matrix
    Set2set. MPNN uses the Set2set method (Vinyals et al., 2015a) as the              of layer t. St denotes the probabilities that a node in layer t can be
readout function to get graph representations. Set2set is designed to deal            assigned to a coarser node in layer t þ 1.
with the unordered set T ¼ fðhTv ; xv Þg and uses a LSTM-based method to                  gPool. gPool (Gao and Ji, 2019) uses a project vector to learn pro-
produce an order invariant representation after a prediﬁned number of                 jection scores for each node and select nodes with top-k scores.
steps.                                                                                Compared to DiffPool, it uses a vector instead of a matrix at each layer,
    SortPooling. SortPooling (Zhang et al., 2018e) ﬁrst sorts the node                thus it reduces the storage complexity. But the projection procedure does
embeddings according to the structural roles of the nodes and then the                not consider the graph structure.
sorted embeddings are fed into CNNs to get the representation.                            EigenPooling. EigenPooling (Ma et al., 2019a) is designed to use the
                                                                                      node features and local structure jointly. It uses the local graph Fourier
3.5.2. Hierarchical pooling modules                                                   transform to extract subgraph information and suffers from the in-
   The methods mentioned before directly learn graph representations                  efﬁciency of graph eigendecomposition.
from nodes and they do not investigate the hierarchical property of the                   SAGPool. SAGPool (Lee et al., 2019) is also proposed to use features
graph structure. Next we will talk about methods that follow a hierar-                and topology jointly to learn graph representations. It uses a
chical pooling pattern and learn graph representations by layers.                     self-attention based method with a reasonable time and space
   Graph Coarsening. Early methods are usually based on graph                         complexity.


                                                                                 66
J. Zhou et al.                                                                                                                             AI Open 1 (2020) 57–81


4. Variants considering graph type and scale                                      4.2.2. Edge-based methods
                                                                                      There are also works which don’t utilize meta-paths. These works
    In the above sections, we assume the graph to be the simplest format.         typically use different functions in terms of sampling, aggregation, etc.
However, many graphs in the real world are complex. In this subsection,           for different kinds of neighbors and edges. HetGNN (Zhang et al., 2019b)
we will introduce the approaches which attempt to address the chal-               addresses the challenge by directly treating neighbors of different types
lenges of complex graph types. An overview of these variants is shown in          differently in sampling, feature encoding and aggregation steps. HGT (Hu
Fig. 4.                                                                           et al., 2020a) deﬁnes the meta-relation to be the type of two neighboring
                                                                                  nodes and their link 〈φðvi Þ; ψ ðeij Þ; φðvj Þ〉. It assigns different attention
                                                                                  weight matrices to different meta-relations, empowering the model to
4.1. Directed graphs                                                              take type information into consideration.

    The ﬁrst type is the directed graphs. Directed edges usually contain          4.2.3. Methods for relational graphs
more information than undirected edges. For example, in a knowledge                   The edge of some graphs may contain more information than the type, or
graph where a head entity is the parent class of a tail entity, the edge          the quantity of types may be too large, exerting difﬁculties to applying the
direction offers information about the partial order. Instead of simply           meta-path or meta-relation based methods. We refer to this kind of graphs as
adopting an asymmetric adjacency matrix in the convolution operator,              relational graphs (Schlichtkrull et al., 2018), To handle the relational
we can model the forward and reverse directions of an edge differently.           graphs, G2S (Beck et al., 2018) converts the original graph to a bipartite
DGP (Kampffmeyer et al., 2019) uses two kinds of weight matrices Wp               graph where the original edges also become nodes and one original edge is
and Wc for the convolution in forward and reverse directions.                     split into two new edges which means there are two new edges between the
                                                                                  edge node and begin/end nodes. After this transformation, it uses a Gated
                                                                                  Graph Neural Network followed by a Recurrent Neural Network to convert
4.2. Heterogeneous graphs                                                         graphs with edge information into sentences. The aggregation function of
                                                                                  GGNN takes both the hidden representations of nodes and the relations as
   The second variant of graphs is heterogeneous graphs, where the                the input. As another approach, R-GCN (Schlichtkrull et al., 2018) doesn’t
nodes and edges are multi-typed or multi-modal. More speciﬁcally, in a            require to convert the original graph format. It assigns different weight
heterogeneous graph fV; E; φ; ψ g, each node vi is associated with a type         matrices for the propagation on different kinds of edges. However, When
φðvi Þ and each edge ej with a type ψ ðej Þ.                                      the number of relations is very large, the number of parameters in the model
                                                                                  explodes. Therefore, it introduces two kinds of regularizations to reduce the
4.2.1. Meta-path-based methods                                                    number of parameters for modeling amounts of relations: basis- and
   Most approaches toward this graph type utilize the concept of meta-            block-diagonal-decomposition. With the basis decomposition, each Wr is
path. Meta-path is a path scheme which determines the type of node in             deﬁned as follows:
                                 ψ1   ψ2    ψL
each position of the path, e.g. φ1 →φ2 →φ3 ⋯→φLþ1 , where L is the length
                                                                                         X
                                                                                         B
of the meta-path. In the training process, the meta-paths are instantiated        Wr ¼         arb Vb:                                                      (25)
as node sequences. By connecting the two end nodes of a meta-path in-                    b¼1
stances, the meta-path captures the similarity of two nodes which may
not be directly connected. Consequently, one heterogeneous graph can              Here each Wr is a linear combination of basis transformations Vb 2
be reduced to several homogeneous graphs, on which graph learning                 Rdin dout with coefﬁcients arb . In the block-diagonal decomposition, R-
algorithms can be applied. In early work, meta-path based similarity              GCN deﬁnes each Wr through the direct sum over a set of low-
search is investigated (Sun et al., 2011). Recently, more GNN models              dimensional matrices, which need more parameters than the ﬁrst one.
which utilize the meta-path are proposed. HAN (Wang et al., 2019a) ﬁrst
performs graph attention on the meta-path-based neighbors under each              4.2.4. Methods for multiplex graphs
meta-path and then uses a semantic attention over output embeddings of                In more complex scenarios, a pair of nodes in a graph can be associ-
nodes under all meta-path schemes to generate the ﬁnal representation of          ated with multiple edges of different types. By viewing under different
nodes. MAGNN (Fu et al., 2020) proposes to take the intermediate nodes            types of edges, the graph can form multiple layers, in which each layer
in a meta-path into consideration. It ﬁrst aggregates the information             represents one type of relation. Therefore, multiplex graph can also be
along the meta-path using a neural module and then performs attention             referred to as multi-view graph (multi-dimensional graph). For example,
over different meta-path instances associated with a node and ﬁnally              in YouTube, there can be three different relations between two users:
performs attention over different meta-path schemes. GTN (Yun et al.,             sharing, subscription, comment. Edge types are not assumed independent
2019) proposes a novel graph transformer layer which identiﬁes new                with each other, therefore simply splitting the graph into subgraphs with
connections between unconnected nodes while learning representations              one type of edges might not be an optimal solution. mGCN (Ma et al.,
of nodes. The learned new connections can connect nodes which are                 2019b) introduces general representations and dimension-speciﬁc rep-
serveral hops away from each other but are closely related, which                 resentations for nodes in each layer of GNN. The dimension-speciﬁc
function as the meta-paths.                                                       representations are projected from general representations using


                                                  Fig. 5. An overview of methods with unsupervised loss.

                                                                             67
J. Zhou et al.                                                                                                                            AI Open 1 (2020) 57–81


different projection matrices and then aggregated to form the next layer’s         to process large-scale graphs. Besides sampling techniques, there are also
general representations.                                                           other methods for the scaling problem. Leveraging approximate
                                                                                   personalized PageRank, methods proposed by Klicpera et al. (2019) and
4.3. Dynamic graphs                                                                Bojchevski et al. (2020) avoid calculating high-order propagation
                                                                                   matrices. Rossi et al. (2020) propose a method to precompute graph
    Another variant of graphs is dynamic graphs, in which the graph                convolutional ﬁlters of different sizes for efﬁcient training and inference.
structure, e.g. the existence of edges and nodes, keeps changing over              PageRank-based models squeeze multiple GCN layers into one single
time. To model the graph structured data together with the time series             propagation layer to mitigate the “neighbor explosion” issue, hence are
data, DCRNN (Li et al., 2018b) and STGCN (Yu et al., 2018) ﬁrst collect            highly scalable and efﬁcient.
spatial information by GNNs, then feed the outputs into a sequence
model like sequence-to-sequence models or RNNs. Differently,                       5. Variants for different training settings
Structural-RNN (Jain et al., 2016) and ST-GCN (Yan et al., 2018) collect
spatial and temporal messages at the same time. They extend static graph               In this section, we introduce variants for different training settings.
structure with temporal connections so they can apply traditional GNNs             For supervised and semi-supervised settings, labels are provided so that
on the extended graphs. Similarly, DGNN (Manessi et al., 2020) feeds the           loss functions are easy to design for these labeled samples. For unsu-
output embeddings of each node from the GCN into separate LSTMs. The               pervised settings, there are no labeled samples so that loss functions
weights of LSTMs are shared between each node. On the other hand,                  should depend on the information provided by the graph itself, such as
EvolveGCN (Pareja et al., 2020) argues that directly modeling dynamics             input features or the graph topology. In this section, we mainly introduce
of the node representation will hamper the model’s performance on                  variants for unsupervised training, which are usually based on the ideas
graphs where node set keeps changing. Therefore, instead of treating               of auto-encoders or contrastive learning. An overview of the methods we
node features as the input to RNN, it feeds the weights of the GCN into            mention is shown in Fig. 5.
the RNN to capture the intrinsic dynamics of the graph interactions.
Recently, a survey (Huang et al., 2020) classiﬁes the dynamic networks             5.1. Graph auto-encoders
into several categories based on the link duration, and groups the existing
models into these categories according to their specialization. It also                For unsupervised graph representation learning, there has been a
establishes a general framework for models of dynamic graphs and ﬁts               trend to extend auto-encoder (AE) to graph domains.
existing models into the general framework.                                            Graph Auto-Encoder (GAE) (Kipf and Welling, 2016) ﬁrst uses GCNs
                                                                                   to encode nodes in the graph. Then it uses a simple decoder to reconstruct
4.4. Other graph types                                                             the adjacency matrix and compute the loss from the similarity between
                                                                                   the original adjacency matrix and the reconstructed matrix:
   For other variants of graphs, such as hypergraphs and signed graphs,
                                                                                   H ¼ GCNðX; AÞ;
there are also some models proposed to address the challenges.                     ~ ¼ ρðHHT Þ:                                                            (28)
                                                                                   A
4.4.1. Hypergraphs                                                                     Kipf and Welling (2016) also train the GAE model in a variational
   A hypergraph can be denoted by G ¼ ðV;E;We Þ, where an edge e 2 E               manner and the model is named as the variational graph auto-encoder
connects two or more vertices and is assigned a weight w 2 We . The                (VGAE).
adjacency matrix of a hypergraph can be represented in a jVj jEj matrix               Adversarially Regularized Graph Auto-encoder (ARGA) (Pan et al.,
L:                                                                                 2018) employs generative adversarial networks (GANs) to regularize a
                                                                                   GCN-based graph auto-encoder, which could learn more robust node
             1;   if v 2 e                                                         representations.
Lv;e ¼                      :                                         (26)
             0;   if v 62 e                                                            Instead of recovering the adjacency matrix, Wang et al. (2017), Park
   HGNN (Feng et al., 2019) proposes hypergraph convolution to pro-                et al. (2019) try to reconstruct the feature matrix. MGAE (Wang et al.,
cess these high order interaction between nodes:                                   2017) utilizes marginalized denoising auto-encoder to get robust node
                                                                                   representation. To build a symmetric graph auto-encoder, GALA (Park
         1                1
H ¼ D      1 T 2
     v LWe De L Dv XW;
      2                                                               (27)         et al., 2019) proposes Laplacian sharpening, the inverse operation of
                                                                                   Laplacian smoothing, to decode hidden states. This mechanism alleviates
where the Dv ; We ; De ; X are the node degree matrix, edge weight matrix,         the oversmoothing issue in GNN training.
edge degree matrix and node feature matrix respectively. W is the                      Different from above, AGE (Cui et al., 2020) states that the recovering
learnable parameters. This formula is derived by approximating the                 losses are not compatible with downstream tasks. Therefore, they apply
hypergraph Laplacian using truncated Chebyshev polynomials.                        adaptive learning for the measurement of pairwise node similarity and
                                                                                   achieve state-of-the-art performance on node clustering and link
4.4.2. Signed graphs                                                               prediction.
    Signed graphs are the graphs with signed edges, i.e. an edge can be
either positive or negative. Instead of simply treating the negative edges         5.2. Contrastive learning
as the absent edges or another type of edges, SGCN (Derr et al., 2018)
utilizes balance theory to capture the interactions between positive edges             Besides graph auto-encoders, contrastive learning paves another way
and negative edges. Intuitively, balance theory suggests that the friend           for unsupervised graph representation learning. Deep Graph Infomax
(positive edge) of my friend is also my friend and the enemy (negative             (DGI) (Velickovic et al., 2019) maximizes mutual information between
edge) of my enemy is my friend. Therefore it provides theoretical foun-            node representations and graph representations. Infograph (Sun et al.,
dation for SGCN to model the interactions between positive edges and               2020) aims to learn graph representations by mutual information maxi-
negative edges.                                                                    mization between graph-level representations and the substructure-level
                                                                                   representations of different scales including nodes, edges and triangles.
4.5. Large graphs                                                                  Multi-view (Hassani and Khasahmadi, 2020) contrasts representations
                                                                                   from ﬁrst-order adjacency matrix and graph diffusion, achieves
    As we mentioned in Section 3.4, sampling operators are usually used            state-of-the-art performances on multiple graph learning tasks.

                                                                              68
J. Zhou et al.                                                                                                                         AI Open 1 (2020) 57–81


6. A design example of GNN                                                           self-supervised graph generation task is designed to learn node em-
                                                                                     beddings. In the ﬁnetuning step, the model is ﬁnetuned based on the
   In this section, we give an existing GNN model to illustrated the                 training data of each task, so that the supervised loss of each task is
design process. Taking the task of heterogeneous graph pretraining as an             applied.
example, we use GPT-GNN (Hu et al., 2020b) as the model to illustrate             4. Build model using computational modules. Finally the model is built
the design process.                                                                  with computational modules. For the propagation module, the au-
                                                                                     thors use a convolution operator HGT (Hu et al., 2020a) that we
1. Find graph structure. The paper focuses on applications on the aca-               mentioned before. HGT incorporates the types of nodes and edges
   demic knowledge graph and the recommendation system. In the ac-                   into the propagation step of the model and the skip connection is also
   ademic knowledge graph, the graph structure is explicit. In                       added in the architecture. For the sampling module, a specially
   recommendation systems, users, items and reviews can be regarded as               designed sampling method HGSampling (Hu et al., 2020a) is applied,
   nodes and the interactions among them can be regarded as edges, so                which is a heterogeneous version of LADIES (Zou et al., 2019). As the
   the graph structure is also easy to construct.                                    model focuses on learning node representations, the pooling module
2. Specify graph type and scale. The tasks focus on heterogeneous graphs,            is not needed. The HGT layer are stacked multiple layers to learn
   so that types of nodes and edges should be considered and incorpo-                better node embeddings.
   rated in the ﬁnal model. As the academic graph and the recommen-
   dation graph contain millions of nodes, so that the model should               7. Analyses of GNNs
   further consider the efﬁciency problem. In conclusion, the model
   should focus on large-scale heterogeneous graphs.                              7.1. Theoretical aspect
3. Design loss function. As downstream tasks in (Hu et al., 2020b) are all
   node-level tasks (e.g. Paper-Field prediction in the academic graph),             In this section, we summarize the papers about the theoretic foun-
   so that the model should learn node representations in the pretraining         dations and explanations of graph neural networks from various
   step. In the pretraining step, no labeled data is available, so that a         perspectives.


                                            Fig. 6. Application scenarios. (Icons made by Freepik from Flaticon)

                                                                             69
J. Zhou et al.                                                                                                                                                    AI Open 1 (2020) 57–81


Table 3
Applications of graph neural networks.
  Area                     Application                    References

  Graph Mining             Graph Matching                 (Riba et al., 2018; Li et al., 2019b)
                           Graph Clustering               (Zhang et al., 2019c; Ying et al., 2018b; Tsitsulin et al., 2020)

  Physics                  Physical Systems Modeling      (Battaglia et al., 2016; Sukhbaatar Ferguset al., 2016; Watters et al., 2017; Hoshen, 2017; Kipf et al., 2018; Sanchez et al.,
                                                          2018)

  Chemistry                Molecular Fingerprints         (Duvenaud et al., 2015; Kearnes et al., 2016)
                           Chemical Reaction              Do et al. (2019)
                           Prediction

  Biology                  Protein Interface Prediction   Fout et al. (2017)
                           Side Effects Prediction        Zitnik et al. (2018)
                           Disease Classiﬁcation          Rhee et al. (2018)

  Knowledge Graph          KB Completion                  (Hamaguchi et al., 2017; Schlichtkrull et al., 2018; Shang et al., 2019)
                           KG Alignment                   (Wang et al., 2018b; Zhang et al., 2019d; Xu et al., 2019c)

  Generation               Graph Generation               (Shchur et al., 2018b; Nowak et al., 2018; Ma et al., 2018; You et al., 2018a, 2018b; De Cao and Kipf, 2018; Li et al.,
                                                          2018d; Shi et al., 2020; Liu et al., 2019; Grover et al., 2019)

  Combinatorial            Combinatorial Optimization     (Khalil et al., 2017; Nowak et al., 2018; Li et al., 2018e; Kool et al., 2019; Bello et al., 2017; Vinyals et al., 2015b; Sutton
    Optimization                                          and Barto, 2018; Dai et al., 2016; Gasse et al., 2019; Zheng et al., 2020a; Selsam et al., 2019; Sato et al., 2019)

  Trafﬁc Network           Trafﬁc State Prediction        (Cui et al., 2018b; Yu et al., 2018; Zheng et al., 2020b; Guo et al., 2019)

  Recommendation           User-item Interaction          (van den Berg et al., 2017; Ying et al., 2018a)
    Systems                Prediction
                           Social Recommendation          (Wu et al., 2019c; Fan et al., 2019)

  Others (Structural)      Stock Market                   (Matsunaga et al., 2019; Yang et al., 2019; Chen et al., 2018c; Li et al., 2020; Kim et al., 2019)
                           Software Deﬁned Networks       Rusek et al. (2019)
                           AMR Graph to Text              (Song et al., 2018a; Beck et al., 2018)

  Text                     Text Classiﬁcation             (Peng et al., 2018; Yao et al., 2019; Zhang et al., 2018d; Tai et al., 2015)
                           Sequence Labeling              (Zhang et al., 2018d; Marcheggiani and Titov, 2017)
                           Neural Machine Translation     (Bastings et al., 2017; Marcheggiani et al., 2018; Beck et al., 2018)
                           Relation Extraction            (Miwa and Bansal, 2016; Peng et al., 2017; Song et al., 2018b; Zhang et al., 2018f)
                           Event Extraction               (Nguyen and Grishman, 2018; Liu et al., 2018)
                           Fact Veriﬁcation               (Zhou et al., 2019; Liu et al., 2020; Zhong et al., 2020)
                           Question Answering             (Song et al., 2018c; De Cao et al., 2019; Qiu et al., 2019; Tu et al., 2019; Ding et al., 2019)
                           Relational Reasoning           (Santoro et al., 2017; Palm et al., 2018; Battaglia et al., 2016)

  Image                    Social Relationship            Wang et al. (2018c)
                           Understanding
                           Image Classiﬁcation            (Garcia and Bruna, 2018; Wang et al., 2018d; Lee et al., 2018b; Kampffmeyer et al., 2019; Marino et al., 2017)
                           Visual Question Answering      (Teney et al., 2017; Wang et al., 2018c; Narasimhan et al., 2018)
                           Object Detection               (Hu et al., 2018; Gu et al., 2018)
                           Interaction Detection          (Qi et al., 2018; Jain et al., 2016)
                           Region Classiﬁcation           Chen et al. (2018d)
                           Semantic Segmentation          (Liang et al., 2016, 2017; Landrieu and Simonovsky, 2018; Wang et al., 2018e; Qi et al., 2017b)

  Other (Non-structural)   Program Veriﬁcation            (Allamanis et al., 2018; Li et al., 2016)


7.1.1. Graph signal processing                                                               that graph convolution is mainly a denoising process for input features, the
    From the spectral perspective of view, GCNs perform convolution                          model performances heavily depend on the amount of noises in the feature
operation on the input features in the spectral domain, which follows                        matrix. To alleviate the over-smoothing issue, Chen et al. (2020b) present
graph signal processing in theory.                                                           two metrics for measuring the smoothness of node representations and the
    There exists several works analyzing GNNs from graph signal pro-                         over-smoothness of GNN models. The authors conclude that the
cessing. Li et al. (2018c) ﬁrst address the graph convolution in graph                       information-to-noise ratio is the key factor for over-smoothing.
neural networks is actually Laplacian smoothing, which smooths the
feature matrix so that nearby nodes have similar hidden representations.                     7.1.2. Generalization
Laplacian smoothing reﬂects the homophily assumption that nearby                                 The generalization ability of GNNs have also received attentions
nodes are supposed to be similar. The Laplacian matrix serves as a                           recently. Scarselli et al. (2018) prove the VC-dimensions for a limited
low-pass ﬁlter for the input features. SGC (Wu et al., 2019b) further                        class of GNNs. Garg et al. (2020) further give much tighter generalization
removes the weight matrices and nonlinearties between layers, showing                        bounds based on Rademacher bounds for neural networks.
that the low-pass ﬁlter is the reason why GNNs work.                                             Verma and Zhang (2019) analyze the stability and generalization
    Following the idea of low-pass ﬁltering, Zhang et al. (2019c), Cui et al.                properties of single-layer GNNs with different convolutional ﬁlters. The
(2020), NT and Maehara (Nt and Maehara, 2019), Chen et al. (2020b)                           authors conclude that the stability of GNNs depends on the largest
analyze different ﬁlters and provide new insights. To achieve low-pass                       eigenvalue of the ﬁlters. Knyazev et al. (2019) focus on the generalization
ﬁltering for all the eigenvalues, AGC (Zhang et al., 2019c) designs a                        ability of attention mechanism in GNNs. Their conclusion shows that
graph ﬁlter I  12 L according to the frequency response function. AGE (Cui                  attention helps GNNs generalize to larger and noisy graphs.
et al., 2020) further demonstrates that ﬁlter with I  λmax
                                                        1
                                                            L could get better
results, where λmax is the maximum eigenvalue of the Laplacian matrix.                       7.1.3. Expressivity
Despite linear ﬁlters, GraphHeat (Xu et al., 2019a) leverages heat kernels for                  On the expressivity of GNNs, Xu et al. (2019b), Morris et al. (2019)
better low-pass properties. NT and Maehara (Nt and Maehara, 2019) state                      show that GCNs and GraphSAGE are less discriminative than


                                                                                        70
J. Zhou et al.                                                                                                                             AI Open 1 (2020) 57–81


Weisfeiler-Leman (WL) test, an algorithm for graph isomorphism testing.            lead to dramatically different rankings of models. Also, simple models
Xu et al. (2019a) also propose GINs for more expressive GNNs. Going                could outperform complicated ones under proper settings. Errica et al.
beyond WL test, Barcelo et al. (2019) discuss if GNNs are expressible for         (2020) review several graph classiﬁcation models and point out that they
FOC2 , a fragment of ﬁrst order logic. The authors ﬁnd that existing GNNs          are compared inproperly. Based on rigorous evaluation, structural in-
can hardly ﬁt the logic. For learning graph topologic structures, Garg             formation turns up to not be fully exploited for graph classiﬁcation. You
et al. (2020) prove that locally dependent GNN variants are not capable            et al. (2020) discuss the architectural designs of GNN models, such as the
to learn global graph properties, including diameters, biggest/smallest            number of layers and the aggregation function. By a huge amount of
cycles, or motifs.                                                                 experiments, this work provides comprehensive guidelines for GNN
    Loukas (2020) and Dehmamy et al. (2019) argue that existing works              designation over various tasks.
only consider the expressivity when GNNs have inﬁnite layers and units.
Their work investigates the representation power of GNNs with ﬁnite                7.2.2. Benchmarks
depth and width. Oono and Suzuki (2020) discuss the asymptotic be-                     High-quality and large-scale benchmark datasets such as ImageNet
haviors of GNNs as the model deepens and model them as dynamic                     are signiﬁcant in machine learning research. However in graph learning,
systems.                                                                           widely-adopted benchmarks are problematic. For example, most node
                                                                                   classiﬁcation datasets contain only 3000 to 20,000 nodes, which are
7.1.4. Invariance                                                                  small compared with real-world graphs. Furthermore, the experimental
   As there are no node orders in graphs, the output embeddings of                 protocols across studies are not uniﬁed, which is hazardous to the liter-
GNNs are supposed to be permutation-invariant or equivariant to the                ature. To mitigate this issue, Dwivedi et al. (2020), Hu et al. (2020d)
input features. Maron et al. (2019a) characterize permutation-invariant            provide scalable and reliable benchmarks for graph learning. Dwivedi
or equivariant linear layers to build invariant GNNs. Maron et al.                 et al. (2020) build medium-scale benchmark datasets in multiple do-
(2019b) further prove the result that the universal invariant GNNs can be          mains and tasks, while OGB (Hu et al., 2020d) offers large-scale datasets.
obtained with higher-order tensorization. Keriven and Peyre (2019)                Furthermore, both works evaluate current GNN models and provide
provide an alternative proof and extend this conclusion to the equivariant         leaderboards for further comparison.
case.    Chen     et   al.    (2019)     build   connections     between
permutation-invariance and graph isomorphism testing. To prove their               8. Applications
equivalence, Chen et al. (2019) leverage sigma-algebra to describe the
expressivity of GNNs.                                                                   Graph neural networks have been explored in a wide range of do-
                                                                                   mains across supervised, semi-supervised, unsupervised and reinforce-
7.1.5. Transferability                                                             ment learning settings. In this section, we generally group the
    A deterministic characteristic of GNNs is that the parameterization is         applications in two scenarios: (1) Structural scenarios where the data has
untied with graphs, which suggests the ability to transfer across graphs           explicit relational structure. These scenarios, on the one hand, emerge
(so-called transferability) with performance guarantees. Levie et al.              from scientiﬁc researches, such as graph mining, modeling physical
(2019) investigate the transferability of spectral graph ﬁlters, showing           systems and chemical systems. On the other hand, they rise from in-
that such ﬁlters are able to transfer on graphs in the same domain. Ruiz           dustrial applications such as knowledge graphs, trafﬁc networks and
et al. (2020) analyze GNN behaviour on graphons. Graphon refers to the             recommendation systems. (2) Non-structural scenarios where the rela-
limit of a sequence of graphs, which can also be seen as a generator for           tional structure is implicit or absent. These scenarios generally include
dense graphs. The authors conclude that GNNs are transferable across               image (computer vision) and text (natural language processing), which
graphs obtained deterministically from the same graphon with different             are two of the most actively developing branches of AI researches. A
sizes.                                                                             simple illustration of these applications is in Fig. 6. Note that we only list
                                                                                   several representative applications instead of providing an exhaustive
7.1.6. Label efﬁciency                                                             list. The summary of the applications could be found in Table 3.
    (Semi-) Supervised learning for GNNs needs a considerable amount of
labeled data to achieve a satisfying performance. Improving the label              8.1. Structural scenarios
efﬁciency has been studied in the perspective of active learning, in which
informative nodes are actively selected to be labeled by an oracle to train            In the following subsections, we will introduce GNNs’ applications in
the GNNs. Cai et al. (2017), Gao et al. (2018b), Hu et al. (2020c)                 structural scenarios, where the data are naturally performed in the graph
demonstrate that by selecting the informative nodes such as the                    structure.
high-degree nodes and uncertain nodes, the labeling efﬁciency can be
dramatically improved.                                                             8.1.1. Graph mining
                                                                                       The ﬁrst application is to solve the basic tasks in graph mining.
7.2. Empirical aspect                                                              Generally, graph mining algorithms are used to identify useful structures
                                                                                   for downstream tasks. Traditional graph mining challenges include
   Besides theoretical analysis, empirical studies of GNNs are also                frequent sub-graph mining, graph matching, graph classiﬁcation, graph
required for better comparison and evaluation. Here we include several             clustering, etc. Although with deep learning, some downstream tasks can
empirical studies for GNN evaluation and benchmarks.                               be directly solved without graph mining as an intermediate step, the
                                                                                   basic challenges are worth being studied in the GNNs’ perspective.
7.2.1. Evaluation                                                                      Graph Matching. The ﬁrst challenge is graph matching. Traditional
    Evaluating machine learning models is an essential step in research.           methods for graph matching usually suffer from high computational
Concerns about experimental reproducibility and replicability have been            complexity. The emergence of GNNs allows researchers to capture the
raised over the years. Whether and to what extent do GNN models work?              structure of graphs using neural networks, thus offering another solution
Which parts of the models contribute to the ﬁnal performance? To                   to the problem. Riba et al. (2018) propose a siamese MPNN model to
investigate such fundamental questions, studies about fair evaluation              learn the graph editing distance. The siamese framework is two parallel
strategies are urgently needed.                                                    MPNNs with the same structure and weight sharing, The training
    On semi-supervised node classiﬁcation task, Shchur et al. (2018a)              objective is to embed a pair of graphs with small editing distance into
explore how GNN models perform under same training strategies and                  close latent space. Li et al. (2019b) design similar methods while ex-
hyperparameter tune. Their works concludes that different dataset splits           periments on more real-world scenario such as similarity search in

                                                                              71
J. Zhou et al.                                                                                                                                AI Open 1 (2020) 57–81


control ﬂow graph.                                                                    Policy Network (Do et al., 2019) encodes the input molecules and gen-
    Graph Clustering. Graph clustering is to group the vertices of a graph            erates an intermediate graph with a node pair prediction network and a
into clusters based on the graph structure and/or node attributes. Various            policy network.
works (Zhang et al., 2019c) in node representation learning are devel-                    Protein Interface Prediction. Proteins interact with each other
oped and the representation of nodes can be passed to traditional clus-               using the interface, which is formed by the amino acid residues from each
tering algorithms. Apart of learning node embeddings, graph pooling                   participating protein. The protein interface prediction task is to deter-
(Ying et al., 2018b) can be seen as a kind of clustering. More recently,              mine whether particular residues constitute part of a protein. Generally,
Tsitsulin et al. (2020) directly target at the clustering task. They study the        the prediction for a single residue depends on other neighboring resi-
desirable property of a good graph clustering method and propose to                   dues. By letting the residues to be nodes, the proteins can be represented
optimize the spectral modularity, which is a remarkably useful graph                  as graphs, which can leverage the GNN-based machine learning algo-
clustering metric.                                                                    rithms. Fout et al. (2017) propose a GCN-based method to learn ligand
                                                                                      and receptor protein residue representation and to merge them for
8.1.2. Physics                                                                        pair-wise classiﬁcation. MR-GNN (Xu et al., 2019d) introduces a
    Modeling real-world physical systems is one of the most fundamental               multi-resolution approach to extract and summarize local and global
aspects of understanding human intelligence. A physical system can be                 features for better prediction.
modeled as the objects in the system and pair-wise interactions between                   Biomedical Engineering. With Protein-Protein Interaction Network,
objects. Simulation in the physical system requires the model to learn the            Rhee et al. (2018) leverage graph convolution and relation network for
law of the system and make predictions about the next state of the sys-               breast cancer subtype classiﬁcation. Zitnik et al. (2018) also suggest a
tem. By modeling the objects as nodes and pair-wise interactions as                   GCN-based model for polypharmacy side effects prediction. Their work
edges, the systems can be simpliﬁed as graphs. For example, in particle               models the drug and protein interaction network and separately deals
systems, particles can interact with each other via multiple interactions,            with edges in different types.
including collision (Hoshen, 2017), spring connection, electromagnetic
force (Kipf et al., 2018), etc., where particles are seen as nodes and in-            8.1.4. Knowledge graph
teractions are seen as edges. Another example is the robotic system,                      The knowledge graph (KG) represents a collection of real-world en-
which is formed by multiple bodies (e.g., arms, legs) connected with                  tities and the relational facts between pairs of the entities. It has wide
joints. The bodies and joints can be seen as nodes and edges, respectively.           application, such as question answering, information retrieval and
The model needs to infer the next state of the bodies based on the current            knowledge guided generation. Tasks on KGs include learning low-
state of the system and the principles of physics.                                    dimensional embeddings which contain rich semantics for the entities
    Before the advent of graph neural networks, works process the graph               and relations, predicting the missing links between entities, and multi-
representation of the systems using the available neural blocks. Interac-             hop reasoning over the knowledge graph. One line of research treats
tion Networks (Battaglia et al., 2016) utilizes MLP to encode the inci-               the graph as a collection of triples, and proposes various kinds of loss
dence matrices of the graph. CommNet (Sukhbaatar Ferguset al., 2016)                  functions to distinguish the correct triples and false triples (Bordes et al.,
performs nodes updates using the nodes’ previous representations and                  2013). The other line leverages the graph nature of KG, and uses
the average of all nodes’ previous representations. VAIN (Hoshen, 2017)               GNN-based methods for various tasks. When treated as a graph, KG can
further introduces the attention mechanism. VIN (Watters et al., 2017)                be seen as a heterogeneous graph. However, unlike other heterogeneous
combines CNNs, RNNs and IN (Battaglia et al., 2016).                                  graphs such as social networks, the logical relations are of more impor-
    The emergence of GNNs let us perform GNN-based reasoning about                    tance than the pure graph structure.
objects, relations, and physics in a simpliﬁed but effective way. NRI (Kipf               R-GCN (Schlichtkrull et al., 2018) is the ﬁrst work to incorporate
et al., 2018) takes the trajectory of objects as input and infers an explicit         GNNs for knowledge graph embedding. To deal with various relations,
interaction graph, and learns a dynamic model simultaneously. The                     R-GCN proposes relation-speciﬁc transformation in the message passing
interaction graphs are learned from former trajectories, and trajectory               steps. Structure-Aware Convolutional Network (Shang et al., 2019)
predictions are generated from decoding the interaction graphs.                       combines a GCN encoder and a CNN decoder together for better
    Sanchez et al. (2018) propose a Graph Network-based model to                      knowledge representations.
encode the graph formed by bodies and joints of a robotic system. They                    A more challenging setting is knowledge base completion for out-of-
further learn the policy of stably controlling the system by combining                knowledge-base (OOKB) entities. The OOKB entities are unseen in the
GNs with Reinforcement learning.                                                      training set, but directly connect to the observed entities in the training
                                                                                      set. The embeddings of OOKB entities can be aggregated from the
8.1.3. Chemistry and biology                                                          observed entities. Hamaguchi et al. (2017) use GNNs to solve the prob-
     Molecular Fingerprints. Molecular ﬁngerprints serve as a way to                  lem, which achieve satisfying performance both in the standard KBC
encode the structure of molecules. The simplest ﬁngerprint can be a one-              setting and the OOKB setting.
hot vector, where each digit represents the existence or absence of a                     Besides knowledge graph representation learning, Wang et al.
particular substructure. These ﬁngerprints can be used in molecule                    (2018b) utilize GCN to solve the cross-lingual knowledge graph align-
searching, which is a core step in computer-aided drug design. Conven-                ment problem. The model embeds entities from different languages into a
tional molecular ﬁngerprints are hand-made and ﬁxed (e.g., the one-hot                uniﬁed embedding space and aligns them based on the embedding sim-
vector). However, molecules can be naturally seen as graphs, with atoms               ilarity. To align large-scale heterogeneous knowledge graphs, OAG
being the nodes and chemical-bonds being the edges. Therefore, by                     (Zhang et al., 2019d) uses graph attention networks to model various
applying GNNs to molecular graphs, we can obtain better ﬁngerprints.                  types of entities. With representing entities as their surrounding sub-
     Duvenaud et al. (2015) propose neural graph ﬁngerprints (Neural                  graphs, Xu et al. (2019c) transfer the entity alignment problem to a graph
FPs), which calculate substructure feature vectors via GCNs and sum to                matching problem and then solve it by graph matching networks.
get overall representations. Kearnes et al. (2016) explicitly model atom
and atom pairs independently to emphasize atom interactions. It in-                   8.1.5. Generative models
troduces edge representation etuv instead of aggregation function, i.e.                   Generative models for real-world graphs have drawn signiﬁcant
         P t
htN v ¼      euv .                                                                    attention for their important applications including modeling social in-
         u2N ðvÞ                                                                      teractions, discovering new chemical structures, and constructing
    Chemical Reaction Prediction. Chemical reaction product predic-                   knowledge graphs. As deep learning methods have powerful ability to
tion is a fundamental issue in organic chemistry. Graph Transformation                learn the implicit distribution of graphs, there is a surge in neural graph

                                                                                 72
J. Zhou et al.                                                                                                                            AI Open 1 (2020) 57–81


generative models recently.                                                        problem as a bipartite graph and utilize GCN to encode it.
    NetGAN (Shchur et al., 2018b) is one of the ﬁrst work to build neural              For speciﬁc combinatorial optimization problems, Nowak et al.
graph generative model, which generates graphs via random walks. It                (2018) focus on Quadratic Assignment Problem i.e. measuring the sim-
transforms the problem of graph generation to the problem of walk                  ilarity of two graphs. The GNN based model learns node embeddings for
generation which takes the random walks from a speciﬁc graph as input              each graph independently and matches them using attention mechanism.
and trains a walk generative model using GAN architecture. While the               This method offers intriguingly good performance even in regimes where
generated graph preserves important topological properties of the orig-            standard relaxation-based techniques appear to suffer. Zheng et al.
inal graph, the number of nodes is unable to change in the generating              (2020a) use a generative graph neural network to model the
process, which is as same as the original graph. GraphRNN (You et al.,             DAG-structure learning problem, which is also a combinatorial optimi-
2018b) manages to generate the adjacency matrix of a graph by gener-               zation and NP-hard problem. NeuroSAT (Selsam et al., 2019) learns a
ating the adjacency vector of each node step by step, which can output             message passing neural network to classify the satisﬁability of SAT
networks with different numbers of nodes. Li et al. (2018d) propose a              problem. It proves that the learned model can generalize to novel dis-
model which generates edges and nodes sequentially and utilizes a graph            tributions of SAT and other problems which can be converted to SAT.
neural network to extract the hidden state of the current graph which is               Unlike previous works which try to design speciﬁc GNNs to solve
used to decide the action in the next step during the sequential generative        combinatorial problems, Sato et al. (2019) provide a theoretical analysis
process. GraphAF (Shi et al., 2020) also formulates graph generation as a          of GNN models on these problems. It establishes connections between
sequential decision process. It combines the ﬂow-based generation with             GNNs and the distributed local algorithms which is a group of classical
the autogressive model. Towards molecule generation, it also conducts              algorithms on graphs for solving these problems. Moreover, it demon-
validity check of the generated molecules using existing chemical rules            strates the optimal approximation ratios to the optimal solutions that the
after each step of generation.                                                     most powerful GNN can reach. It also proves that most of existing GNN
    Instead of generating graph sequentially, other works generate the             models cannot exceed this upper bound. Furthermore, it adds coloring to
adjacency matrix of graph at once. MolGAN (De Cao and Kipf, 2018)                  the node feature to improve the approximation ratios.
utilizes a permutation-invariant discriminator to solve the node variant
problem in the adjacency matrix. Besides, it applies a reward network for          8.1.7. Trafﬁc networks
RL-based optimization towards desired chemical properties. What’s                     Predicting trafﬁc states is a challenging task since trafﬁc networks are
more, Ma et al. (2018) propose constrained variational auto-encoders to            dynamic and have complex dependencies. Cui et al. (2018b) combine
ensure the semantic validity of generated graphs. And, GCPN (You et al.,           GNNs and LSTMs to capture both spatial and temporal dependencies.
2018a) incorporates domain-speciﬁc rules through reinforcement                     STGCN (Yu et al., 2018) constructs ST-Conv blocks with spatial and
learning. GNF (Liu et al., 2019) adapts normalizing ﬂow to the graph               temporal convolution layers, and applies residual connection with
data. Normalizing ﬂow is a kind of generative model which uses a                   bottleneck strategies. Zheng et al. (2020b), Guo et al. (2019) both
invertable mapping to transform observed data into latent vector space.            incorporate attention mechanism to better model spatial temporal
Transforming from the latent vector back into the observed data using              correlation.
the inverse matrix serves as the generating process. GNF combines
normalizing ﬂow with a permutation-invariant graph auto-encoder to                 8.1.8. Recommendation systems
take graph structured data as the input and generate new graphs at the                 User-item interaction prediction is one of the classic problems in
test time. Graphite (Grover et al., 2019) integrates GNN into variational          recommendation. By modeling the interaction as a graph, GNNs can be
auto-encoders to encode the graph structure and features into latent               utilized in this area. GC-MC (van den Berg et al., 2017) ﬁrstly applies
variables. More speciﬁcally, it uses isotropic Gaussian as the latent var-         GCN on user-item rating graphs to learn user and item embeddings. To
iables and then uses iterative reﬁnement strategy to decode from the               efﬁciently adopt GNNs in web-scale scenarios, PinSage (Ying et al.,
latent variables.                                                                  2018a) builds computational graphs with weighted sampling strategy for
                                                                                   the bipartite graph to reduce repeated computation.
8.1.6. Combinatorial optimization                                                      Social recommendation tries to incorporate user social networks to
    Combinatorial optimization problems over graphs are set of NP-hard             enhance recommendation performance. GraphRec (Fan et al., 2019)
problems which attract much attention from scientists of all ﬁelds. Some           learns user embeddings from both item side and user side. Wu et al.
speciﬁc problems like traveling salesman problem (TSP) and minimum                 (2019c) go beyond static social effects. They attempt to model homophily
spanning trees (MST) have got various heuristic solutions. Recently,               and inﬂuence effects by dual attentions.
using a deep neural network for solving such problems has been a hot-
spot, and some of the solutions further leverage graph neural network              8.1.9. Other Applications in structural scenarios
because of their graph structure.                                                      Because of the ubiquity of graph-structured data, GNNs have been
    Bello et al. (2017) ﬁrst propose a deep-learning approach to tackle            applied to a larger variety of tasks than what we have introduced above.
TSP. Their method consists of two parts: a Pointer Network (Vinyals                We list more scenarios very brieﬂy. In ﬁnancial market, GNNs are used to
et al., 2015b) for parameterizing rewards and a policy gradient (Sutton            model the interaction between different stocks to predict the future
and Barto, 2018) module for training. This work has been proved to be              trends of the stocks (Matsunaga et al., 2019; Yang et al., 2019; Chen et al.,
comparable with traditional approaches. However, Pointer Networks are              2018c; Li et al., 2020). Kim et al. (2019) also predict the market index
designed for sequential data like texts, while order-invariant encoders are        movement by formulating it as a graph classiﬁcation problem. In
more appropriate for such work.                                                    Software-Deﬁned Networks (SDN), GNNs are used to optimize the rout-
    Khalil et al. (2017), Kool et al. (2019) improve the above method by           ing performance (Rusek et al., 2019). In Abstract Meaning Representa-
including graph neural networks. The former work ﬁrst obtains the node             tion (AMR) graph to Text generation tasks, Song et al. (2018a), Beck et al.
embeddings from structure2vec (Dai et al., 2016), then feed them into a            (2018) use GNNs to encode the graph representation of the abstract
Q-learning module for making decisions. The latter one builds an                   meaning.
attention-based encoder-decoder system. By replacing reinforcement
learning module with an attention-based decoder, it is more efﬁcient for           8.2. Non-structural scenarios
training. These works achieve better performances than previous algo-
rithms, which prove the representation power of graph neural networks.                In this section we will talk about applications on non-structural sce-
More generally, Gasse et al. (2019) represent the state of a combinatorial         narios. Generally, there are two ways to apply GNNs on non-structural


                                                                              73
J. Zhou et al.                                                                                                                           AI Open 1 (2020) 57–81


scenarios: (1) Incorporate structural information from other domains to            message-passing tools between humans and objects. In region classiﬁ-
improve the performance, for example using information from knowl-                 cation (Chen et al., 2018d), GNNs perform reasoning on graphs that
edge graphs to alleviate the zero-shot problems in image tasks; (2) Infer          connects regions and classes.
or assume the relational structure in the task and then apply the model to             Semantic Segmentation. Semantic segmentation is a crucial step
solve the problems deﬁned on graphs, such as the method in (Zhang                  towards image understanding. The task here is to assign a unique label
et al., 2018d) which models text into graphs. Common non-structure                 (or category) to every single pixel in the image, which can be considered
scenarios include image, text, and programming source code (Allama-                as a dense classiﬁcation problem. However, regions in images are often
nis et al., 2018; Li et al., 2016). However, we only give detailed intro-          not grid-like and need non-local information, which leads to the failure of
duction to the ﬁrst two scenarios.                                                 traditional CNN. Several works utilize graph-structured data to handle it.
                                                                                       Liang et al. (2016) use Graph-LSTM to model long-term dependency
8.2.1. Image                                                                       together with spatial connections by building graphs in the form of
    Few(Zero)-shot Image Classiﬁcation. Image classiﬁcation is a very              distance-based superpixel map and applying LSTM to propagate neigh-
basic and important task in the ﬁeld of computer vision, which attracts            borhood information globally. Subsequent work improves it from the
much attention and has many famous datasets like ImageNet (Russa-                  perspective of encoding hierarchical information (Liang et al., 2017).
kovsky et al., 2015). Recently, zero-shot and few-shot learning become                 Furthermore, 3D semantic segmentation (RGBD semantic segmenta-
more and more popular in the ﬁeld of image classiﬁcation. In N-shot                tion) and point clouds classiﬁcation utilize more geometric information
learning, to make predictions for the test data samples in some classes,           and therefore are hard to model by a 2D CNN. Qi et al. (2017b) construct
only N training samples in the same classes are provided in the training           a k-nearest neighbor (KNN) graph and use a 3D GNN as the propagation
set. Thereby, few-shot learning restricts N to be small, and zero-shot re-         model. After unrolling for several steps, the prediction model takes the
quires N to be 0. Models must learn to generalize from the limited                 hidden state of each node as input and predicts its semantic label. As
training data to make new predictions for testing data. Graph neural               there are always too many points in point clouds classiﬁcation task,
networks, on the other hand, can assist the image classiﬁcation system in          Landrieu and Simonovsky (2018) solve large-scale 3D point clouds seg-
these challenging scenarios.                                                       mentation by building superpoint graphs and generating embeddings for
    First, knowledge graphs can be used as extra information to guide              them. To classify supernodes, Landrieu and Simonovsky (2018) leverage
zero-shot recognition classiﬁcation (Wang et al., 2018d; Kampffmeyer               GGNN and graph convolution. Wang et al. (2018e) propose to model
et al., 2019). Wang et al. (2018d) make the visual classiﬁers learn not            point interactions through edges. They calculate edge representation
only from the visual input but also from word embeddings of the cate-              vectors by feeding the coordinates of its terminal nodes. Then node
gories’ names and their relationships to other categories. A knowledge             embeddings are updated by edge aggregation.
graph is developed to help connect the related categories, and they use a
6-layer GCN to encode the knowledge graph. As the over-smoothing ef-               8.2.2. Text
fect happens when the graph convolution architecture becomes deep, the                 The graph neural networks could be applied to several tasks based on
6-layer GCN used in (Wang et al., 2018d) will wash out much useful                 texts. It could be applied to both sentence-level tasks (e.g. text classiﬁ-
information in the representation. To solve the smoothing problem,                 cation) as well as word-level tasks (e.g. sequence labeling). We list
Kampffmeyer et al. (2019) use a single layer GCN with a larger neigh-              several major applications on text in the following.
borhood which includes both one-hop and multi-hop nodes in the graph.                  Text Classiﬁcation. Text classiﬁcation is an important and classical
And it is proven effective in building a zero-shot classiﬁer from existing         problem in natural language processing. Traditional text classiﬁcation
ones. As most knowledge graphs are large for reasoning, Marino et al.              uses bag-of-words features. However, representing a text as a graph of
(2017) select some related entities to build a sub-graph based on the              words can further capture semantics between non-consecutive and long
result of object detection and apply GGNN to the extracted graph for               distance words (Peng et al., 2018). Peng et al. (2018) use a graph-CNN
prediction. Besides, Lee et al. (2018b) also leverage the knowledge graph          based deep learning model to ﬁrst convert texts to graph-of-words, and
between categories. It further deﬁnes three types of relations between             then use graph convolution operations in (Niepert et al., 2016) to
categories: super-subordinate, positive correlation, and negative corre-           convolve the word graph. Zhang et al. (2018d) propose the Sentence
lation and propagates the conﬁdence of relation labels in the graph                LSTM to encode text. They view the whole sentence as a single state,
directly.                                                                          which consists of sub-states for individual words and an overall
    Except for the knowledge graph, the similarity between images in the           sentence-level state. They use the global sentence-level representation for
dataset is also helpful for the few-shot learning (Garcia and Bruna, 2018).        classiﬁcation tasks. These methods either view a document or a sentence
Garcia and Bruna (2018) build a weighted fully-connected image                     as a graph of word nodes. Yao et al. (2019) regard the documents and
network based on the similarity and do message passing in the graph for            words as nodes to construct the corpus graph and use the Text GCN to
few-shot recognition.                                                              learn embeddings of words and documents. Sentiment classiﬁcation
    Visual Reasoning. Computer-vision systems usually need to perform              could also be regarded as a text classiﬁcation problem and a Tree-LSTM
reasoning by incorporating both spatial and semantic information. So it is         approach is proposed by (Tai et al., 2015).
natural to generate graphs for reasoning tasks.                                        Sequence Labeling. Given a sequence of observed variables (such as
    A typical visual reasoning task is visual question answering (VQA). In         words), sequence labeling is to assign a categorical label for each vari-
this task, a model needs to answer the questions about an image given the          able. Typical tasks include POS-tagging, where we label the words in a
text description of the questions. Usually, the answer lies in the spatial         sentence by their part-of-speech, and Named Entity Recognition (NER),
relations among objects in the image. Teney et al. (2017) construct an             where we predict whether each word in a sentence belongs to a part of a
image scene graph and a question syntactic graph. Then they apply                  Named Entity. If we consider each variable in the sequence as a node and
GGNN to train the embeddings for predicting the ﬁnal answer. Despite               the dependencies as edges, we can utilize the hidden state of GNNs to
spatial connections among objects, Norcliffebrown et al. (2018) build the          address the task. Zhang et al. (2018d) utilize the Sentence LSTM to label
relational graphs conditioned on the questions. With knowledge graphs,             the sequence. They have conducted experiments on POS-tagging and
Wang et al. (2018c), Narasimhan et al. (2018) can perform ﬁner relation            NER tasks and achieves promising performances.
exploration and more interpretable reasoning process.                                  Semantic role labeling is another task of sequence labeling. Marche-
    Other applications of visual reasoning include object detection,               ggiani and Titov (2017) present a Syntactic GCN to solve the problem.
interaction detection, and region classiﬁcation. In object detection (Hu           The Syntactic GCN which operates on the direct graph with labeled edges
et al., 2018; Gu et al., 2018), GNNs are used to calculate RoI features. In        is a special variant of the GCN (Kipf and Welling, 2017). It integrates
interaction detection (Qi et al., 2018; Jain et al., 2016), GNNs are               edge-wise gates which let the model regulate contributions of individual

                                                                              74
J. Zhou et al.                                                                                                                             AI Open 1 (2020) 57–81


dependency edges. The Syntactic GCNs over syntactic dependency trees                remarkable that GNN models are not good enough to offer satisfying
are used as sentence encoders to learn latent feature representations of            solutions for any graph in any condition. In this section, we list some
words in the sentence.                                                              open problems for further researches.
    Neural Machine Translation. The neural machine translation                          Robustness. As a family of models based on neural networks, GNNs
(NMT) task is to translate text from source language to target language             are also vulnerable to adversarial attacks. Compared to adversarial at-
automatically using neural networks. It is usually considered as a                  tacks on images or text which only focuses on features, attacks on graphs
sequence-to-sequence task. Transformer (Vaswani et al., 2017) in-                   further consider the structural information. Several works have been
troduces the attention mechanisms and replaces the most commonly used               proposed to attack existing graph models (Zügner et al., 2018; Dai et al.,
recurrent or convolutional layers. In fact, the Transformer assumes a fully         2018b) and more robust models are proposed to defend (Zhu et al.,
connected graph structure between words. Other graph structure can be               2019). We refer to (Sun et al., 2018) for a comprehensive review.
explored with GNNs.                                                                     Interpretability. Interpretability is also an important research di-
    One popular application of GNN is to incorporate the syntactic or               rection for neural models. But GNNs are also black-boxes and lack of
semantic information into the NMT task. Bastings et al. (2017) utilize the          explanations. Only a few methods (Ying et al., 2019; Baldassarre and
Syntactic GCN on syntax-aware NMT tasks. Marcheggiani et al. (2018)                 Azizpour, 2019) are proposed to generate example-level explanations for
incorporate information about the predicate-argument structure of                   GNN models. It is important to apply GNN models on real-world appli-
source sentences (namely, semantic-role representations) using Syntactic            cations with trusted explanations. Similar to the ﬁelds of CV and NLP,
GCN and compare the results between incorporating only syntactic, only              interpretability on graphs is also an important direction to investigate.
semantic information and both of the information. Beck et al. (2018)                    Graph Pretraining. Neural network-based models require abundant
utilize the GGNN in syntax-aware NMT. They convert the syntactic de-                labeled data and it is costly to obtain enormous human-labeled data. Self-
pendency graph into a new structure called the Levi graph (Levi, 1942)              supervised methods are proposed to guide models to learn from unla-
by turning the edges into additional nodes and thus edge labels can be              beled data which is easy to obtain from websites or knowledge bases.
represented as embeddings.                                                          These methods have achieved great success in the area of CV and NLP
    Relation Extraction. Extracting semantic relations between entities             with the idea of pretraining (Krizhevsky et al., 2012; Devlin et al., 2019).
in texts helps to expand existing knowledge base. Traditional methods               Recently, there have been works focusing on pretraining on graphs (Qiu
use CNNs or RNNs to learn entities’ feature and predict the relation type           et al., 2020; Hu et al., 2020b, 2020e; Zhang et al., 2020), but they have
for a pair of entities. A more sophisticated way is to utilize the de-              different problem settings and focus on different aspects. This ﬁeld still
pendency structure of the sentence. A document graph can be built where             has many open problems requiring research efforts, such as the design of
nodes represent words and edges represent various dependencies such as              the pretraining tasks, the effectiveness of existing GNN models on
adjacency, syntactic dependencies and discourse relations. Zhang et al.             learning structural or feature information, etc.
(2018f) propose an extension of graph convolutional networks that is                    Complex Graph Structures. Graph structures are ﬂexible and com-
tailored for relation extraction and apply a pruning strategy to the input          plex in real life applications. Various works are proposed to deal with
trees.                                                                              complex graph structures such as dynamic graphs or heterogeneous
    Cross-sentence N-ary relation extraction detects relations among n              graphs as we have discussed before. With the rapid development of social
entities across multiple sentences. Peng et al. (2017) explore a general            networks on the Internet, there are certainly more problems, challenges
framework for cross-sentence n-ary relation extraction by applying graph            and application scenarios emerging and requiring more powerful models.
LSTMs on the document graphs. Song et al. (2018b) also use a graph-state
LSTM model and speed up computation by allowing more parallelization.               10. Conclusion
    Event Extraction. Event extraction is an important information
extraction task to recognize instances of speciﬁed types of events in texts.            Over the past few years, graph neural networks have become
This is always conducted by recognizing the event triggers and then                 powerful and practical tools for machine learning tasks in graph domain.
predicting the arguments for each trigger. Nguyen and Grishman (2018)               This progress owes to advances in expressive power, model ﬂexibility,
investigate a convolutional neural network (which is the Syntactic GCN              and training algorithms. In this survey, we conduct a comprehensive
exactly) based on dependency trees to perform event detection. Liu et al.           review of graph neural networks. For GNN models, we introduce its
(2018) propose a Jointly Multiple Events Extraction (JMEE) framework                variants categorized by computation modules, graph types, and training
to jointly extract multiple event triggers and arguments by introducing             types. Moreover, we also summarize several general frameworks and
syntactic shortcut arcs to enhance information ﬂow to attention-based               introduce several theoretical analyses. In terms of application taxonomy,
graph convolution networks to model graph information.                              we divide the GNN applications into structural scenarios, non-structural
    Fact Veriﬁcation. Fact veriﬁcation is a task requiring models to                scenarios, and other scenarios, then give a detailed review for applica-
extract evidence to verify given claims. However, some claims require               tions in each scenario. Finally, we suggest four open problems indicating
reasoning on multiple pieces of evidence. GNN-based methods like GEAR               the major challenges and future research directions of graph neural
(Zhou et al., 2019) and KGAT (Liu et al., 2020) are proposed to conduct             networks, including robustness, interpretability, pretraining and com-
evidence aggregating and reasoning based on a fully connected evidence              plex structure modeling.
graph. Zhong et al. (2020) build an inner-sentence graph with the in-
formation from semantic role labeling and achieve promising results.                Declaration of competing interest
    Other Applications on Text. GNNs can also be applied to many other
tasks on text. For example, GNNs are also used in question answering and                The authors declare that they have no known competing ﬁnancial
reading comprehension (Song et al., 2018c; De Cao et al., 2019; Qiu et al.,         interests or personal relationships that could have appeared to inﬂuence
2019; Tu et al., 2019; Ding et al., 2019). Another important direction is           the work reported in this paper.
relational reasoning, relational networks (Santoro et al., 2017), interac-
tion networks (Battaglia et al., 2016) and recurrent relational networks            Acknowledgements
(Palm et al., 2018) are proposed to solve the relational reasoning task
based on text.                                                                         This work is supported by the National Key Research and Develop-
                                                                                    ment Program of China (No. 2018YFB1004503), the National Natural
9. Open problems                                                                    Science Foundation of China (NSFC No.61772302) and the Beijing
                                                                                    Academy of Artiﬁcial Intelligence (BAAI). This work is also supported by
    Although GNNs have achieved great success in different ﬁelds, it is             2019 Tencent Marketing Solution Rhino-Bird Focused Research Program

                                                                               75
J. Zhou et al.                                                                                                                                              AI Open 1 (2020) 57–81


FR201908.

Appendix A. Datasets

   Many tasks related to graphs are released to test the performance of various graph neural networks. Such tasks are based on the following commonly
used datasets. We list the datasets in Table A.4.

Table A.4
Datasets commonly used in tasks related to graph.

  Field                   Datasets

  Citation Networks       Pubmed (Yang et al., 2016) Cora (Yang et al., 2016) Citeseer (Yang et al., 2016) DBLP (Tang et al., 2008)
  Bio-chemical            MUTAG (Debnath et al., 1991) NCI-1 (Wale et al., 2008) PPI (Zitnik and Leskovec, 2017) D&D (Dobson and Doig, 2003) PROTEIN (Borgwardt et al., 2005) PTC
    Graphs                (Toivonen et al., 2003)
  Social Networks         Reddit (Hamilton et al., 2017c) BlogCatalog (Zafarani and Liu, 2009)
  Knowledge Graphs        FB13 (Socher et al., 2013) FB15K (Bordes et al., 2013) FB15K237 (Toutanova et al., 2015) WN11 (Socher et al., 2013) WN18 (Bordes et al., 2013) WN18RR
                          (Dettmers et al., 2018)

    There are also a broader range of open source datasets repository which contains more graph datasets. We list them in Table A.5.

Table A.5
Popular graph learning dataset collections.

  Repository                         Introduction                                                                                Link

  Network Repository                 A scientiﬁc network data repository interactive visualization and mining tools.             http://networkrepository.com
  Graph Kernel Datasets              Benchmark datasets for graph kernels.                                                       https://ls11-www.cs.tu-dortmund.de/staff/m
                                                                                                                                 orris/graphkerneldatasets
  Relational Dataset Repository      To support the growth of relational machine learning                                        https://relational.ﬁt.cvut.cz/
  Stanford Large Network             The SNAP library is developed to study large social and information networks.               https://snap.stanford.edu/data/
    Dataset Collection
  Open Graph Benchmark               Open Graph Benchmark (OGB) is a collection of benchmark datasets, data-loaders and          https://ogb.stanford.edu
                                     evaluators for graph machine learning in PyTorch.


Appendix B. Implementations

    We ﬁrst list several platforms that provide codes for graph computing in Table B.6.

                      Table B.6
                      Popular platforms for graph computing.

                       Platform                                   Link                                                               Reference

                       PyTorch Geometric                          https://github.com/rusty1s/pytorch_geometric                       Fey and Lenssen (2019)
                       Deep Graph Library                         https://github.com/dmlc/dgl                                        Wang et al. (2019b)
                       AliGraph                                   https://github.com/alibaba/aligraph                                Zhu et al. (2019a)
                       GraphVite                                  https://github.com/DeepGraphLearning/graphvite                     Zhu et al. (2019b)
                       Paddle Graph Learning                      https://github.com/PaddlePaddle/PGL
                       Euler                                      https://github.com/alibaba/euler
                       Plato                                      https://github.com/tencent/plato
                       CogDL                                      https://github.com/THUDM/cogdl/
                       OpenNE                                     https://github.com/thunlp/OpenNE/tree/pytorch

    Next we list the hyperlinks of the current open source implementations of some famous GNN models in Table B.7:

                 Table B.7
                 Source code of the models mentioned in the survey.

                  Model                                       Link

                  GGNN (2015)                                 https://github.com/yujiali/ggnn
                  Neurals FPs (2015)                          https://github.com/HIPS/neural-ﬁngerprint
                  ChebNet (2016)                              https://github.com/mdeff/cnn_graph
                  DNGR (2016)                                 https://github.com/ShelsonCao/DNGR
                  SDNE (2016)                                 https://github.com/suanrong/SDNE
                  GAE (2016)                                  https://github.com/limaosen0/Variational-Graph-Auto-Encoders
                  DRNE (2016)                                 https://github.com/tadpole/DRNE
                  Structural RNN (2016)                       https://github.com/asheshjain399/RNNexp
                  DCNN (2016)                                 https://github.com/jcatw/dcnn
                  GCN (2017)                                  https://github.com/tkipf/gcn
                  CayleyNet (2017)                            https://github.com/amoliu/CayleyNet
                  GraphSage (2017)                            https://github.com/williamleif/GraphSAGE
                  GAT (2017)                                  https://github.com/PetarV-/GAT
                  CLN (2017)                                  https://github.com/trangptm/Column_networks
                  ECC (2017)                                  https://github.com/mys007/ecc
                  MPNNs (2017)                                https://github.com/brain-research/mpnn
                  MoNet (2017)                                https://github.com/pierrebaque/GeometricConvolutionsBench
                                                                                                                                        (continued on next column)


                                                                                           76
J. Zhou et al.                                                                                                                                                                AI Open 1 (2020) 57–81

                  Table B.7 (continued )
                    Model                                          Link

                    JK-Net (2018)                                  https://github.com/ShinKyuY/Representation_Learning_on_Graphs_with_Jumping_Knowledge_Networks
                    SSE (2018)                                     https://github.com/Hanjun-Dai/steady_state_embedding
                    LGCN (2018)                                    https://github.com/divelab/lgcn/
                    FastGCN (2018)                                 https://github.com/matenure/FastGCN
                    DiffPool (2018)                                https://github.com/RexYing/diffpool
                    GraphRNN (2018)                                https://github.com/snap-stanford/GraphRNN
                    MolGAN (2018)                                  https://github.com/nicola-decao/MolGAN
                    NetGAN (2018)                                  https://github.com/danielzuegner/netgan
                    DCRNN (2018)                                   https://github.com/liyaguang/DCRNN
                    ST-GCN (2018)                                  https://github.com/yysijie/st-gcn
                    RGCN (2018)                                    https://github.com/tkipf/relational-gcn
                    AS-GCN (2018)                                  https://github.com/huangwb/AS-GCN
                    DGCN (2018)                                    https://github.com/ZhuangCY/DGCN
                    GaAN (2018)                                    https://github.com/jennyzhang0215/GaAN
                    DGI (2019)                                     https://github.com/PetarV-/DGI
                    GraphWaveNet (2019)                            https://github.com/nnzhan/Graph-WaveNet
                    HAN (2019)                                     https://github.com/Jhy1993/HAN

   As the research ﬁled grows rapidly, we recommend our readers the paper list published by our team, GNNPapers (https://github.com/thunlp/
gnnpapers), for recent papers.


References                                                                                              Chen, X., Li, L.-J., Fei-Fei, L., Gupta, A., 2018d. Iterative visual reasoning beyond
                                                                                                            convolutions. In: Proceedings of CVPR, pp. 7239–7248.
                                                                                                        Chen, Z., Villar, S., Chen, L., Bruna, J., 2019. On the equivalence between graph
Allamanis, M., Brockschmidt, M., Khademi, M., 2018. Learning to represent programs
                                                                                                            isomorphism testing and function approximation with gnns. In: Proceedings of
     with graphs. Proc. ICLR.
                                                                                                            NeurIPS, pp. 15894–15902.
Atwood, J., Towsley, D., 2016. Diffusion-convolutional neural networks. In: Proceedings
                                                                                                        Chen, L., Li, J., Peng, J., Xie, T., Cao, Z., Xu, K., He, X., Zheng, Z., 2020a. A Survey of
     of NIPS, pp. 2001–2009.
                                                                                                            Adversarial Learning on Graphs arXiv preprint arXiv:2003.05730.
Bahdanau, D., Cho, K., Bengio, Y., 2015. Neural machine translation by jointly learning to
                                                                                                        Chen, D., Lin, Y., Li, W., Li, P., Zhou, J., Sun, X., 2020b. Measuring and relieving the over-
     align and translate. In: Proceedings of ICLR.
                                                                                                            smoothing problem for graph neural networks from the topological view.
Baldassarre, F., Azizpour, H., 2019. Explainability Techniques for Graph Convolutional
                                                                                                            Proceedings of AAAI 3438–3445.
     Networks. ICML Workshop on Learning and Reasoning with Graph-Structured
                                                                                                        Cheng, J., Dong, L., Lapata, M., 2016. Long short-term memory-networks for machine
     Representations.
                                                                                                            reading. In: Proceedings of EMNLP, pp. 551–561.
Barcelo, P., Kostylev, E.V., Monet, M., Perez, J., Reutter, J., Silva, J.P., 2019. The logical
                                                                                                        Chiang, W.-L., Liu, X., Si, S., Li, Y., Bengio, S., Hsieh, C.-J., 2019. Cluster-gcn: an efﬁcient
     expressiveness of graph neural networks. Proceedings of ICLR.
                                                                                                            algorithm for training deep and large graph convolutional networks. In: Proceedings
Bastings, J., Titov, I., Aziz, W., Marcheggiani, D., Simaan, K., 2017. Graph convolutional
                                                                                                            of KDD, pp. 257–266.
     encoders for syntax-aware neural machine translation. In: Proceedings of EMNLP,
                                                                                                        Cho, K., Van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H.,
     pp. 1957–1967.
                                                                                                            Bengio, Y., 2014. Learning phrase representations using rnn encoder–decoder for
Battaglia, P., Pascanu, R., Lai, M., Rezende, D.J., et al., 2016. Interaction networks for
                                                                                                            statistical machine translation. Proceedings of EMNLP 1724–1734.
     learning about objects, relations and physics. Proceedings of NIPS 4509–4517.
                                                                                                        Cui, P., Wang, X., Pei, J., Zhu, W., 2018a. A survey on network embedding. IEEE TKDE
Battaglia, P.W., Hamrick, J.B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V.,
                                                                                                            833–852.
     Malinowski, M., Tacchetti, A., Raposo, D., Santoro, A., Faulkner, R., et al., 2018.
                                                                                                        Cui, Z., Henrickson, K., Ke, R., Wang, Y., 2018b. Trafﬁc Graph Convolutional Recurrent
     Relational Inductive Biases, Deep Learning, and Graph Networks. arXiv preprint
                                                                                                            Neural Network: A Deep Learning Framework for Network-Scale Trafﬁc Learning and
     arXiv:1806.01261.
                                                                                                            Forecasting arXiv preprint arXiv:1802.07007.
Beck, D., Haffari, G., Cohn, T., 2018. Graph-to-sequence learning using gated graph
                                                                                                        Cui, G., Zhou, J., Yang, C., Liu, Z., 2020. Adaptive graph encoder for attributed graph
     neural networks. Proceedings of ACL 273–283.
                                                                                                            embedding. In: Proceedings of KDD, pp. 976–985.
Bello, I., Pham, H., Le, Q.V., Norouzi, M., Bengio, S., 2017. Neural Combinatorial
                                                                                                        Dai, H., Dai, B., Song, L., 2016. Discriminative embeddings of latent variable models for
     Optimization with Reinforcement Learning. arXiv preprint arXiv:1611.09940.
                                                                                                            structured data. Proceedings of ICML 2702–2711.
Bojchevski, A., Klicpera, J., Perozzi, B., Kapoor, A., Blais, M., R  ozemberczki, B.,
                                                                                                        Dai, H., Kozareva, Z., Dai, B., Smola, A., Song, L., 2018a. Learning steady-states of
     Lukasik, M., Günnemann, S., 2020. Scaling graph neural networks with approximate
                                                                                                            iterative algorithms over graphs. Proceedings of ICML 1106–1114.
     pagerank. In: Proceedings of KDD. ACM, pp. 2464–2473.
                                                                                                        Dai, H., Li, H., Tian, T., Huang, X., Wang, L., Zhu, J., Song, L., 2018b. Adversarial attack
Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., Yakhnenko, O., 2013. Translating
                                                                                                            on graph structured data. In: Proceedings of ICML, pp. 1115–1124.
     embeddings for modeling multi-relational data. Proceedings of NIPS 1–9.
                                                                                                        De Cao, N., Kipf, T., 2018. MolGAN: an Implicit Generative Model for Small Molecular
Borgwardt, K.M., Ong, C.S., Sch€   onauer, S., Vishwanathan, S., Smola, A.J., Kriegel, H.-P.,
                                                                                                            Graphs. ICML 2018 Workshop on Theoretical Foundations and Applications of Deep
     2005. Protein function prediction via graph kernels. Bioinformatics 21, i47–i56.
                                                                                                            Generative Models.
Boscaini, D., Masci, J., Rodol a, E., Bronstein, M., 2016. Learning shape correspondence
                                                                                                        De Cao, N., Aziz, W., Titov, I., 2019. Question answering by reasoning across documents
     with anisotropic convolutional neural networks. In: Proceedings of NIPS,
                                                                                                            with graph convolutional networks. Proceedings of NAACL 2306–2317.
     pp. 3197–3205.
                                                                                                        Debnath, A.K., Lopez de Compadre, R.L., Debnath, G., Shusterman, A.J., Hansch, C., 1991.
Bronstein, M.M., Bruna, J., LeCun, Y., Szlam, A., Vandergheynst, P., 2017. Geometric
                                                                                                            Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro
     deep learning: going beyond euclidean data. IEEE SPM 34, 18–42.
                                                                                                            compounds. J. Med. Chem. 786–797.
Bruna, J., Zaremba, W., Szlam, A., Lecun, Y., 2014. Spectral networks and locally
                                                                                                        Defferrard, M., Bresson, X., Vandergheynst, P., 2016. Convolutional neural networks on
     connected networks on graphs. In: Proceedings of ICLR.
                                                                                                            graphs with fast localized spectral ﬁltering. In: Proceedings of NIPS, pp. 3844–3852.
Buades, A., Coll, B., Morel, J.-M., 2005. A non-local algorithm for image denoising.
                                                                                                        Dehmamy, N., Barabasi, A.-L., Yu, R., 2019. Understanding the representation power of
     Proceedings of CVPR, 2. IEEE, pp. 60–65.
                                                                                                            graph neural networks in learning graph topology. Proceedings of NeurIPS
Cai, H., Zheng, V.W., Chang, K.C.-C., 2017. Active Learning for Graph Embedding. arXiv
                                                                                                            15413–15423.
     preprint arXiv:1705.05085.
                                                                                                        Derr, T., Ma, Y., Tang, J., 2018. Signed graph convolutional networks. Proceedings of
Cai, H., Zheng, V.W., Chang, K.C.-C., 2018. A comprehensive survey of graph embedding:
                                                                                                            ICDM, pp. 929–934.
     problems, techniques, and applications. IEEE TKDE 30, 1616–1637.
                                                                                                        Dettmers, T., Minervini, P., Stenetorp, P., Riedel, S., 2018. Convolutional 2d knowledge
Chami, I., Abu-El-Haija, S., Perozzi, B., Re, C., Murphy, K., 2020. Machine Learning on
                                                                                                            graph embeddings. Proceedings of AAAI 1811–1818.
     Graphs: A Model and Comprehensive Taxonomy. arXiv preprint arXiv:2005.03675.
                                                                                                        Devlin, J., Chang, M.-W., Lee, K., Toutanova, K., 2019. Bert: pre-training of deep
Chang, M., Ullman, T., Torralba, A., Tenenbaum, J.B., 2017. A compositional object-based
                                                                                                            bidirectional transformers for language understanding. Proceedings of NAACL
     approach to learning physical dynamics. Proceedings of ICLR.
                                                                                                            4171–4186.
Chen, J., Zhu, J., Song, L., 2018a. Stochastic training of graph convolutional networks
                                                                                                        Dhillon, I.S., Guan, Y., Kulis, B., 2007. Weighted graph cuts without eigenvectors a
     with variance reduction. In: Proceedings of ICML, pp. 942–950.
                                                                                                            multilevel approach. IEEE TPAMI 29, 1944–1957.
Chen, J., Ma, T., Xiao, C., 2018b. Fastgcn: fast learning with graph convolutional
                                                                                                        Ding, M., Zhou, C., Chen, Q., Yang, H., Tang, J., 2019. Cognitive graph for multi-hop
     networks via importance sampling. Proceedings of ICLR.
                                                                                                            reading comprehension at scale. In: Proceedings of ACL, pp. 2694–2703.
Chen, Y., Wei, Z., Huang, X., 2018c. Incorporating corporation relationship via graph
                                                                                                        Do, K., Tran, T., Venkatesh, S., 2019. Graph transformation policy network for chemical
     convolutional neural networks for stock price prediction. In: Proceedings of CIKM,
                                                                                                            reaction prediction. Proceedings of SIGKDD 750–760.
     pp. 1655–1658.


                                                                                                   77
J. Zhou et al.                                                                                                                                                            AI Open 1 (2020) 57–81

Dobson, P.D., Doig, A.J., 2003. Distinguishing enzyme structures from non-enzymes                     Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., Leskovec, J., 2020d.
     without alignments. J. Mol. Biol. 330, 771–783.                                                       Open graph benchmark: datasets for machine learning on graphs. Proceedings of
Duvenaud, D.K., Maclaurin, D., Aguileraiparraguirre, J., Gomezbombarelli, R.,                              NeurIPS.
     Hirzel, T.D., Aspuruguzik, A., Adams, R.P., 2015. Convolutional networks on graphs               Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., Leskovec, J., 2020e. Strategies
     for learning molecular ﬁngerprints. In: Proceedings of NIPS, pp. 2224–2232.                           for pre-training graph neural networks. Proceedings of ICLR.
Dwivedi, V.P., Joshi, C.K., Laurent, T., Bengio, Y., Bresson, X., 2020. Benchmarking Graph            Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q., 2017. Densely connected
     Neural Networks. arXiv preprint arXiv:2003.00982.                                                     convolutional networks. In: Proceedings of CVPR, pp. 4700–4708.
Errica, F., Podda, M., Bacciu, D., Micheli, A., 2020. A fair comparison of graph neural               Huang, W., Zhang, T., Rong, Y., Huang, J., 2018. Adaptive sampling towards fast graph
     networks for graph classiﬁcation. In: Proceedings of ICLR.                                            representation learning. Proceedings of NeurIPS 4558–4567.
Fan, W., Ma, Y., Li, Q., He, Y., Zhao, E., Tang, J., Yin, D., 2019. Graph neural networks for         Huang, Y., Xu, H., Duan, Z., Ren, A., Feng, J., Wang, X., 2020. Modeling Complex Spatial
     social recommendation. In: Proceedings of WWW, pp. 417–426.                                           Patterns with Temporal Features via Heterogenous Graph Embedding Networks.
Feng, Y., You, H., Zhang, Z., Ji, R., Gao, Y., 2019. Hypergraph neural networks. In:                       arXiv preprint arXiv:2008.08617.
     Proceedings of AAAI, vol. 33, pp. 3558–3565.                                                     Jaeger, H., 2001. The “Echo State” Approach to Analysing and Training Recurrent Neural
Fey, M., Lenssen, J.E., 2019. Fast graph representation learning with PyTorch Geometric.                   Networks-With an Erratum Note, vol. 148. German National Research Center for
     In: ICLR Workshop on Representation Learning on Graphs and Manifolds.                                 Information Technology GMD Technical Report, p. 13.
Fout, A., Byrd, J., Shariat, B., Ben-Hur, A., 2017. Protein interface prediction using graph          Jain, A., Zamir, A.R., Savarese, S., Saxena, A., 2016. Structural-rnn: deep learning on
     convolutional networks. In: Proceedings of NIPS, pp. 6533–6542.                                       spatio-temporal graphs. In: Proceedings of CVPR, pp. 5308–5317.
Frasconi, P., Gori, M., Sperduti, A., 1998. A general framework for adaptive processing of            Kampffmeyer, M., Chen, Y., Liang, X., Wang, H., Zhang, Y., Xing, E.P., 2019. Rethinking
     data structures. IEEE TNN 9, 768–786.                                                                 Knowledge Graph Propagation for Zero-Shot Learning. Proceedings of CVPR,
Fu, X., Zhang, J., Meng, Z., King, I., Magnn, 2020. Metapath aggregated graph neural                       pp. 11487–11496.
     network for heterogeneous graph embedding. Proceedings of WWW 2331–2341.                         Kearnes, S., McCloskey, K., Berndl, M., Pande, V., Riley, P., 2016. Molecular graph
Gallicchio, C., Micheli, A., 2010. Graph echo state networks. In: Proceedings of IJCNN.                    convolutions: moving beyond ﬁngerprints. J. Comput. Aided Mol. Des. 30, 595–608.
     IEEE, pp. 1–8.                                                                                   Keriven, N., Peyre, G., 2019. Universal invariant and equivariant graph neural networks.
Gao, H., Ji, S., 2019. Graph u-nets. In: Proceedings of ICML, pp. 2083–2092.                               In: Proceedings of NeurIPS, pp. 7092–7101.
Gao, H., Wang, Z., Ji, S., 2018a. Large-scale learnable graph convolutional networks. In:             Khalil, E., Dai, H., Zhang, Y., Dilkina, B., Song, L., 2017. Learning combinatorial
     Proceedings of KDD. ACM, pp. 1416–1424.                                                               optimization algorithms over graphs. Proceedings of NIPS 6348–6358.
Gao, L., Yang, H., Zhou, C., Wu, J., Pan, S., Hu, Y., 2018b. Active discriminative network            Khamsi, M.A., Kirk, W.A., 2011. An Introduction to Metric Spaces and Fixed Point Theory,
     representation learning. In: Proceedings of IJCAI.                                                    vol. 53. John Wiley & Sons.
Garcia, V., Bruna, J., 2018. Few-shot learning with graph neural networks. Proceedings of             Kim, R., So, C.H., Jeong, M., Lee, S., Kim, J., Kang, J., 2019. Hats: A Hierarchical Graph
     ICLR.                                                                                                 Attention Network for Stock Movement Prediction arXiv preprint arXiv:1908.07999.
Garg, V.K., Jegelka, S., Jaakkola, T., 2020. Generalization and representational limits of            Kipf, T.N., Welling, M., 2016. Variational graph auto-encoders. In: NIPS Bayesian Deep
     graph neural networks. Proceedings of ICML 3419–3430.                                                 Learning Workshop.
Gasse, M., Chetelat, D., Ferroni, N., Charlin, L., Lodi, A., 2019. Exact combinatorial               Kipf, T.N., Welling, M., 2017. Semi-supervised classiﬁcation with graph convolutional
     optimization with graph convolutional neural networks. In: Proceedings of NeurIPS,                    networks. Proceedings of ICLR.
     pp. 15580–15592.                                                                                 Kipf, T.N., Fetaya, E., Wang, K., Welling, M., Zemel, R.S., 2018. Neural relational
Gehring, J., Auli, M., Grangier, D., Dauphin, Y.N., 2017. A convolutional encoder model                    inference for interacting systems. In: Proceedings of ICML. PMLR, pp. 2688–2697.
     for neural machine translation. In: Proceedings of ACL, 1, pp. 123–135.                          Klicpera, J., Bojchevski, A., Günnemann, S., 2019. Predict then propagate: graph neural
Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., Dahl, G.E., 2017. Neural message                   networks meet personalized pagerank. In: Proceedings of ICLR.
     passing for quantum chemistry. Proceedings of ICML 1263–1272.                                    Knyazev, B., Taylor, G.W., Amer, M., 2019. Understanding attention and generalization in
Gori, M., Monfardini, G., Scarselli, F., 2005. A new model for learning in graph domains.                  graph neural networks. In: Proceedings of NeurIPS, pp. 4202–4212.
     Proceedings of IJCNN, 2. IEEE, pp. 729–734.                                                      Kool, W., van Hoof, H., Welling, M., 2019. Attention, learn to solve routing problems!. In:
Goyal, P., Ferrara, E., 2018. Graph embedding techniques, applications, and performance:                   Proceedings of ICLR.
     a survey. Knowl. Base Sys. 151, 78–94.                                                           Krizhevsky, A., Sutskever, I., Hinton, G.E., 2012. Imagenet classiﬁcation with deep
Grover, A., Leskovec, J., 2016. node2vec: scalable feature learning for networks. In:                      convolutional neural networks. Proceedings of NIPS 1097–1105.
     Proceedings of KDD. ACM, pp. 855–864.                                                            Landrieu, L., Simonovsky, M., 2018. Large-scale point cloud semantic segmentation with
Grover, A., Zweig, A., Ermon, S., 2019. Graphite: iterative generative modeling of graphs.                 superpoint graphs. Proceedings of CVPR 4558–4567.
     In: Proceedings of ICML, pp. 2434–2444.                                                          LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., 1998. Gradient-based learning applied to
Gu, J., Hu, H., Wang, L., Wei, Y., Dai, J., 2018. Learning region features for object                      document recognition. In: Proceedings of the IEEE, 86, pp. 2278–2324.
     detection. In: Proceedings of ECCV, pp. 381–395.                                                 LeCun, Y., Bengio, Y., Hinton, G., 2015. Deep learning. Nature 521, 436–444.
Guo, S., Lin, Y., Feng, N., Song, C., Wan, H., 2019. Attention based spatial-temporal graph           Lee, J.B., Rossi, R.A., Kim, S., Ahmed, N.K., Koh, E., 2018a. Attention Models in Graphs: A
     convolutional networks for trafﬁc ﬂow forecasting. Proceedings of AAAI 33,                            Survey. TKDD 13, 1–25.
     922–929.                                                                                         Lee, C., Fang, W., Yeh, C., Wang, Y.F., 2018b. Multi-label zero-shot learning with
Hamaguchi, T., Oiwa, H., Shimbo, M., Matsumoto, Y., 2017. Knowledge transfer for out-                      structured knowledge graphs. In: Proceedings of CVPR, pp. 1576–1585.
     of-knowledge-base entities : a graph neural network approach. In: Proceedings of                 Lee, J., Lee, I., Kang, J., 2019. Self-attention graph pooling. In: Proceedings of ICML,
     IJCAI, pp. 1802–1808. https://doi.org/10.24963/ijcai.2017/250.                                        pp. 3734–3743.
Hamilton, W.L., Ying, Z., Leskovec, J., 2017a. Inductive representation learning on large             Levi, F.W., 1942. Finite Geometrical Systems: Six Public Lectues Delivered in February,
     graphs. In: Proceedings of NIPS, pp. 1024–1034.                                                       1940. the University of Calcutta, University of Calcutta.
Hamilton, W.L., Ying, R., Leskovec, J., 2017b. Representation learning on graphs:                     Levie, R., Huang, W., Bucci, L., Bronstein, M.M., Kutyniok, G., 2019. Transferability of
     methods and applications. IEEE Data(base) Engineering Bulletin 40, 52–74.                             Spectral Graph Convolutional Neural Networks. arXiv preprint arXiv:1907.12972.
Hamilton, W.L., Zhang, J., Danescu-Niculescu-Mizil, C., Jurafsky, D., Leskovec, J., 2017c.            Li, Y., Tarlow, D., Brockschmidt, M., Zemel, R.S., 2016. Gated graph sequence neural
     Loyalty in online communities. In: Proceedings of ICWSM, pp. 540–543.                                 networks. In: Proceedings of ICLR.
Hammer, B., Micheli, A., Sperduti, A., Strickert, M., 2004. Recursive self-organizing                 Li, R., Wang, S., Zhu, F., Huang, J., 2018a. Adaptive graph convolutional neural networks.
     network models. Neural Network. 17, 1061–1085.                                                        In: Proceedings of AAAI, 32, 2018.
Hammond, D.K., Vandergheynst, P., Gribonval, R., 2011. Wavelets on graphs via spectral                Li, Y., Yu, R., Shahabi, C., Liu, Y., 2018b. Diffusion convolutional recurrent neural
     graph theory. Appl. Comput. Harmon. Anal. 30, 129–150.                                                network: Data-driven trafﬁc forecasting. In: Proceedings of ICLR.
Hassani, K., Khasahmadi, A.H., 2020. Contrastive multi-view representation learning on                Li, Q., Han, Z., Wu, X.-M., 2018c. Deeper insights into graph convolutional networks for
     graphs. In: Proceedings of ICML, pp. 4116–4126.                                                       semi-supervised learning. Proceedings of AAAI 32.
He, K., Zhang, X., Ren, S., Sun, J., 2016a. Deep residual learning for image recognition. In:         Li, Y., Vinyals, O., Dyer, C., Pascanu, R., Battaglia, P., 2018d. Learning Deep Generative
     Proceedings of CVPR, pp. 770–778.                                                                     Models of Graphs arXiv preprint arXiv:1803.03324.
He, K., Zhang, X., Ren, S., Sun, J., 2016b. Identity mappings in deep residual networks. In:          Li, Z., Chen, Q., Koltun, V., 2018e. Combinatorial optimization with graph convolutional
     Proceedings of ECCV. Springer, pp. 630–645.                                                           networks and guided tree search. Proceedings of NeurIPS 537–546.
Henaff, M., Bruna, J., Lecun, Y., 2015. Deep Convolutional Networks on Graph-Structured               Li, G., Muller, M., Thabet, A., Ghanem, B., 2019a. Deepgcns: can gcns go as deep as cnns?.
     Data. arXiv preprint arXiv:1506.05163.                                                                In: Proceedings of ICCV, pp. 9267–9276.
Hochreiter, S., Schmidhuber, J., 1997. Long short-term memory. Neural comput. 9,                      Li, Y., Gu, C., Dullien, T., Vinyals, O., Kohli, P., 2019b. Graph Matching Networks for
     1735–1780.                                                                                            Learning the Similarity of Graph Structured Objects. In: Proceedings of ICML,
Hoshen, Y., 2017. Vain: attentional multi-agent predictive modeling. In: Proceedings of                    pp. 3835–3845.
     NIPS, pp. 2698–2708.                                                                             Li, W., Bao, R., Harimoto, K., Chen, D., Xu, J., Su, Q., 2020. Modeling the stock relation
Hu, H., Gu, J., Zhang, Z., Dai, J., Wei, Y., 2018. Relation networks for object detection. In:             with graph network for overnight stock movement prediction. In: Proceedings of
     Proceedings of CVPR, pp. 3588–3597.                                                                   IJCAI, pp. 4541–4547.
Hu, Z., Dong, Y., Wang, K., Sun, Y., 2020a. Heterogeneous graph transformer. In:                      Liang, X., Shen, X., Feng, J., Lin, L., Yan, S., 2016. Semantic object parsing with graph
     Proceedings of WWW, pp. 2704–2710.                                                                    lstm. In: Proceedings of ECCV, pp. 125–143.
Hu, Z., Dong, Y., Wang, K., Chang, K.-W., Sun, Y., 2020b. Gpt-gnn: generative pre-training            Liang, X., Lin, L., Shen, X., Feng, J., Yan, S., Xing, E.P., 2017. Interpretable structure-
     of graph neural networks. In: Proceedings of KDD, pp. 1857–1867.                                      evolving lstm. In: Proceedings of CVPR, pp. 2175–2184.
Hu, S., Xiong, Z., Qu, M., Yuan, X., C^  ote, M.-A., Liu, Z., Tang, J., 2020c. Graph policy          Liu, X., Luo, Z., Huang, H., 2018. Jointly multiple events extraction via attention-based
     network for transferable active learning on graphs. In: Proceedings of NeurIPS, 33.                   graph information aggregation. Proceedings of EMNLP 1247–1256.


                                                                                                 78
J. Zhou et al.                                                                                                                                                           AI Open 1 (2020) 57–81

Liu, J., Kumar, A., Ba, J., Kiros, J., Swersky, K., 2019. Graph normalizing ﬂows. In:                Qi, S., Wang, W., Jia, B., Shen, J., Zhu, S.-C., 2018. Learning human-object interactions by
     Proceedings of NeurIPS, pp. 13578–13588.                                                             graph parsing neural networks. In: Proceedings of ECCV, pp. 401–417.
Liu, Z., Xiong, C., Sun, M., Liu, Z., 2020. Fine-grained fact veriﬁcation with kernel graph          Qiu, L., Xiao, Y., Qu, Y., Zhou, H., Li, L., Zhang, W., Yu, Y., 2019. Dynamically fused graph
     attention network. In: Proceedings of ACL, pp. 7342–7351.                                            network for multi-hop reasoning. In: Proceedings of ACL, pp. 6140–6150.
Loukas, A., 2020. What graph neural networks cannot learn: depth vs width. In:                       Qiu, J., Chen, Q., Dong, Y., Zhang, J., Yang, H., Ding, M., Wang, K., Tang, J., 2020. Gcc:
     Proceedings of ICLR.                                                                                 graph contrastive coding for graph neural network pre-training. Proceedings of KDD
Ma, T., Chen, J., Xiao, C., 2018. Constrained generation of semantically valid graphs via                 1150–1160.
     regularizing variational autoencoders. Proceedings of NeurIPS 7113–7124.                        Rahimi, A., Cohn, T., Baldwin, T., 2018. Semi-supervised user geolocation via graph
Ma, Y., Wang, S., Aggarwal, C.C., Tang, J., 2019a. Graph convolutional networks with                      convolutional networks. In: Proceeding of ACL, vol. 1, pp. 2009–2019.
     eigenpooling. In: Proceedings of KDD, pp. 723–731.                                              Raposo, D., Santoro, A., Barrett, D.G.T., Pascanu, R., Lillicrap, T.P., Battaglia, P., 2017.
Ma, Y., Wang, S., Aggarwal, C.C., Yin, D., Tang, J., 2019b. Multi-dimensional graph                       Discovering objects and their relations from entangled scene representations.
     convolutional networks. In: Proceedings of SDM, pp. 657–665.                                         Proceedings of ICLR.
Mallat, S., 1999. A Wavelet Tour of Signal Processing. Elsevier.                                     Rhee, S., Seo, S., Kim, S., 2018. Hybrid Approach of Relation Network and Localized
Manessi, F., Rozza, A., Manzo, M., 2020. Dynamic graph convolutional networks. Pattern                    Graph Convolutional Filtering for Breast Cancer Subtype Classiﬁcation. In:
     Recogn. 97, 107000.                                                                                  Proceedings of IJCAI, pp. 3527–3534.
Marcheggiani, D., Titov, I., 2017. Encoding sentences with graph convolutional networks              Riba, P., Fischer, A., Llados, J., Fornes, A., 2018. Learning graph distances with message
     for semantic role labeling. In: Proceedings of EMNLP, pp. 1506–1515.                                 passing neural networks. In: Proceedings of ICPR. IEEE, pp. 2239–2244.
Marcheggiani, D., Bastings, J., Titov, I., 2018. Exploiting semantics in neural machine              Rossi, A., Tiezzi, M., Dimitri, G.M., Bianchini, M., Maggini, M., Scarselli, F., 2018.
     translation with graph convolutional networks. Proceedings of NAACL 486–492.                         Inductive–transductive learning with graph neural networks. In: IAPR Workshop on
Marino, K., Salakhutdinov, R., Gupta, A., 2017. The more you know: using knowledge                        Artiﬁcial Neural Networks in Pattern Recognition. Springer, pp. 201–212.
     graphs for image classiﬁcation. Proceedings of CVPR 20–28.                                      Ruiz, L., Chamon, L.F., Ribeiro, A., 2020. Graphon neural networks and the transferability
Maron, H., Ben-Hamu, H., Shamir, N., Lipman, Y., 2019a. Invariant and equivariant graph                   of graph neural networks. In: Proceeding of NeurIPS, 33.
     networks. In: Proceedings of ICLR.                                                              Rusek, K., Suarez-Varela, J., Mestres, A., Barlet-Ros, P., Cabellos-Aparicio, A., 2019.
Maron, H., Fetaya, E., Segol, N., Lipman, Y., 2019b. On the universality of invariant                     Unveiling the potential of graph neural networks for network modeling and
     networks. In: Proceedings of ICML. PMLR, pp. 4363–4371.                                              optimization in sdn. In: Proceedings of SOSR, pp. 140–151.
Masci, J., Boscaini, D., Bronstein, M., Vandergheynst, P., 2015. Geodesic convolutional              Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A.,
     neural networks on riemannian manifolds. ICCV workshops 37–45.                                       Khosla, A., Bernstein, M., et al., 2015. Imagenet large scale visual recognition
Matsunaga, D., Suzumura, T., Takahashi, T., 2019. Exploring Graph Neural Networks for                     challenge. In: Proceedings of IJCV, 115, pp. 211–252.
     Stock Market Predictions with Rolling Window Analysis. arXiv preprint arXiv:                    Sanchez, A., Heess, N., Springenberg, J.T., Merel, J., Hadsell, R., Riedmiller, M.A.,
     1909.10660.                                                                                          Battaglia, P., 2018. Graph networks as learnable physics engines for inference and
Micheli, A., 2009. Neural network for graphs: a contextual constructive approach. IEEE                    control. In: Proceedings of ICML, pp. 4470–4479.
     TNN 20, 498–511.                                                                                Santoro, A., Raposo, D., Barrett, D.G., Malinowski, M., Pascanu, R., Battaglia, P.,
Micheli, A., Sona, D., Sperduti, A., 2004. Contextual processing of structured data by                    Lillicrap, T., 2017. A simple neural network module for relational reasoning.
     recursive cascade correlation. IEEE TNN 15, 1396–1410.                                               Proceedings of NIPS 4967–4976.
Mikolov, T., Chen, K., Corrado, G., Dean, J., 2013. Efﬁcient estimation of word                      Sato, R., Yamada, M., Kashima, H., 2019. Approximation ratios of graph neural networks
     representations in vector space. In: Proceedings of ICLR.                                            for combinatorial problems. Proceedings of NeurIPS, pp. 4081–4090.
Miwa, M., Bansal, M., 2016. End-to-end relation extraction using lstms on sequences and              Scarselli, F., Gori, M., Tsoi, A.C., Hagenbuchner, M., Monfardini, G., 2009. The graph
     tree structures. Proceedings of ACL 1105–1116.                                                       neural network model. IEEE TNN 20, 61–80.
Monti, F., Boscaini, D., Masci, J., Rodola, E., Svoboda, J., Bronstein, M.M., 2017.                  Scarselli, F., Tsoi, A.C., Hagenbuchner, M., 2018. The vapnik–chervonenkis dimension of
     Geometric deep learning on graphs and manifolds using mixture model cnns. In:                        graph and recursive neural networks. Neural Network 108, 248–259.
     Proceedings of CVPR, pp. 5425–5434.                                                             Schlichtkrull, M., Kipf, T.N., Bloem, P., van den Berg, R., Titov, I., Welling, M., 2018.
Morris, C., Ritzert, M., Fey, M., Hamilton, W.L., Lenssen, J.E., Rattan, G., Grohe, M., 2019.             Modeling relational data with graph convolutional networks. In: Proceedings of
     Weisfeiler and leman go neural: higher-order graph neural networks. Proceedings of                   ESWC. Springer, pp. 593–607.
     AAAI 33, 4602–4609.                                                                             Selsam, D., Lamm, M., Bünz, B., Liang, P., de Moura, L., Dill, D.L., 2019. Learning a SAT
Narasimhan, M., Lazebnik, S., Schwing, A.G., 2018. Out of the box: reasoning with graph                   solver from single-bit supervision. In: Proceedings of ICLR.
     convolution nets for factual visual question answering. In: Proceedings of NeurIPS,             Shang, C., Tang, Y., Huang, J., Bi, J., He, X., Zhou, B., 2019. End-to-end structure-aware
     pp. 2654–2665.                                                                                       convolutional networks for knowledge base completion. Proceedings of AAAI 33,
Nguyen, T.H., Grishman, R., 2018. Graph convolutional networks with argument-aware                        3060–3067.
     pooling for event detection. Proceedings of AAAI 5900–5907.                                     Shchur, O., Mumme, M., Bojchevski, A., Günnemann, S., 2018a. Pitfalls of graph neural
Niepert, M., Ahmed, M., Kutzkov, K., 2016. Learning convolutional neural networks for                     network evaluation arXiv preprint arXiv:1811.05868.
     graphs. Proceedings of ICML 2014–2023.                                                          Shchur, O., Zugner, D., Bojchevski, A., Gunnemann, S., 2018b. Netgan: generating graphs
Norcliffebrown, W., Vafeias, S., Parisot, S., 2018. Learning conditioned graph structures                 via random walks. In: Proceedings of ICML, pp. 609–618.
     for interpretable visual question answering. In: Proceedings of NeurIPS,                        Shi, C., Xu, M., Zhu, Z., Zhang, W., Zhang, M., Tang, J., 2020. Graphaf: a Flow-Based
     pp. 8334–8343.                                                                                       Autoregressive Model for Molecular Graph Generation. Proceedings of ICLR.
Nowak, A., Villar, S., Bandeira, A.S., Bruna, J., 2018. Revised note on learning quadratic           Shuman, D.I., Narang, S.K., Frossard, P., Ortega, A., Vandergheynst, P., 2013. The
     assignment with graph neural networks. In: IEEE DSW. IEEE, pp. 1–5.                                  emerging ﬁeld of signal processing on graphs: extending high-dimensional data
Nt, H., Maehara, T., 2019. Revisiting Graph Neural Networks: All We Have Is Low-Pass                      analysis to networks and other irregular domains. IEEE SPM 30, 83–98.
     Filters. arXiv preprint arXiv:1905.09550.                                                       Simonovsky, M., Komodakis, N., 2017. Dynamic edge-conditioned ﬁlters in convolutional
Oono, K., Suzuki, T., 2020. Graph neural networks exponentially lose expressive power                     neural networks on graphs. In: Proceedings of CVPR, pp. 3693–3702.
     for node classiﬁcation. In: Proceedings of ICLR.                                                Socher, R., Chen, D., Manning, C.D., Ng, A., 2013. Reasoning with neural tensor networks
Palm, R., Paquet, U., Winther, O., 2018. Recurrent relational networks. Proceedings of                    for knowledge base completion. In: Proceedings of NIPS, pp. 926–934.
     NeurIPS 3368–3378.                                                                              Song, L., Zhang, Y., Wang, Z., Gildea, D., 2018a. A graph-to-sequence model for amr-to-
Pan, S., Hu, R., Long, G., Jiang, J., Yao, L., Zhang, C., 2018. Adversarially regularized                 text generation. In: Proceedings of ACL, pp. 1616–1626.
     graph autoencoder for graph embedding. Proceedings of IJCAI 2609–2615.                          Song, L., Zhang, Y., Wang, Z., Gildea, D., 2018b. N-ary relation extraction using graph
Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T.,                     state lstm. In: Proceedings of EMNLP, pp. 2226–2235.
     Schardl, T., Leiserson, C., 2020. Evolvegcn: evolving graph convolutional networks              Song, L., Wang, Z., Yu, M., Zhang, Y., Florian, R., Gildea, D., 2018c. Exploring Graph-
     for dynamic graphs. Proceedings of AAAI 34, 5363–5370.                                               Structured Passage Representation for Multi-Hop Reading Comprehension with
Park, J., Lee, M., Chang, H.J., Lee, K., Choi, J.Y., 2019. Symmetric graph convolutional                  Graph Neural Networks arXiv preprint arXiv:1809.02040".
     autoencoder for unsupervised graph representation learning. In: Proceedings of ICCV,            Sperduti, A., Starita, A., 1997. Supervised neural networks for the classiﬁcation of
     pp. 6519–6528.                                                                                       structures. IEEE TNN 8, 714–735.
Peng, N., Poon, H., Quirk, C., Toutanova, K., Yih, W.-t., 2017. Cross-sentence n-ary                 Sukhbaatar, S., Fergus, R., et al., 2016. Learning multiagent communication with
     relation extraction with graph lstms. TACL 5, 101–115.                                               backpropagation. In: Proceedings of NIPS, pp. 2244–2252.
Peng, H., Li, J., He, Y., Liu, Y., Bao, M., Wang, L., Song, Y., Yang, Q., 2018. Large-scale          Sun, Y., Han, J., Yan, X., Yu, P.S., Wu, T., 2011. Pathsim: Meta Path-Based Top-K
     hierarchical text classiﬁcation with recursively regularized deep graph-cnn. In:                     Similarity Search in Heterogeneous Information Networks. Proceedings of the VLDB
     Proceedings of WWW, pp. 1063–1072.                                                                   Endowment, vol. 4, pp. 992–1003.
Peng, Y., Choi, B., Xu, J., 2020. Graph Embedding for Combinatorial Optimization: A                  Sun, L., Wang, J., Yu, P.S., Li, B., 2018. Adversarial Attack and Defense on Graph Data: A
     Survey. arXiv preprint arXiv:2008.12646.                                                             survey. arXiv preprint arXiv:1812.10528.
Perozzi, B., Al-Rfou, R., Skiena, S., 2014. Deepwalk: online learning of social                      Sun, F.-Y., Hoffmann, J., Verma, V., Tang, J., 2020. Infograph: unsupervised and semi-
     representations. In: Proceedings of KDD. ACM, pp. 701–710.                                           supervised graph-level representation learning via mutual information maximization.
Pham, T., Tran, T., Phung, D., Venkatesh, S., 2017. Column networks for collective                        Proceedings of ICLR.
     classiﬁcation. In: Proceedings of AAAI, pp. 2485–2491.                                          Sutton, R.S., Barto, A.G., 2018. Reinforcement Learning: an Introduction. MIT press.
Qi, C.R., Su, H., Mo, K., Guibas, L.J., 2017a. Pointnet: deep learning on point sets for 3d          Tai, K.S., Socher, R., Manning, C.D., 2015. Improved semantic representations from tree-
     classiﬁcation and segmentation. Proceedings of CVPR 652-660.                                         structured long short-term memory networks. In: Proceeding of IJCNLP,
Qi, X., Liao, R., Jia, J., Fidler, S., Urtasun, R., 2017b. 3d graph neural networks for rgbd              pp. 1556–1566.
     semantic segmentation. Proceedings of CVPR 5199–5208.                                           Tang, J., Zhang, J., Yao, L., Li, J., Zhang, L., Su, Z., 2008. Arnetminer: extraction and
                                                                                                          mining of academic social networks. In: Proceedings of KDD. ACM, pp. 990–998.


                                                                                                79
J. Zhou et al.                                                                                                                                                            AI Open 1 (2020) 57–81

Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., Mei, Q., 2015. Line: large-scale                     Yan, S., Xiong, Y., Lin, D., 2018. Spatial temporal graph convolutional networks for
     information network embedding. In: Proceedings of WWW, pp. 1067–1077.                               skeleton-based action recognition. Proceedings of AAAI 32.
Teney, D., Liu, L., Den Hengel, A.V., 2017. Graph-structured representations for visual              Yang, C., Liu, Z., Zhao, D., Sun, M., Chang, E.Y., 2015. Network representation learning
     question answering. In: Proceedings of CVPR, pp. 3233–3241.                                         with rich text information. In: Proceedings of IJCAI, pp. 2111–2117.
Tiezzi, M., Marra, G., Melacci, S., Maggini, M., 2020. Deep Lagrangian Constraint-Based              Yang, Z., Cohen, W., Salakhudinov, R., 2016. Revisiting semi-supervised learning with
     Propagation in Graph Neural Networks. arXiv preprint arXiv:2005.02392.                              graph embeddings. In: Proceedings of ICML. PMLR, pp. 40–48.
Toivonen, H., Srinivasan, A., King, R.D., Kramer, S., Helma, C., 2003. Statistical                   Yang, Y., Wei, Z., Chen, Q., Wu, L., 2019. Using external knowledge for ﬁnancial event
     evaluation of the predictive toxicology challenge 2000–2001. Bioinformatics 19,                     prediction based on graph neural networks. In: Proceedings of CIKM, pp. 2161–2164.
     1183–1193.                                                                                      Yang, C., Xiao, Y., Zhang, Y., Sun, Y., Han, J., 2020. Heterogeneous Network
Toutanova, K., Chen, D., Pantel, P., Poon, H., Choudhury, P., Gamon, M., 2015.                           Representation Learning: Survey, Benchmark, Evaluation, and beyond. arXiv preprint
     Representing text for joint embedding of text and knowledge bases. In: Proceedings                  arXiv:2004.00216.
     of EMNLP, pp. 1499–1509.                                                                        Yao, L., Mao, C., Luo, Y., 2019. Graph convolutional networks for text classiﬁcation.
Tsitsulin, A., Palowitch, J., Perozzi, B., Müller, E., 2020. Graph Clustering with Graph                 Proceedings of AAAI 33, 7370–7377.
     Neural Networks. arXiv preprint arXiv:2006.16904.                                               Ying, R., He, R., Chen, K., Eksombatchai, P., Hamilton, W.L., Leskovec, J., 2018a. Graph
Tu, M., Wang, G., Huang, J., Tang, Y., He, X., Zhou, B., 2019. Multi-hop reading                         convolutional neural networks for web-scale recommender systems. In: Proceedings
     comprehension across multiple documents by reasoning over heterogeneous graphs.                     of KDD Update: 974-983.
     In: Proceedings of ACL, pp. 2704–2713.                                                          Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W., Leskovec, J., 2018b. Hierarchical
van den Berg, R., Kipf, T.N., Welling, M., 2017. Graph convolutional matrix completion.                  graph representation learning with differentiable pooling. Proceedings of NeurIPS
     arXiv preprint arXiv:1706.02263.                                                                    4805–4815.
Vaswani, A., Shazeer, N., Parmar, N., Jones, L., Uszkoreit, J., Gomez, A.N., Kaiser, L.,             Ying, Z., Bourgeois, D., You, J., Zitnik, M., Leskovec, J., Gnnexplainer, 2019. Generating
     2017. Attention is all you need. In: Proceeding of NIPS, pp. 5998–6008.                             explanations for graph neural networks. Proceedings of NeurIPS 9244–9255.
Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., Bengio, Y., 2018. Graph             You, J., Liu, B., Ying, Z., Pande, V., Leskovec, J., 2018a. Graph convolutional policy
     attention networks. In: Proceedings of ICLR.                                                        network for goal-directed molecular graph generation. Proceedings of NeurIPS
Velickovic, P., Fedus, W., Hamilton, W.L., Li  o, P., Bengio, Y., Hjelm, R.D., 2019. Deep               6410–6421.
     graph infomax. In: Proceedings of ICLR.                                                         You, J., Ying, R., Ren, X., Hamilton, W., Leskovec, J., Graphrnn, 2018b. Generating
Verma, S., Zhang, Z.-L., 2019. Stability and generalization of graph convolutional neural                realistic graphs with deep auto-regressive models. Proceedings of ICML 5694–5703.
     networks. In: Proceedings of KDD, pp. 1539–1548.                                                You, J., Ying, Z., Leskovec, J., 2020. Design space for graph neural networks. In:
Vinyals, O., Bengio, S., Kudlur, M., 2015a. Order Matters: Sequence to Sequence for Sets                 Proceedings of NeurIPS, 33.
     arXiv preprint arXiv:1511.06391.                                                                Yu, B., Yin, H., Zhu, Z., 2018. Spatio-temporal Graph Convolutional Networks: A Deep
Vinyals, O., Fortunato, M., Jaitly, N., 2015b. Pointer networks. In: Proceedings of NIPS,                Learning Framework for Trafﬁc Forecasting. Proceedings of IJCAI, pp. 3634–3640.
     pp. 2692–2700.                                                                                  Yun, S., Jeong, M., Kim, R., Kang, J., Kim, H.J., 2019. Graph transformer networks. In:
Wale, N., Watson, I.A., Karypis, G., 2008. Comparison of descriptor spaces for chemical                  Proceedings of NeurIPS, pp. 11983–11993.
     compound retrieval and classiﬁcation. Knowl. Inf. Syst. 14, 347–375.                            Zafarani, R., Liu, H., 2009. Social Computing Data Repository at ASU. http://soci
Wang, H., Leskovec, J., 2020. Unifying Graph Convolutional Neural Networks and Label                     alcomputing.asu.edu.
     Propagation. arXiv preprint arXiv:2002.06755.                                                   Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R.R., Smola, A.J.,
Wang, C., Pan, S., Long, G., Zhu, X., Jiang, J., 2017. Mgae: marginalized graph                          2017. Deep sets. In: Proceedings of NIPS, pp. 3391–3401.
     autoencoder for graph clustering. In: Proceedings of CIKM, pp. 889–898.                         Zang, C., Wang, F., 2020. Neural dynamics on complex networks. Proceedings of KDD,
Wang, X., Girshick, R., Gupta, A., He, K., 2018a. Non-local neural networks. In:                         pp. 892–902.
     Proceedings of CVPR, pp. 7794–7803.                                                             Zayats, V., Ostendorf, M., 2018. Conversation modeling on reddit using a graph-
Wang, Z., Lv, Q., Lan, X., Zhang, Y., 2018b. Cross-lingual knowledge graph alignment via                 structured lstm. TACL 6, 121–132.
     graph convolutional networks. Proceedings of EMNLP 349–357.                                     Zeng, H., Zhou, H., Srivastava, A., Kannan, R., Prasanna, V.K., 2020. Graphsaint: graph
Wang, Z., Chen, T., Ren, J.S.J., Yu, W., Cheng, H., Lin, L., 2018c. Deep reasoning with                  sampling based inductive learning method. In: Proceedings of ICLR.
     knowledge graph for social relationship understanding. Proceedings of IJCAI                     Zhang, D., Yin, J., Zhu, X., Zhang, C., 2018a. Network Representation Learning: A Survey.
     1021–1028.                                                                                          IEEE TBD 6 (1), 3–28.
Wang, X., Ye, Y., Gupta, A., 2018d. Zero-shot recognition via semantic embeddings and                Zhang, Z., Cui, P., Zhu, W., 2018b. Deep Learning on Graphs: A Survey. IEEE TKDE.
     knowledge graphs. Proceedings of CVPR 6857–6866.                                                Zhang, J., Shi, X., Xie, J., Ma, H., King, I., Yeung, D.-Y., 2018c. Gaan: gated attention
Wang, Y., Sun, Y., Liu, Z., Sarma, S.E., Bronstein, M.M., Solomon, J.M., 2018e. Dynamic                  networks for learning on large and spatiotemporal graphs. In: Proceedings of UAI.
     Graph Cnn for Learning on Point Clouds. ACM Transactions on Graphics 38.                        Zhang, Y., Liu, Q., Song, L., 2018d. Sentence-state lstm for text representation. In:
Wang, X., Ji, H., Shi, C., Wang, B., Ye, Y., Cui, P., Yu, P.S., 2019a. Heterogeneous graph               Proceedings of ACL, 1, pp. 317–327.
     attention network. In: Proceedings of WWW, pp. 2022–2032.                                       Zhang, M., Cui, Z., Neumann, M., Chen, Y., 2018e. An end-to-end deep learning
Wang, M., Yu, L., Zheng, D., Gan, Q., Gai, Y., Ye, Z., Li, M., Zhou, J., Huang, Q., Ma, C.,              architecture for graph classiﬁcation. Proceedings of AAAI 32.
     Huang, Z., Guo, Q., Zhang, H., Lin, H., Zhao, J., Li, J., Smola, A.J., Zhang, Z., 2019b.        Zhang, Y., Qi, P., Manning, C.D., 2018f. Graph convolution over pruned dependency trees
     Deep Graph Library: towards Efﬁcient and Scalable Deep Learning on Graphs, ICLR                     improves relation extraction. In: Proceedings of EMNLP, pp. 2205–2215.
     Workshop on Representation Learning on Graphs and Manifolds.                                    Zhang, S., Tong, H., Xu, J., Maciejewski, R., 2019a. Graph convolutional networks: a
Watters, N., Zoran, D., Weber, T., Battaglia, P., Pascanu, R., Tacchetti, A., 2017. Visual               comprehensive review. Computational Social Networks 6 (1), 1–23.
     interaction networks: learning a physics simulator from video. In: Proceedings of               Zhang, C., Song, D., Huang, C., Swami, A., Chawla, N.V., 2019b. Heterogeneous graph
     NIPS, pp. 4539–4547.                                                                                neural network. In: Proceedings of KDD, pp. 793–803.
Wu, Y., Lian, D., Xu, Y., Wu, L., Chen, E., 2020. Graph convolutional networks with                  Zhang, X., Liu, H., Li, Q., Wu, X., 2019c. Attributed graph clustering via adaptive graph
     markov random ﬁeld reasoning for social spammer detection. In: Proceedings of                       convolution. In: Proceedings of IJCAI, pp. 4327–4333.
     AAAI, 34, pp. 1054–1061.                                                                        Zhang, F., Liu, X., Tang, J., Dong, Y., Yao, P., Zhang, J., Gu, X., Wang, Y., Shao, B., Li, R.,
Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., Yu, P.S., 2019a. A Comprehensive Survey on               et al., 2019d. Oag: toward linking large-scale heterogeneous entity graphs.
     Graph Neural Networks arXiv preprint arXiv:1901.00596.                                              Proceedings of KDD 2585–2595.
Wu, F., Souza Jr., A.H., Zhang, T., Fifty, C., Yu, T., Weinberger, K.Q., 2019b. Simplifying          Zhang, J., Zhang, H., Sun, L., Xia, C., 2020. Graph-bert: only attention is needed for
     graph convolutional networks. In: Volume 97 of Proceedings of Machine Learning                      learning graph representations. arXiv preprint arXiv:2001.05140.
     Research. PMLR, pp. 6861–6871.                                                                  Zheng, X., Dan, C., Aragam, B., Ravikumar, P., Xing, E., 2020a. Learning sparse
Wu, Q., Zhang, H., Gao, X., He, P., Weng, P., Gao, H., Chen, G., 2019c. Dual graph                       nonparametric dags. In: Proceedings of AISTATS. PMLR, pp. 3414–3425.
     attention networks for deep latent representation of multifaceted social effects in             Zheng, C., Fan, X., Wang, C., Qi, J., Gman, 2020b. A graph multi-attention network for
     recommender systems. In: Proceedings of WWW, pp. 2091–2102.                                         trafﬁc prediction. Proceedings of AAAI 34, 1234–1241.
Xu, K., Li, C., Tian, Y., Sonobe, T., Kawarabayashi, K., Jegelka, S., 2018. Representation           Zhong, W., Xu, J., Tang, D., Xu, Z., Duan, N., Zhou, M., Wang, J., Yin, J., 2020. Reasoning
     learning on graphs with jumping knowledge networks. In: Proceeding of ICML,                         over semantic-level graph for fact checking. Proceedings of ACL 6170–6180.
     pp. 5449–5458.                                                                                  Zhou, J., Han, X., Yang, C., Liu, Z., Wang, L., Li, C., Sun, M., 2019. Gear: graph-based
Xu, B., Shen, H., Cao, Q., Qiu, Y., Cheng, X., 2019a. Graph wavelet neural network. In:                  evidence aggregating and reasoning for fact veriﬁcation. In: Proceedings of ACL,
     Proceedings of ICLR.                                                                                pp. 892–901.
Xu, K., Hu, W., Leskovec, J., Jegelka, S., 2019b. How powerful are graph neural                      Zhu, D., Zhang, Z., Cui, P., Zhu, W., 2019. Robust graph convolutional networks against
     networks?. In: Proceedings of ICLR.                                                                 adversarial attacks. Proceedings of KDD 1399–1407.
Xu, K., Wang, L., Yu, M., Feng, Y., Song, Y., Wang, Z., Yu, D., 2019c. Cross-lingual                 Zhu, R., Zhao, K., Yang, H., Lin, W., Zhou, C., Ai, B., Li, Y., Zhou, J., 2019a. Aligraph.
     knowledge graph alignment via graph matching neural network. In: Proceedings of                     Proceedings of the VLDB Endowment 12 (12), 2094–2105.
     ACL. Association for Computational Linguistics, pp. 3156–3161.                                  Zhu, Z., Xu, S., Qu, M., Tang, J., 2019b. Graphvite: a high-performance cpu-gpu hybrid
Xu, N., Wang, P., Chen, L., Tao, J., Zhao, J., Gnn, M.R.-, 2019d. multi-resolution and dual              system for node embedding. In: Proceddings of WWW. ACM, pp. 2494–2504.
     graph neural network for predicting structured entity interactions. In: Proceedings of          Zhuang, C., Ma, Q., 2018. Dual Graph Convolutional Networks for Graph-Based Semi-
     IJCAI, pp. 3968–3974.                                                                               supervised Classiﬁcation. Proceedings of WWW, pp. 499–508.


                                                                                                80
J. Zhou et al.                                                                                                                                                     AI Open 1 (2020) 57–81

Zilly, J.G., Srivastava, R.K., Koutnik, J., Schmidhuber, J., 2016. Recurrent highway                Zou, D., Hu, Z., Wang, Y., Jiang, S., Sun, Y., Gu, Q., 2019. Layer-dependent importance
     networks. In: Proceedings of ICML, pp. 4189–4198.                                                  sampling for training deep and large graph convolutional networks. In: Proceedings
Zitnik, M., Leskovec, J., 2017. Predicting multicellular function through multi-layer tissue            of NeurIPS, pp. 11249–11259.
     networks. Bioinformatics 33, i190–i198.                                                        Zügner, D., Akbarnejad, A., Günnemann, S., 2018. Adversarial attacks on neural networks
Zitnik, M., Agrawal, M., Leskovec, J., 2018. Modeling polypharmacy side effects with                    for graph data. In: Proceedings of KDD, pp. 2847–2856.
     graph convolutional networks. Bioinformatics 34, i457–i466.                                    E. Rossi, F. Frasca, B. Chamberlain, D. Eynard, M. Bronstein, F. Monti, 2020. Sign:
                                                                                                        scalable inception graph neural networks, arXiv preprint arXiv:2004.11198.


                                                                                               81
                     Update
                    AI Open
        Volume 6, Issue , 2025, Page 331–332


DOI: https://doi.org/10.1016/j.aiopen.2024.01.002
                                                                  AI Open 6 (2025) 331–332


                                                          Contents lists available at ScienceDirect


                                                                        AI Open
                                          journal homepage: www.keaipublishing.com/en/journals/ai-open


Erratum regarding Declaration of Competing Interest statements in
previously published articles

    Declaration of Competing Interest statements were incorrectly                        personal relationships which may be considered as potential
included in the published version of the following articles that appeared                competing interests: Zhilin Yang is currently employed by Recur­
in previous issues of AI Open.                                                           rent AI.
    The appropriate Declaration of Competing Interest statements, pro­                6. "Lawformer: A pre-trained language model for Chinese legal long
vided by the Authors, are included below.                                                documents"[AI Open, 2021; 79–84] https://doi.org/10.1016/j.
                                                                                         aiopen.2021.06.003Declaration of competing interest: The au­
    1. "On the distribution alignment of propagation in graph neural                     thors declare the following financial interests/personal relation­
       networks"[AI Open, 2022; 3:218–228] https://doi.org/10.1016/j                     ships which may be considered as potential competing interests:
       .aiopen.2022.11.006Declaration of competing interest: The au­                     Cunchao Tu is currently employed by Beijing Powerlaw Intelli­
       thors declare the following financial interests/personal relation­                gent Technology Co., Ltd.
       ships which may be considered as potential competing interests:                7. "Advances and challenges in conversational recommender sys­
       Evgeny Kharlamov is currently employed by Bosch Center for                        tems: A survey"[AI Open, 2021; 2:100–126] https://doi.org/10
       Artificial Intelligence. The research project is funded by                        .1016/j.aiopen.2021.06.002Declaration of competing interest:
       Tsinghua-Bosch Joint ML Center.                                                   The authors declare the following financial interests/personal
    2. "A comprehensive survey of entity alignment for knowledge                         relationships which may be considered as potential competing
       graphs"[AI Open, 2021; 2:1–13] https://doi.org/10.1016/j.                         interests: Maarten de Rijke is currently employed by Ahold
       aiopen.2021.02.002Declaration of competing interest: The au­                      Delhaize.
       thors declare the following financial interests/personal relation­             8. "CokeBERT: Contextual knowledge selection and embedding to­
       ships which may be considered as potential competing interests:                   wards enhanced pre-trained language models"[AI Open, 2021;
       Chengjiang Li is currently employed by Meituan.                                   2:127–134] https://doi.org/10.1016/j.aiopen.2021.06.004Decla­
    3. "Know what you don’t need: Single-Shot Meta-Pruning for                           ration of competing interest: The authors declare the following
       attention heads"[AI Open, 2021; 2:36–42] https://doi.org/10.10                    financial interests/personal relationships which may be consid­
       16/j.aiopen.2021.05.003Declaration of competing interest: The                     ered as potential competing interests: Yankai Lin, Peng Li and Jie
       authors declare the following financial interests/personal re­                    Zhou are currently employed by Tencent Inc.
       lationships which may be considered as potential competing in­                 9. "Pre-trained models: Past, present and future"[AI Open, 2021;
       terests: Qun Liu is currently employed by Huawei Noah’s Ark Lab.                  2:225–250] https://doi.org/10.1016/j.aiopen.2021.08.002Decla­
    4. "Network representation learning: A macro and micro view"[AI                      ration of competing interest: The authors declare the following
       Open, 2021; 2: 43–64] https://doi.org/10.1016/j.aiopen.2021.0                     financial interests/personal relationships which may be consid­
       2.001Declaration of competing interest: The authors declare the                   ered as potential competing interests: Jinhui Yuan is currently
       following financial interests/personal relationships which may be                 employed by OneFlow Inc.
       considered as potential competing interests: Jie Tang is currently            10. "Graph neural networks: A review of methods and applica­
       employed by Tsinghua University, Tsinghua National Laboratory                     tions"[AI Open, 2020; 1: 57–81] https://doi.org/10.1016/j.
       for Information Science and Technology (TNList), Tsinghua-                        aiopen.2021.01.001Declaration of competing interest: The au­
       Bosch Joint ML Center. The research project is funded by                          thors declare the following financial interests/personal relation­
       Tsinghua-Bosch Joint ML Center.                                                   ships which may be considered as potential competing interests:
    5. "WuDaoCorpora: A super large-scale Chinese corpora for pre-                       Lifeng Wang and Changcheng Li are currently employed by
       training language models"[AI Open, 2021; 2:65–68] https://doi.                    Tencent Incorporation. The research project is funded by 2019
       org/10.1016/j.aiopen.2021.06.001Declaration of competing in­                      Tencent Marketing Solution Rhino-Bird Focused Research Pro­
       terest: The authors declare the following financial interests/                    gram FR201908.


    DOIs of original article: https://doi.org/10.1016/j.aiopen.2021.02.001, https://doi.org/10.1016/j.aiopen.2022.11.006, https://doi.org/10.1016/j.aiopen.2021.
06.002, https://doi.org/10.1016/j.aiopen.2021.06.004, https://doi.org/10.1016/j.aiopen.2021.06.001, https://doi.org/10.1016/j.aiopen.2021.02.002, https://doi.
org/10.1016/j.aiopen.2021.05.003, https://doi.org/10.1016/j.aiopen.2021.01.001, https://doi.org/10.1016/j.aiopen.2021.06.003, https://doi.org/10.1016/j.
aiopen.2021.08.002.

https://doi.org/10.1016/j.aiopen.2024.01.002

Available online 9 January 2024
2666-6510/© 2024 The Authors. Published by Elsevier B.V. on behalf of KeAi Communications Co., Ltd. This is an open access article under the CC BY-NC-ND
license (http://creativecommons.org/licenses/by-nc-nd/4.0/).
G. Marotz-Clausen                                                                                                                AI Open 6 (2025) 331–332


   11. “Towards a universal continuous knowledge base”[AI Open,                  12. “Neural machine translation: A review of methods, resources, and
       2021; 2:197–204] https://doi.org/10.1016/j.aiopen.2021.11.00                  tools”[AI Open, 2020; 1: 5–21] https://doi.org/10.1016/j.
       1Declaration of competing interest: The authors declare the                   aiopen.2020.11.001Declaration of competing interest: The au­
       following financial interests/personal relationships which may be             thors declare the following financial interests/personal relation­
       considered as potential competing interests: the research project             ships which may be considered as potential competing interests:
       is funded by Huawei Noah’s Ark Lab.                                           the research project is funded by Huawei Noah’s Ark Lab.


                                                                           332
