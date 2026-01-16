# Dynamic Gnn Survey

> **Note**: This is an auto-converted document from PDF. Some formatting may be imperfect.

---

                                     Foundations and modelling of dynamic
                                     networks using Dynamic Graph Neural
                                     Networks: A survey
                                     JOAKIM SKARDING1 , BOGDAN GABRYS1 , AND KATARZYNA MUSIAL .1
                                     1
                                         Complex Adaptive Systems Lab, Data Science Institute, University of Technology Sydney, Sydney, NSW 2007, Australia
                                     Corresponding author: Joakim Skarding (e-mail: joakim.skarding@uts.edu.au).
                                     This work was supported by the Australian Research Council, “Dynamics and Control of Complex Social Networks” under Grant
                                     DP190101087.
arXiv:2005.07496v2 [cs.SI] 13 Jun 2021


                                           ABSTRACT Dynamic networks are used in a wide range of fields, including social network analysis,
                                           recommender systems and epidemiology. Representing complex networks as structures changing over time
                                           allow network models to leverage not only structural but also temporal patterns. However, as dynamic
                                           network literature stems from diverse fields and makes use of inconsistent terminology, it is challenging to
                                           navigate. Meanwhile, graph neural networks (GNNs) have gained a lot of attention in recent years for their
                                           ability to perform well on a range of network science tasks, such as link prediction and node classification.
                                           Despite the popularity of graph neural networks and the proven benefits of dynamic network models,
                                           there has been little focus on graph neural networks for dynamic networks. To address the challenges
                                           resulting from the fact that this research crosses diverse fields as well as to survey dynamic graph neural
                                           networks, this work is split into two main parts. First, to address the ambiguity of the dynamic network
                                           terminology we establish a foundation of dynamic networks with consistent, detailed terminology and
                                           notation. Second, we present a comprehensive survey of dynamic graph neural network models using the
                                           proposed terminology.


                                           INDEX TERMS Dynamic network models, graph neural networks, link prediction, temporal networks.


                                         I. INTRODUCTION                                                                             part of the title). The second part of this survey (section III
                                     The bulk of network science literature focuses on static                                        and section IV) is narrower in scope and more detailed than
                                     networks, yet every network existing in the real world                                          related surveys, and is a survey on dynamic graph neural
                                     changes over time. In fact, dynamic network structure has                                       networks (referring to the ’using Dynamic Graph Neural
                                     been frequently seen as a complication to be suppressed, to                                     Networks’ part of the title).
                                     ease progress in the study of networks [1]. Since networks                                         Foundations of dynamic networks:
                                     have been used as representations of complex systems in                                            Dynamic networks suffer from a known terminology
                                     fields as diverse as biology and social science, advances in                                    problem [6]. Complex networks which change over time
                                     dynamic network analysis can have a large and far-reaching                                      have been referred to, among others, as; dynamic networks
                                     impact on any field using network analytics [2].                                                [7], [8], temporal networks [2], [9], evolutionary networks
                                        Dynamic networks add a new dimension to network mod-                                         [3] or time-varying networks [10]. With models often work-
                                     elling and prediction – time. This new dimension radically                                      ing only on specific types of networks, a clear and more
                                     influences network properties which enable a more powerful                                      detailed terminology for dynamic networks is necessary. We
                                     representation of network data which in turn increases                                          describe dynamic networks foundations as well as propose
                                     predictive capabilities of methods using such data [3], [4]. In                                 and develop an associated taxonomy of dynamic networks
                                     fact, dynamic networks are not mere generalizations of static                                   to contextualize the models in this survey and enable a more
                                     networks, they exhibit different structural and algorithmic                                     thorough comparison between the models. We are unaware
                                     properties [5].                                                                                 of any work with a comprehensive taxonomy of dynamic
                                        This work is both broader and narrower in scope than                                         networks and therefore it can be considered as the first major
                                     previous works. The first part of this survey (section II)                                      contribution of this paper.
                                     is broader in scope than related surveys and introduces                                            Dynamic networks is a vast and interdisciplinary field.
                                     dynamic networks and dynamic network models (referring                                          Models of dynamic networks are designed by researchers
                                     to the ’foundations and modelling of dynamic networks’                                          from different disciplines and they usually use modelling

                                     VOLUME 9, 2021                                                                                                                                               1
methods from their fields. This survey provides a cross-          representation learning. The distinction is that they survey
disciplinary overview of dynamic network models. This             the broader topic of representation learning on dynamic
overview is not intended to be seen as a dynamic models           networks whereas we survey dynamic graph neural networks
survey, but rather as a context for dynamic graph neural          which is a subset of representation learning on dynamic net-
networks and as a reference point for further exploration of      works. We thus survey a more narrow scope than dynamic
the field of dynamic networks modelling.                          representation learning surveys and a different network type
   We consider a dynamic network to be a network where            from the GNN surveys which focus on static networks
nodes and edges appear and/or disappear over time. Due            [8], [26], [27]. Wu et al. [27] and Zhou et al. [8] also
to the terminology problem establishing a terminology and         survey spatio-temporal graph neural networks, which encode
a clear definition of a dynamic network is a necessity for        spatio-temporal networks (static networks with dynamic
a survey of any kind of dynamic network models such as            node attributes).
dynamic graph neural networks. In the process, we introduce          This survey’s contributions are: (i) A conceptual frame-
a specific and comprehensive terminology that enable future       work and a taxonomy for dynamic networks, (ii) an
works to forego the extensive definition process and simply       overview of dynamic network models, (iii) a survey of
apply our terminology.                                            dynamic graph neural networks (iv) an overview of how
   Related surveys [2], [6], [11] focus either on specific        dynamic graph neural networks are used for prediction of
kinds of dynamic networks, for example, temporal networks         dynamic networks (dynamic link prediction).
[2], [6] or on specific types of models, for example, repre-         This work follows the encoder-decoder framework used
sentation learning [11]–[13]. We are unaware of any work          by Hamilton et al. [28] and is split into three distinct sections
which gives as complete a picture of dynamic networks and         each building upon the previous one.
dynamic network models as we do. The first section is thus           1) Section II is a discussion on dynamic networks. It
broader in scope than other surveys that focus on only one              serves as a foundation to the following sections. In
network type or one type of network model.                              this section we explore different definitions of links
   Modelling dynamic networks using Dynamic Graph                       and introduce a novel dynamic network taxonomy. We
Neural Networks: A dynamic graph neural network                         also give a brief overview of the dynamic network
(DGNN) is considered to be a neural network architecture                model landscape, which contextualizes the rest of the
that can encode a dynamic network and where the aggre-                  survey.
gation of neighbouring node features is part of the neural           2) Section III is a survey of the deep learning models
network architecture. DGNNs encode both structural and                  for encoding dynamic network topology. This covers
temporal patterns in dynamic networks. To encode structural             dynamic network encoders.
patterns DGNNs often make use of a graph neural network              3) Section IV is an overview of how the encoders from
(GNN) and for temporal patterns, they tend to use time                  section III are used for prediction. This includes dy-
series modules such as recurrent neural networks (RNN)                  namic network decoders, loss functions and evaluation
or positional attention. Spatio-temporal networks (graphs               metrics.
where the topology is static and only node or edge features
change [14]) are out of the scope of this survey and thus so      II. DYNAMIC NETWORKS
are Spatio-temporal graph neural networks [14], [15].             A complex network is a representation of a complex system.
   DGNNs, like GNNs and other representation learning             A network that changes over time can be represented as a
models, are versatile in which tasks they can be applied to.      dynamic network. A dynamic network has both temporal
With different decoders and different data, different tasks are   and structural patterns, and these patterns are described by
possible. In practice, so far DGNNs have been applied to          a dynamic network model.
similar tasks as GNNs, the most common of these tasks are            The definition of a link is essential to any network repre-
node classification [16]–[19] and link prediction [16], [18]–     sentation. It is even more essential in dynamic networks, as
[20], which both have diverse and interesting application         it dictates when a link appears and disappears. Different
across many disciplines. Link prediction may for example          link definitions affect network properties which in turn
be applied in knowledge graph completion [21], [22] or            affect which models are capable of representing the dynamic
by recommender systems [18], [19]. DGNNs have also                network.
been used for novel tasks such as predicting path-failure            Dynamic networks are complex networks that change
in dynamic graphs [23], quantifying scientific impact [24],       over time. Links and nodes may appear and disappear.
and detecting dominance, deception and nervousness [25].          With only this insight we can form a general definition for
   There are several surveys on graph neural networks [8],        dynamic networks. Our definition is inspired by Rossetti and
[26], [27] as well as surveys on network representation           Cazabet [30].
learning [28], [29], our work differs from theirs as we              Definition 1 (Dynamic Network) A Dynamic Network
cover GNNs which encode dynamic networks. Kazemi et               is a graph G = (V, E) where: V = {(v, ts , te )}, with v
al. [11], Xie et al. [12] and Barros et al. [13] are the works    a vertex of the graph and ts , te are respectively the start
most similar to this paper as they survey dynamic network         and end timestamps for the existence of the vertex (with
2                                                                                                                     VOLUME 9, 2021
ts ≤ te ). E = {(u, v, ts , te )}, with u, v ∈ V and ts , te are         network. The most straightforward example of this is
respectively the start and end timestamps for the existence              a static network with the edges labelled with the time
of the edge (with ts ≤ te ).                                             they were last active.
   This definition and any of the later definitions represent         • Discrete networks are represented in discrete time in-
unlabeled and undirected networks, but they can however                  tervals. These can be represented by multiple snapshots
trivially be extended with both direction and labels taken               of the network at different time intervals.
into account.                                                         • Continuous networks have no temporal aggregation
   Whereas dynamic networks are defined as complex net-                  applied to them. This representation carries the most
works where links and nodes may appear and disappear,                    information but is also the most complex.
dynamic network models are often designed to work on                  Static and edge-weighted networks are used to model
specific kinds of dynamic networks and specific dynamic            stable patterns or the actual state of the network, whereas
network representations. It, therefore, makes sense to dis-        discrete and continuous methods are used for more dynamic
tinguish between different kinds of dynamic networks and           modelling [30]. This work focuses on dynamic networks and
how they are represented.                                          will therefore only cover discrete and continuous represen-
   Table 7 an overview of the notation and Table 8 is an           tations.
overview of the abbreviations used in this work.                      Fine-grained representations can be trivially aggregated
   There are several surveys on dynamic network methods            to produce coarser representations. For example, links in a
[2], [3], [6], [11], [30]–[35]. These surveys focus either         continuous representation can be aggregated into snapshots
on specific kinds of dynamic networks or on a specific             (or time-windows) which is a discrete representation. Any
discipline and limit the scope of the survey to models             discrete representation can combine the snapshots, yielding
in that discipline. To the best of our knowledge there is          an edge-weighted representation and any edge-weighted
no comprehensive survey of dynamic networks, nor does              representation can discard the weights thus yielding a static
any dynamic network model survey present a complete                network.
foundation or framework for dynamic networks. The aim
of this section is to set the stage for the dynamic graph          1) Discrete Representation
neural network survey by creating a conceptual framework           Discrete representations use an ordered set of graphs (snap-
for dynamic networks with more precise terminology and             shots) to represent a dynamic graph.
to add context by giving an overview of methods used for
modelling dynamic network topology.                                                 DG = {G1 , G2 , . . . , GT },             (1)

A. DYNAMIC NETWORK REPRESENTATIONS                                    where T is the number of snapshots. Discrete represen-
                                                                   tations, often simply referred to as "snapshots" is common
Dynamic networks can be represented in different ways
                                                                   for dynamic networks [2], [3], [9]. Using a discrete rep-
and there are advantages and disadvantages inherent to the
                                                                   resentation of the dynamic network allows for the use of
different representation types.
                                                                   static network analysis methods on each of the snapshots.
   Dynamic network representations can be grouped into
                                                                   Repeated use of the static methods on each snapshot can
four distinct levels ordered by temporal granularity: (i)
                                                                   then collectively give insight into the network’s dynamics.
static, (ii) edge-weighted, (iii) discrete, and (iv) continuous
                                                                      There are other approaches that effectively use snap-
networks [36].
                                                                   shots as well. Overlapping snapshots such as sliding time-
                                                                   windows [37] are also used in dynamic network analysis
                                                                   to have less radical change from one network snapshot
                                                                   to the next [38]. Discrete dynamic networks need not be
                                                                   represented as an ordered set of graphs, they may also be
FIGURE 1: Network representations ordered by temporal              represented as a multi-layered network [39] or as a tensor
granularity. Static networks are the most coarse-grained and       [40].
continuous representations are the most fine-grained. With
increasing temporal granularity comes increasing model             2) Continuous Representation
complexity. The figure is inspired by Fig. 5.1 from Rossetti       Continuous network representations are the only represen-
[36]                                                               tations that have exact temporal information. This makes
                                                                   them the most complex but also the representation with the
   Fig. 1 shows those four representations with increasing         most potential. We cover three continuous representations:
model complexity as the model becomes more temporally              (i) the event-based; (ii) the contact sequence; and (iii) the
fine-grained:                                                      graph streams. The first two representations are taken from
   • Static networks have no temporal information.                 the temporal network literature and they are suitable for
   • Edge weighted networks have temporal information              networks where links do not persist for long [2], [6], [9]. The
     included as labels on the edges and/or nodes of a static      third representation, i.e. the graph stream, is used in dynamic
VOLUME 9, 2021                                                                                                                  3
networks where edges persist for longer [3]. The focus in                Which of the above representations is suitable for the
these representations is on when edges are active, with no            network depends on the link duration with the intricacies of
mention of change on nodes. All three representations are             link duration covered in the next section.
described in more detail below:
   1) The event-based representation includes the time                B. LINK DURATION SPECTRUM
      interval at which the edge on a graph is active [9].            Dynamic networks go by many names and sometimes these
      An event is synonymous with a link in this case. It             names indicate specific types of dynamic networks. There
      is a representation for dynamic networks focusing on            is substantial literature on ’temporal networks’ [2], [6], [9]
      link duration. The network is given by a time-ordered           which focuses on highly dynamic networks where links
      list of events which include the time at which the event        may represent events such as human interactions or a single
      appeared and the duration of the event.                         email. On the other hand, there is also literature that refers to
                                                                      slowly evolving networks, where links represent persistent
                EB = {(ui , vi , ti , ∆i ); i = 1, 2, . . .},   (2)   relations [3]. To the best of our knowledge, there are only
                                                                      two works that take note of this distinction, Rossetti and
       where ui and vi is a node pair on which the i-th event
                                                                      Cazabet [30], and Holme [6].
       occurs, ti is the timestamp for when the event starts
       and ∆i is the duration of the event. This is very similar         Rossetti and Cazabet [30] refer to temporal interaction
       to, and serves the same purpose as, the interval graph         and relational networks (our temporal and evolving networks
       [2]. The difference is that the interval graph has the         respectively), but they do not categorize or make a formal
       time at which the event ends while the event-based             distinction between the different networks.
       representation has the duration of the event.                     Holme [6] suggests that temporal networks can be dis-
    2) The contact sequence representation is a simplifi-             tinguished by two requirements: (i) The dynamics on the
       cation of the event-based representation. In a contact,        network being at the same or at a similar time scale as the
       sequence the link is instantaneous and thus no link            dynamics of the network; and (ii) The dynamic network
       duration is provided.                                          is non-trivial at any given time (an instantaneous snapshot
                                                                      yield little to no network structure).
                  CS = {(ui , vi , ti ); i = 1, 2, . . .},      (3)      The distinction manifests itself in networks even when
                                                                      not considering dynamics on the networks, and this work
       It is common to consider event times in real systems           is limited to the dynamics of the network. Therefore we
       instantaneous if the duration of the event is short or         distinguish temporal networks purely based on network
       not important [2], [9]. Examples of systems where this         topology. We use the second requirement noted by Holme
       representation is suitable, include message networks           [6].
       such as text message and email networks.                          This work not only provides a way to distinguish be-
    3) The graph stream representation is used to repre-              tween temporal networks and dynamic networks, but it also
       sent static graphs that are too large to fit in memory         proposes a framework in which all networks of dynamic
       but can also be used as a representation of a dynamic          topology fit. We do this by introducing the link duration
       network [32]. It is similar to the event-based repre-          spectrum.
       sentation, however, it treats link appearance and link
       disappearance as separate events.

                          GS = {e1 , e2 , . . .} ,              (4)
       where ei = (ui , vi , ti , δi ), and ui and vi is the node
       pair on which the i-th event occurs, ti is the time at
       which the event occurs, and δi ∈ {−1, 1} where −1
       represents an edge removal and 1 represents that an
       edge is added.
       The original representation (used for large graphs)
       does not include timestamped information of when an            FIGURE 2: Temporal and evolving networks on the link
       edge is added/removed [32]. Timestamps will have to            duration spectrum. The spectrum go from 0 (links have no
       be added for retrieving temporal information.                  duration) to infinity (links last forever).
       Since graph streams are mostly used to circumvent
       hardware limitations rather than a limitation of net-             Fig. 2 shows different types of networks on the link
       work representations, we will not survey them in detail        duration spectrum. The scale goes from interactions with
       here. For a more in-depth discussion of the graph              no link duration to links that have infinite link duration. No
       streams, we refer the interested reader to [3], [32],          link ever disappears in a network with infinite link duration.
       [34].                                                          Temporal networks reside on the lower end of the link
4                                                                                                                         VOLUME 9, 2021
duration spectrum, whereas evolving networks reside on the                        link requires an action after it has been established
higher end. The distinction is as follows:                                        (termination of contract) to change its state, but also
   • Temporal networks. Highly dynamic networks which                             maintenance (continued work from the employee). This
     are too dynamic to be represented statically. The net-                       network resides in the fuzzy area between temporal and
     work is at any given time non-trivial. These networks                        evolving networks and can be treated as either.
     are studied in the temporal network literature [2], [9].                   • The Internet is an example of the network where we

     Network properties such as degree distribution and                           consider nodes linked if data-packets can flow between
     clustering coefficient cannot be adopted directly from                       nodes. A link tends to persist for a long time once
     static networks and are non-trivial to define. It is more                    established and thus the internet can be thought of as
     natural to think of a link as an event with a duration.                      an evolving network.
   • Evolving networks. Dynamic networks where events                           • Citation networks where links are defined as one paper

     persist for long enough to establish a network struc-                        citing another have the most persisting links. Once a
     ture. An instantaneous snapshot yields a well-defined                        paper cites another paper, the link lasts forever. This
     network. Network properties such as degree distribu-                         leads to a strictly growing network where no edges
     tion and clustering coefficient can be adopted from                          disappear. These networks have the additional special
     static networks and gradually updated. These are the                         characteristic that edges only appear when new nodes
     networks most often referred to when the term dynamic                        appear.
     network is used. Links persist for so long that it is more                  Link definitions influence link duration, which in turn
     natural to think of link appearance as an event and link                 influences a network type. Links can be modified in ways
     disappearance as another event.                                          that alter their link duration (also known as time to live, TTL
   Furthermore, there is one notable special case for each of                 [30]). An email network could define a link as: Actors have
the dynamic network types. These are types of networks that                   once sent an email between each other. This would modify
reside on the extreme ends of the link duration spectrum:                     the email link, which is usually nearly instant in duration
   • Interaction networks. A type of temporal network                         to a link that will never disappear. This modification moves
     where links are instantaneous events. These networks                     the network all the way to the right on the spectrum shown
     are studied in the temporal network literature and often                 in Fig. 2. It transforms an interaction network into a strictly
     represented as contact sequences [2], [9].                               evolving one. Another example of a modification is to use
   • Strictly evolving networks. A type of evolving net-                      a time-window to force forgetting. A time-window can be
     work where events have infinite duration. This implies                   applied to a citation network such that only citations which
     that the links never disappear.                                          occurred during the time-window appear as links. This will
                                                                              move the network to the left on the link duration spectrum.
                                                                              Depending on the size of the time-window the modified
                                                                              network may be either an evolving or a temporal network.
                                                                                 An additional theoretical special case that is not covered
                                                                              by this concept is a network where links may only disappear.
                                                                              This special case may justify another dimension along which
                                                                              dynamic networks should be distinguished.

FIGURE 3: Examples of networks on the link duration                           C. NODE DYNAMICS
spectrum.                                                                     Another distinguishing factor among dynamic networks is
                                                                              whether nodes may appear or disappear. When modelling
  Fig. 3 shows examples of networks on the link duration                      networks, it is sometimes simpler to assume that the number
spectrum.                                                                     of nodes may not change so that the only possible new links
                                                1
  • An email is a nearly instantaneous event , an email                       are links between already existing nodes.
     network can therefore be considered an interaction                          Many evolving network models assume that edges appear
     network.                                                                 as a new node appears. These models include pseudo-
  • Proximity networks are used as an example of a                            dynamic models such as preferential attachment [41], forest
     temporal network in [2]. The link is defined by who is                   fire [42] and GraphRNN [43]. This is fitting for a citation
     close to whom at what time. Links require maintenance                    network where every node is a paper and the edges are cited
     and do not typically last very long.                                     papers, though, in many real-world networks, edges can
  • Employment networks are social networks where links                       appear and disappear regardless of whether nodes appear.
     are formed between employees and employers. The                             With respect to node change, we can distinguish between
                                                                              two kinds of networks.
   1 If you model information propagation then in practice it takes time
from the moment an email is sent until it is read, so that case considering     •   Static where the number of nodes stays static over
the email an instantaneous event is an approximation.                               time; and
VOLUME 9, 2021                                                                                                                             5
TABLE 1: Dynamic network types by node dynamics and                             link duration spectrum (temporal and evolving) from Section
link duration, excluding special cases.                                         II-B and node dynamics (node-dynamic and node-static)
                                               Link duration                    from Section II-C.
                                    Temporal                  Evolving             Additionally, Table 2 presents the suggested terminology
                     Static    Node-static temporal      Node-static evolving
    Node dynamics
                    Dynamic   Node-dynamic temporal    Node-dynamic evolving
                                                                                for each of the dynamic network types. The precise dynamic
                                                                                network term column show the suggested terms for the
                                                                                different network types. These eight types represent domain-
    •Dynamic where the nodes may appear and disappear.                          independent types of dynamic networks.
   A notable special case of node-dynamic networks are the
                                                                                E. DYNAMIC NETWORK MODELS
networks where nodes may only appear:
                                                                                This brief discussion on dynamic network models is in-
   • Growing networks are those where nodes may only ap-
                                                                                tended to give a high-level overview of the dynamic model
     pear. We consider this a special case of node-dynamic
                                                                                landscape without discussing different kinds of models in
     networks.
                                                                                detail. For a detailed discussion, we refer to dedicated
   We are unaware of any real-world networks where nodes                        works. The aim of this section, is to give the reader the
may only disappear. But it should be noted as at least a                        background and context needed to navigate through the field
theoretical special case. Node growing networks on the other                    of dynamic network models.
hand are rather common.                                                            A network model may model a variety of different net-
   Any kind of node dynamics can be combined with any                           work characteristics or dynamics. In this work, we focus on
kind of link duration network. We can thus have, a grow-                        models of dynamic network structure. Many models define
ing evolving network or a node-static temporal network.                         rules for how links are established [41], [42]. The rules are
Similarly to the edge duration spectrum, a node duration                        defined such that a network evolved with those rules express
spectrum could theoretically be established, but it has no                      some desired features. These features are often observed in
direct impact on dynamic network structure and we, there-                       real-world networks and then included in models as a rule.
fore, chose to keep node dynamics a discrete distinction.                       The search for a good dynamic network model is thus also
   The node dynamics is an important consideration when                         a search for accurate rules on link formation.
modelling the network. Some models support node dynam-                             Network models might aim to replicate characteristics like
ics whereas others do not.                                                      node degree distribution or average shortest path between
                                                                                nodes [44]. The models define probabilistic rules for how
D. THE DYNAMIC NETWORK CUBE                                                     links form such that the emerging network has certain
Many models assume that nodes disappear when there are                          distributions of given characteristics observed in real-world
no longer any links connected to such nodes. This scheme                        networks [44]. Some dynamic network models, particularly
can work for evolving networks, but in temporal networks,                       temporal network models, focus on temporal aspects. An
it is common that nodes have no links for the majority of                       example of a temporal characteristic is the distribution of
the time. Thus for a temporal network, it makes sense to                        inter-event times [9].
model node dynamics separately from link dynamics.                                 There are several use cases for network models. They may
   Different aspects of dynamic network representation have                     be used as reference models [2], [6] or as realistic models
been covered in the previous sections. Section II-A defined                     [45]–[47], and depending on their purpose there are several
different dynamic representations ordered by temporal gran-                     tasks the model can be used for. These include:
ularity, section II-B defined network types by link duration                       • Reference models are used in the analysis of static
and section II-C defined network types by node dynamics.                              networks to study the importance and role of structural
This section will consider these previous sections jointly and                        features of static networks. Reference models aim
discuss how the different network types fit together.                                 to preserve some characteristic such as node degree
   Table 3 includes a comprehensive list of the different                             distribution and otherwise create maximally random
dynamic network types. The types are grouped by node                                  networks. The goal is to determine how the observed
dynamic, temporal granularity and link duration type. Types                           network is different from a completely random network
of networks in each group can generally be combined, thus                             with the same characteristics. This approach has been
we can have a continuous node-static temporal network. The                            adapted to temporal networks [2].
three groups can be thought of as dimensions of a space                            • Realistic models aim to replicate the change in the
where different points in the space would represent different                         network as closely as possible. They can be used for
types of dynamic networks.                                                            several tasks such as network prediction [11], [47],
   The 3D network type space resulting from excluding                                 [48] and community detection [30]. Examples include
special cases is visualised in Fig. 4. When excluding special                         probabilistic models such as the dynamic stochastic
cases there are two types of networks along each dimension.                           block model [49] and representation learning based
The nodes are organised along three dimensions: temporal                              models such as E-LSTM-D [47]. Some realistic models
granularity (discrete and continuous) from Section II-A, the                          aim to generate (simulate) realistic networks [43], [50].
6                                                                                                                                 VOLUME 9, 2021
FIGURE 4: The dynamic network cube. The cube is a novel framework that succinctly represents different kinds of dynamic
networks. Each node represents a specific type of dynamic networks. The nodes are organised along three dimensions:
temporal granularity (discrete and continuous) from Section II-A, the link duration spectrum (temporal and evolving) from
Section II-B and node dynamics (node-dynamic and node-static) from Section II-C. The complete list of terminology from
the cube is presented in Table 2.

                                            TABLE 2: Terminology of the dynamic network cube.
                   Node      Temporal granularity       Node dynamics          Link duration   Precise dynamic network term
                      1      Discrete                   Node-static            Evolving        Discrete node-static evolving network
                      2                                                        Temporal        Discrete node-static temporal network
                      3                                 Node-dynamic           Evolving        Discrete node-dynamic evolving network
                      4                                                        Temporal        Discrete node-dynamic temporal network
                      5      Continuous                 Node-static            Evolving        Continuous node-static evolving network
                      6                                                        Temporal        Continuous node-static temporal network
                      7                                 Node-dynamic           Evolving        Continuous node-dynamic evolving network
                      8                                                        Temporal        Continuous node-dynamic temporal network


TABLE 3: Types of dynamic networks along three dimen-                                 networks by modelling the activity of each node [51].
sions. Static networks and edge-weighted networks are not                             Relational event models are continuous-time models for
dynamic networks, but they are included for completeness.                             interaction networks, they define the propensity for a future
If we exclude special cases, we are left with two elements                            event to happen between node pairs.
in each dimension.                                                                       Latent space models and stochastic block models are
  Dimension                                Network types                              generative probabilistic models. Latent space models require
  Temporal granularity       Static, edge-weighted, discrete, continuous              the fitting of parameters with Markov chain Monte Carlo
  Link duration           Interaction, temporal, evolving, strictly evolving
                                      Node-static, node-dynamic,                      (MCMC) methods and are very flexible but scale to only
  Node dynamics
                                 node-appearing, node-disappearing                    a few hundred nodes [52]. Stochastic block models, on the
                                                                                      other hand, scale to an order of magnitude larger networks,
                                                                                      at a few thousand nodes [52].
   We establish a typology of models for dynamic network                                 Stochastic actor oriented models (SAOM) are continuous-
topology. The typology is based on the type of method used                            time models which consider each node an actor and model
to model the network (see Fig. 5).                                                    actor behaviour. SAOMs learn to represent the dependencies
   We group models intended for inference or identifying                              between a network structure, the position of the actor and
statistical regularities under statistical models. These include                      the actor behaviour [53].
dynamic random graph models, probabilistic models, ac-                                   Dynamic network representation learning includes a di-
tivity driven models and relational event models. Random                              verse set of methods that can be used to embed the dynamic
graph models (RGM) and Exponential random graph models                                graph in a latent space. Representation learning on dynamic
(ERGM) are random graph models which produce randomly                                 networks includes models based on tensor decomposition,
connected graphs while following known common network                                 random walks and deep learning. Since latent space models
topology [44]. Activity driven models are fit to interaction                          and stochastic block models also generate variables in a
VOLUME 9, 2021                                                                                                                                    7
FIGURE 5: An overview of dynamic network models with dynamic graph neural networks outlined. Statistical models are
models intended for inference or identifying statistical regularities in dynamic networks. Representation learning models are
models which automatically detect features needed for the intended task. Stochastic actor oriented models are agent-based
models. Dynamic network representation learning consist of shallow (tensor decomposition and random walk based) methods
and deep learning based methods. This work explores dynamic graph neural networks in detail.


latent space they are closely related to dynamic network        learning based models for dynamic networks is a rapidly
representation learning.                                        growing and exciting field, however, no existing survey
   Tensor decomposition is analogous to matrix factorization    focuses exclusively on dynamic graph neural networks
where the extra dimension is time [11]. Random walk             (Kazemi et al. [11], Xie et al. [12] and Barros et al. [13]
approaches for dynamic graphs are generally extensions of       being the closest).
random walk based embedding methods for static graphs              For the models not discussed in section III there are
or they apply temporal random walks [9]. Deep learning          several works describing and discussing them in detail. Ran-
models include deep learning techniques to generate em-         dom reference models for temporal networks are surveyed
beddings of the dynamic network. Deep models can be             in [2] and [6]. For activity-driven models see Perra et al.
contrasted with the other networks representation learning      [51] and for an introduction to the Relational Event Model
models which are shallow models. We distinguish between         (REM) see Butts [57]. See Hanneke et al. [58] for Temporal
two types of deep learning models: (i) Temporal restricted      ERGMs (TERGM) on discrete dynamic networks. Block et
Boltzmann machines and (ii) Dynamic graph neural net-           al. [59] provides a comparison of TERGM and SAOM. Fritz
works. Temporal restricted Boltzmann machines are prob-         et al. [33] provide a comparison of a discrete-time model,
abilistic generative models which have been applied to the      based on the TERGM, and the Relational Event Model
dynamic link prediction problem [4], [54]–[56]. Dynamic         (REM), a continuous-time model. Goldenberg et al. [60]
graph neural networks combine deep time series encoding         survey dynamic network models and their survey include
with the aggregation of neighbouring nodes. Often discrete      dynamic random graph models and probabilistic models.
versions of these models take the form of a combination         Kim et al. [31] surveys latent space models and stochastic
of a GNN and an RNN. Continuous versions of dynamic             block models for dynamic networks. For an introduction to
graph neural networks cannot make direct use of a GNN           SOAM see Snijders et al. [53]. For surveys of representation
since a GNN require a static graph. Continuous DGNNs            learning on dynamic networks see Kazemi et al. [11], Xie et
must therefore modify how node aggregation is done.             al. [12] and Barros et al. [13], and for a survey of dynamic
   A detailed survey of all kinds of dynamic network models     link prediction, including Temporal restricted Boltzmann
is too broad a topic to cover in detail by one survey. Deep     machines, see Divakaran et al. [54].
8                                                                                                                VOLUME 9, 2021
F. DISCUSSION AND SUMMARY                                                models tend to process the entire graph in each snapshot.
We have given a comprehensive overview of dynamic                        In which case the run-time will increase linearly with the
networks. This establishes a foundation on which dynamic                 number of snapshots. The run-time problem is exacerbated
network models can be defined and thus sets the stage for                by the fact that a lot of real-world graphs are huge which
the survey on dynamic graph neural networks. Establishing                make the run-time on each snapshot significant.
this foundation included the introduction of a new taxonomy                 Continuous representations offer superior temporal gran-
for dynamic networks and an overview of dynamic network                  ularity and thus theoretically a higher potential to model
models.                                                                  dynamic networks. However, continuous-time models tend
   Section II-A presents representations of dynamic net-                 to be more complex and require either completely new
works and distinguishes between discrete and continuous                  models or significant changes to existing ones to work on
dynamic networks. In section II-B we introduce the link                  the continuous representation. Continuous models are less
duration spectrum and distinguish between temporal and                   common than discrete-time models [3], [11], [30]. This is
evolving networks, and in section II-C node dynamics is                  likely due to continuous methods being significantly more
discussed, we distinguish between node-static and node-                  difficult to develop than discrete methods [3].
dynamic networks. Section II-D brings together the previous                 When modelling dynamic networks in continuous time
sections to arrive at a comprehensive dynamic network                    it is essential to specify which kind of network is being
taxonomy.                                                                modelled. As models for temporal and evolving networks
   Discrete representations have seen great success in use on            may not be mutually exclusive and many models work on
evolving networks with slow dynamics. Graph streams are                  only specific types of networks. In these cases, it might
used on evolving networks that update too frequently to be               be possible to modify the link duration of a network to
represented well by snapshots [3]. Both discrete and contin-             run a model on the network. This modification may come
uous representations are used to represent temporal networks             at the loss of information, for example when modifying
[2], [9]. Table 4 combines information from section II-A and             an interaction network to a strictly evolving network, any
section II-B and summarizes the existing representations in              reappearing link will be removed.
terms of temporal granularity and link duration.                            This entire background section establishes a foundation
TABLE 4: Suitable dynamic network representations for                    and a conceptual framework in which dynamic networks
temporal and evolving networks.                                          can be understood. By providing an overview of dynamic
                                                                         network models, it maps out the landscape around deep
  Temporal granularity      Temporal network          Evolving network
                         Event-based representation                      learning on dynamic graphs thus providing the necessary
      Continuous                                       Graph stream
                           or Contact sequence                           context. The following sections will explore dynamic graph
        Discrete              Time-windows               Snapshots       neural networks in detail.
   Discrete representations have several advantages. A
                                                                         III. DYNAMIC GRAPH NEURAL NETWORKS
model which works on the static network case can be
extended to dynamic networks by applying it on each                      Network representation learning and Graph Neural Net-
snapshot and then aggregating the results of the model                   works (GNN) have seen rapid progress recently and they
[11], [31]. This makes it relatively easy, compared to the               are becoming increasingly important in complex network
continuous representation to design dynamic network mod-                 analysis. Most of the progress has been done in the context
els. Furthermore, the distinction between an evolving and a              of static networks, with some advances being extended to
temporal network is less important. If modelling a temporal              dynamic networks. Particularly GNNs have been used in
network, one only needs to make sure that a time-window                  a wide variety of disciplines such as chemistry [61], [62],
size is large enough that the network structure emerges in               recommender systems [63], [64] and social networks [65],
each snapshot. However, the discrete representations have                [66].
their disadvantages too. Chief among them is coarse-grained                 GNNs are deep neural network architectures that encode
temporal granularity. When modelling a temporal network                  graph structures. They do this by aggregating features of
the use of a time-window is a must. By using a time-window               neighbouring nodes together. One might think of this node
the appearance order of the links and temporal clustering                aggregation as similar to the convolution of pixels in con-
(links appearing frequently together) is lost.                           volutional neural networks (CNN). By aggregating features
   Reducing the size of the time-window or the interval                  of neighbouring nodes together GNNs can learn to encode
between snapshots is a way to increase temporal granularity.             both local and global structure.
There are however some fundamental problems with this. In                   Several surveys exist of works on static graph represen-
the case of a temporal network, a small time-window will                 tation learning [29], [67] and static graph neural networks
eventually yield a snapshot with no network structure. In                [8], [26], [27]. Time-series analysis is relevant for work on
the case of an evolving network, we will have a sensible                 dynamic graphs, thus recent advances in this domain is of
network no matter how small the time-window, however,                    relevance. For and up to date survey of deep learning on
there is a trade-off with run-time complexity. Discrete                  time series we refer to Fawaz et al. [68].
VOLUME 9, 2021                                                                                                                       9
FIGURE 6: An overview of the different types of dynamic graph neural networks. This is an extension of Fig 5 where
we zoom in on graph neural networks. Different models are first grouped by which type of network they encode (pseudo-
dynamic, edge-weighted, discrete or continuous). Discrete models are grouped by whether the structural layers and temporal
layers are stacked, or integrated into one layer. Continuous models are grouped by how they encode temporal patterns.


   If dealing with an evolving graph, a static graph algorithm    work structure is learned using other methods than node
can be used to maintain a model of the graph. Minor changes       aggregation (temporal random walks for example), are not
to the graph would most likely not change the predictions of      considered DGNNs.
a static model too much, and the model can then be updated           The previous section (Section II) introduced a framework
at regular intervals to avoid getting too outdated. We suspect    for dynamic networks and an overview of dynamic network
that a spatial GNN is likely to stay accurate for longer than a   models. The overview presented in Fig. 5 shows dynamic
spectral GNN, since the spectral graph convolution is based       graph neural networks to be a part of deep representation
on the graph laplacian which will go through more changes         learning, which in turn is part of dynamic network repre-
than the local changes in a spatial GNN.                          sentation learning. We further extend the overview in Fig.
                                                                  5 to show a hierarchical overview of dynamic graph neural
   It is important to define what we mean by a dynamic
                                                                  networks, Fig. 6.
graph neural network (DGNN). Informally we can say that
a DGNN is a neural network that encodes a dynamic                   An overview of the types of DGNN encoders is seen in
graph. However, there are some representation learning            Fig. 6. The encoders are grouped first by which type of net-
models for dynamic graphs using deep methods, which we            work they encode, then by model type. The pseudo-dynamic
do not consider dynamic graph neural networks. A key              approaches model a network with changing topology, but
characteristic of a graph neural network is an aggregation of     not time. Discrete DGNNs model discrete networks and
neighbouring node features (also known as message passing)        continuous DGNNs model continuous networks. A discrete
[8]. Thus, if a deep representation learning model aggregates     DGNNs encode the network snapshot by snapshot and
neighbouring nodes as part of its neural architecture we call     encode a snapshot all at once, similar to how a GNN encode
it a dynamic graph neural network. In the discrete case,          a static network. A continuous DGNN iterate over the
a DGNN is a combination of a GNN and a time series                network edge by edge and is thus completely independent
model. Whereas in the continuous case we have more variety        of any snapshot size.
since the node aggregation can no longer be done using              Common to all DGNNs is that the encoders aim to capture
traditional GNNs. Given this definition of representation         both structural and temporal patterns and store these patterns
learning, network models where RNNs are used but net-             in embeddings. A stacked DGNNs separate encoding of
10                                                                                                                  VOLUME 9, 2021
structural and temporal patterns in separate layers, having     DGNNs and covers how embeddings are encoded. The next
one layer for structural patterns (using a static GNN) and      section (Section IV) covers decoding of the embeddings.
one layer for temporal patterns (often using some form of an
RNN), these models often make use of existing layers and
                                                                A. PSEUDO-DYNAMIC MODELS
combine them in new ways to encode dynamic networks.
Integrated DGNNs combine structural and temporal patterns       Goldenberg et al. [60] refer to network models as "pseudo-
in one layer. This means that integrated DGNNs require the      dynamic" when they contain dynamic processes, but the
design of new layers, not just a combination of existing        dynamic properties of the model are not fit to the dynamic
layers. The continuous DGNNs consist of RNN, Temporal           data. A well-known example of a non-DGNN pseudo-
point process (TPP) and time embedding based methods.           dynamic model is the Barabasi-Albert model [45].
   A timeline of dynamic network models with a focus on            G-GCN [79] can be seen as an extension of the Varia-
DGNNs is shown in Fig. 7. The timeline includes the first       tional Graph Autoencoder (VGAE) [80] which is able to
appearance of each of the models found in Fig. 5, significant   predict links for nodes with no prior connections, the so-
network embedding models preceding DGNNs and DGNNs.             called cold start problem. It uses the same encoder and
   We consider the Albert-Barabasi model [45] the first         decoder as VGAE, namely a GCN [75] for encoding and
dynamic network model, although it is only a pseudo-            the inner product between node embeddings as a decoder.
dynamic model (see section III-A). The Dynamic Social           The resulting model learns to predict links of nodes that
Network in Latent space" (DSNL) model [70] is the first         have only just appeared.
dynamic latent space model [31]. The Temporal Exponential
Random Graph Model (TERGM) [58] a type of dynamic               B. EDGE-WEIGHTED MODELS
random graph model was introduced in 2009. Snijders et al.
introduced Stochastic Actor Oriented Models (SAOM) [53]         As noted earlier in Section II-A, dynamic network represen-
for dynamic networks in 2010. The first dynamic stochastic      tations can be simplified. One way to simplify the modelling
block model (DSBM) was introduced by Yang et al. [71].          is to convert the dynamic network to an edge-weighted
The first restricted boltzmann machine (RBM) for static         network and then use a static GNN on the edge-weighted
social networks [56] in 2013 was shortly followed by the        network. This is exactly what Temporal Dependent GNN
first RBM for dynamic networks, the Temporal Restricted         (TDGNN) does [81]. They convert an interaction network
Boltzmann Machine (TRBM) in 2014.                               to an edge weighted network by using an exponential
   Prior to DGNNs there were several influential static         distribution. An edge which appeared more recently gets
embedding methods and graph neural networks. The first          a high weight and one that appeared long ago gets a
GNN [72] was introduced in 2008. Deepwalk [73], a highly        low weight. After the conversion an standard GCN [75] is
influential node embedding fueled by random walks was           applied to the edge-weighted network. While the conversion
introduced in 2014. Some Graph Convolutional Neural net-        from interaction network (a continuous network) to edge-
works (GCN) [74], [75] which function as building blocks        weighted is done as part of the model in the original work,
and inspiration for several DGNNs were released in 2016.        there appears to be is no reason why it cannot be done as
   The first DGNNs were discrete DGNNs. First (GCRN-            a pre-processing step and thus we classify it as an edge-
M1 & GCRN-M2) was introduced by Seo et al. [69],                weighted model.
followed by Manessi et al. [76] a few months later. Know-
Evolve [21] a TPP based model was the first continuous          C. DISCRETE DYNAMIC GRAPH NEURAL NETWORKS
model, which in turn directly inspired DyREP [48] by the        Modelling using discrete graphs has the advantage that static
same author. JODIE [77] is notable as the RNN based             graph models can be used on each snapshot of the graph.
DGNN, and it was quickly followed by Streaming GNN [78]         Discrete DGNNs use a GNN to encode each graph snapshot.
which was the first DGNN for continuous strictly evolving       We identify two kinds of discrete DGNNs: Stacked DGNNs
networks. DySAT [17] introduced the first discrete DGNN         and Integrated DGNNs.
which was based solely on attention, thus not using an
RNN. EvolveGCN [16] introduced the first design that had           Autoencoders use either static graph encoders or DGNN
an RNN feed into a GCN, rather than what the previous           encoders, however since they are trained a little differently
models did, which was to have a GCN feed into an RNN.           from DGNNs and generally make use of (and thus extend)
The first pseudo-dynamic GNN, G-GCN was introduced in           a DGNN encoder they are here distinguished from other
early 2019. TGAT [18] is the first DGNN to encode inter-        models.
event time as a vector, while TGN [19] adds a memory               A discrete DGNN combines some form of deep time-
module to TGAT. HDGNN showed how to use DGNNs                   series modelling with a GNN. The time-series model often
for encoding discrete heterogeneous dynamic networks and        comes in the form of an RNN, but self-attention has also
TDGNN although simple was the first GNN to explicitly           been used.
weight the edges to enable interaction network encoding.           Given a discrete graph DG = {G1 , G2 , . . . , GT } a
   This section surveys DGNNs, identifies different types of    discrete DGNN using a function f for temporal modelling
VOLUME 9, 2021                                                                                                             11
FIGURE 7: Timeline of dynamic graph models and dynamic graph neural networks. The timeline shows the first dynamic
network models of each type of model from Fig 5 and significant representation learning models leading up to the first
DGNN. After the first DGNNs (GCRN-M1 and GCRN-M2 [69]) in Dec 2016, only DGNNs are marked on the timeline.
DGNNs are marked by the month they were first publicised as they appeared in tight succession. The timeline indicates
when a model was first publicized (the timeline may therefore show a different year than that in the citation if the paper
was pre-published)


can be expressed as:                                             model stacks the spectral GCN from [74] and a standard
                                                                 peephole LSTM [82]:
          z1t , . . . , znt = GNN Gt
                                      
                                                           (5)
                        htj = f ht−1  t
                                        
                                 j , zj   for j ∈ [1, n]
                                                                         zt = GNN (Xt )
   where f is a neural architecture for temporal modelling                i = σ (Wi zt + Ui ht−1 + wi      ct−1 + bi )
(in the methods surveyed f is almost always an RNN but can
also be self-attention), zit ∈ Rl is the vector representation           f = σ (Wf zt + Uf ht−1 + wf        ct−1 + bf )
of node i at time t produced by the GNN, where l is the                  ct = ft     ct−1                                       (7)
output dimension of the GNN. Similarity hti ∈ Rk is the                       + it     tanh (Wc zt + Uc ht−1 + bc )
vector representation produced by f , where k is the output               o = σ (Wo zt + Uo ht−1 + wo       ct + bo )
dimension of f .
   This can also be written as:                                         ht = o       tanh (ct )

                     Z t = GNN Gt
                                       
                                                           (6)      Let Xt ∈ Rn×d , W ∈ Rk×nl , U ∈ Rk×k and
                     H t = f H t−1 , Z t
                                         
                                                                 h, w, c, b, i, f, o ∈ Rk . The gates which are normally vectors
   Informally we can say that the GNN is used to encode          in the LSTM are now matrices. Also, zt ∈ Rnl×1 is a vector
each network snapshot and f (the RNN or self-attention)          and not a matrix. Even though the GNN used by Seo et al.
encodes across the snapshots.                                    [69] can output features with the same structure as the input,
   Seo et al. [69] introduce two deep learning models which      they reshaped the matrix into a vector. This allows them to
encode a static graph with dynamically changing attributes.      use a one-dimensional LSTM to encode the entire dynamic
Whereas the modelling of this kind of graph is outside the       network.
scope of the survey, the two models they introduced are, to         Whereas [69] use a spectral GCN and a peephole LSTM
the best of our knowledge, the first DGNNs. They introduce       this is not a limitation of the architecture as any GNN and
both a stacked DGNN and an integrated DGNN: (i) Graph            RNN can be used. Other examples of stacked DGNNs are:
Convolutional Recurrent Network Model 1 (GCRN-M1) and            RgCNN [83] which use the Spatial GCN, PATCHY-SAN
(ii) GCRN model 2 (GCRN-M2) respectively. Very similar           [84] stacked with a standard LSTM and DyGGNN [85]
encoders have been used in later publications for dynamic        which uses a gated graph neural network (GGNN) [86]
graphs.                                                          combined with a standard LSTM.
                                                                    Manessi et al. [76] present two stacked DGNN en-
1) Stacked Dynamic Graph Neural Networks                         coders: Waterfall Dynamic-GCN (WD-GCN) and Concate-
The most straightforward way to model a discrete dynamic         nated Dynamic-GCN (CD-GCN). These architectures are
graph is to have a separate GNN handle each snapshot of          distinct in that they use a separate LSTM per node (although
the graph and feed the output of each GNN to a time series       the weights across the LSTMs are shared). The GNN in this
component, such as an RNN. We refer to a structure like          case is a GCN [75] stacked with an LSTM per node. The
this as a stacked DGNN.                                          WD-GCN encoder with a vertex level decoder is shown in
   There are several works using this architecture with          Fig. 8. WD-GCN and CD-GCN differ only in that CD-GCN
different kinds of GNNs and different kinds of RNNs. We’ll       adds skip-connections past the GCN. The equations below
use GCRN-M1 [69] as an example of a stacked DGNN. This           are for the WD-GCN encoder.
12                                                                                                                    VOLUME 9, 2021
FIGURE 8: Stacked DGNN structure from Manessi et al. [76]. The graph convolution layer (GC) encode the graph structure
in each snapshot while the LSTMs encode temporal patterns.


                                                                    2) Integrated Dynamic Graph Neural Networks
     Z1 , . . . , Zt = GNN(A1 , X1 ), . . . , GNN(At , Xt )         Integrated DGNNs are encoders that combine GNNs and
                                                              (8)   RNNs in one layer and thus combine modelling of the
                 H = v-LSTMk (Z1 , . . . , Zt )
                                                                    spatial and the temporal domain in that one layer.
   Let A ∈ Rn×n be the adjacency matrix, n be the number               Inspired by convLSTM [94] Seo et al. [69] introduced
of nodes, d be the number of features per node and Xt ∈             GCRN-M2. GCRN-M2 amounts to convLSTM where the
Rn×d be the matrix describing the features of each node at          convolutions are replaced by graph convolutions. ConvL-
time t. Zt ∈ Rn×l where l is the output size of the GNN             STM uses a 3D tensor as input whereas here we are using
and H ∈ Rk×n×t where k is the output size of the LSTMs.             a two-dimensional signal since we have a feature vector for
         v-LSTMk (Z1 , . . . , Zt )) =                              each node.
                     LSTMk (V10 Z1 , . . . , V10 Zt )
                                                     
                                       ..                     (9)
                                                                   ft = σ (Wf ∗G Xt + Uf ∗G ht−1 + wf        ct−1 + bf )
                                       .             
                          LSTMk (Vn0 Z1 , . . . , Vn0 Zt )           it = σ (Wi ∗G Xt + Ui ∗G ht−1 + wi       ct−1 + bi )
where LSTM is a normal LSTM [87] and Vp ∈ Rn is                      ct = ft     ct−1
                                                                                                                             (10)
defined as Vp = δpi where δ is the Kronecker delta.                       + it     tanh (Wc ∗G Xt + Uc ∗G ht−1 + bc )
Due to the v-LSTM layer the encoder can store a hidden               ot = σ (Wo ∗G Xt + Uo ∗G Ht−1 + wo        c t + bo )
representation per node.
                                                                     ht = o      tanh (ct )
   Since a set of snapshots is a time-series, one is not
restricted to the use of RNNs and other works have stacked             where xt ∈ Rn×d , n is the number of nodes and xi is
GNNs with other types of deep time-series models. Sankar            a signal for the i-th node at time t. W ∈ RK×k×l and
et al. [17] present a stacked architecture that consists com-       U ∈ RK×k×k where k is the size of the hidden layer and K
pletely of self-attention blocks. They use attention along the      is the number of Chebyshev coefficients . Wf ∗G xt denotes
spatial and temporal dimensions. For the spatial dimension,         the graph convolution on xt .
they use the Graph Attention Network (GAT) [88] and for                EvolveGCN [16] integrates an RNN into a GCN. The
the temporal dimension, they use a transformer [89]. Wang           RNN is used to update the weights W of the GCN. [16]
et al. [25], [90] stacks a GNN with 1D temporal convolution         name their layer the Evolving Graph Convolution Unit
(TNDCN) similar to the dilated convolution in WaveNet               (EGCU) and present two versions of it: (i) EGCU-H where
[91].                                                               the weights W are treated as the hidden layer of the RNN
   Stacked DGNN architectures also exist for specific types         and (ii) EGCU-O where the weights W are treated as the
of dynamic networks. There is HDGNN [24] for hetero-                input and output of the RNN. In both EGCU-H and EGCU-
geneous dynamic networks and TeMP [22] for knowledge                O, the RNN operate on matrices rather than vectors as in
networks.                                                           the standard LSTM. The EGCU-H layer is given by the
   When encoding graphs one option is to split the graph            following equations, where (l) indicates the neural network
into sub-graphs and use a GNN to project each sub-graph             layer:
as done by Zhang et al. [92] for static GNNs. This approach                                                    
has also been applied to DGNNs by Cai et al. [93], where                              (l)            (l)    (l)
                                                                                   Wt = GRU Ht , Wt−1
they split each snapshot into sub-graphs and use a stacked                                                                (11)
                                                                                   (l+1)                 (l)     (l)
DGNN for anomaly detection.                                                      Ht       = GNN At , Ht , Wt
VOLUME 9, 2021                                                                                                                 13
FIGURE 9: Integrated DGNN structure of EvolveGCN with an EGCU-O layer [16]. The EGCU-O layer constitutes the
GC (graph convolution) and the W-LSTM (LSTM for GC weights). W-LSTM is used to initialize the weights of the GC.


     And the EGCU-O layer is given by the equations:               we classify it as an integrated DGNN, despite the layer itself
                  (l)
                              
                                  (l)
                                                                  being stacked.
               Wt = LSTM Wt−1
                                                   (12)
               (l+1)                 (l)  (l)                      3) Dynamic graph autoencoders and generative models
              Ht      = GNN At , Ht , Wt
                                                                   The Dynamic Graph Embedding model (DynGEM) [98]
The RNN in both layers can be replaced with any other              uses a deep autoencoder to encode snapshots of discrete
RNN, and the GCN [75] can be replaced with any GNN                 node-dynamic graphs. Inspired by an autoencoder for static
given minor modifications.                                         graphs [99] DynGEM makes some modifications to improve
   Other integrated DGNN approaches are similar to GCRN-           computation on dynamic graphs. The main idea is to have
M2. They may differ in which GNN and/or which RNN                  the autoencoder initialized with the weights from the pre-
they use, the target use case or even the kind of graph they       vious snapshot. This speeds up computation significantly
are built for, but the structures of the neural architecture are   and makes the embeddings stable (i.e. no major changes
similar. Examples of these include GC-LSTM [20], LRGCN             from snapshot to snapshot). To handle new nodes the
[23], RE-Net [95] and TNA [96].                                    Net2WiderNet and Net2DeeperNet approaches from [100]
   Chen et al. [20] present GC-LSTM, an encoder very               are used to add width and depth to the encoder and decoder
similar to GCRN-M2. GC-LSTM takes the adjacency matrix             while the embedding layer stays fixed in size. This allows
At at a given time as an input to the LSTM and performs            the autoencoder to expand while approximately preserving
a spectral graph convolution [74] on the hidden layer. In          the function the neural network is computing.
contrast, GCRN-M2 runs a convolution on both the input                Dyngraph2vec [101] is a continuation of the work done
and the hidden layer.                                              on DynGEM. dyngraph2vec considers the last l snapshots
   LRGCN [23] integrates an R-GCN [97] into an LSTM as             in the encoding and can thus be thought of as a slid-
a step towards predicting path failure in dynamic graphs.          ing time-window. The adjacency matrices At , . . . , At+l are
   RE-Net [95] encodes a dynamic knowledge graph by in-            used to predict At+l+1 , it is assumed that no new nodes
tegrating an R-GCN [97] in several RNNs. Other modelling           are added. The architecture comes in three variations: (1)
changes enable them to encode dynamic knowledge graphs,            dyngraph2vecAE, an autoencoder similar to DynGEM ex-
thus extending the use of discrete DGNNs to knowledge              cept that it leverages information from the past to make
graphs.                                                            the future prediction; (2) dyngraph2vecRNN, where the
   A temporal neighbourhood aggregation (TNA) layer [96]           encoder and decoder consist of stacked LSTMs; (3) dyn-
stacks a GCN, a GRU and a linear layer. Bonner et al.              graph2vecAERNN, where the encoder has first a few dense
designs an encoder that stacks two TNA layers, to achieve a        feed-forward layers followed by LSTM layers and the
2-hop convolution and employs variational sampling for use         decoder is similar to dyngraph2vecAE, namely a deep feed-
on link prediction. This architecture is arguably a stacked        forward network.
DGNN, but since the authors define the TNA as one layer,              E-LSTM-D [47] like DynGEM, encode and decode with
14                                                                                                                   VOLUME 9, 2021
dense layers, however, they run an LSTM on the encoded              The Streaming graph neural network [78] maintains a
hidden vector to predict the new embeddings. Although            hidden representation in each node. The architecture consists
trained like an autoencoder, the model aims to perform a         of two components: (i) an update; and (ii) a propagation
dynamic link prediction.                                         component. The update component is responsible for up-
   Hajiramezanali et al. [102] introduce two variational         dating the state of the nodes involved in an interaction and
autoencoder versions for dynamic graphs: the Variational         the propagation component propagates the update to the
Graph Recurrent Neural Network (VGRNN) and Semi-                 involved node’s neighbours.
implicit VGRNN (SI-VGRNN). They can operate on node-                The update and propagation component consist of 3
dynamic graphs. Both models use a GCN integrated into            units each: (i) the interact unit; (ii) the update / propagate
an RNN as an encoder (similar to GCRN-M2 [69]) to keep           unit; and (iii) the merge unit. The difference between the
track of the temporal evolution of the graph. VGRNN uses         update component and the propagation component is thus
a VGAE [80] on each snapshot that is fed the hidden state        the second unit where the update component makes use of
of the RGNN ht−1 . This is to help the VGAE take into            the update unit and the propagate component makes use of
account how the dynamic graph changed in the past. Each          the propagate unit.
node is represented in the latent space and the decoding is         The model maintains several vectors for each node.
done by taking the inner product decoder of the embeddings       Among them are: (i) a hidden state for the source role of the
[80]. By integrating semi-implicit variational inference [103]   node; and (ii) a hidden state of the target role of the node.
with VGRNN they create SI-VGRNN. Both models aim to              This is required to treat source and target nodes differently.
improve dynamic link prediction.                                 The model also contains a hidden state which is based on
   Generative adversarial networks (GAN) [104] have              both the source and target state of the node. The interact unit
proven to be very successful in the computer vision field        and merge units can be thought of as wrappers that handle
[105]. They have subsequently been adapted for dynamic           many node states. The interact unit generates an encoding
network generation as well. GCN-GAN [106] and Dyn-               based on the interacting nodes and this can be thought of
GraphGAN [107] are two such models. Both models are              as an encoding of the interaction. The merge unit updates
aimed towards the dynamic link prediction task. The gen-         the combined hidden state of the nodes based on the change
erator is used to generate an adjacency matrix and the           done to the source and target hidden states by the middle
discriminator tries to distinguish between the generated and     unit.
the real adjacency matrix. The aim is to have the generator,        The middle units and core of the update and propagate
generate realistic adjacency matrices which can be used as       components are the update and the propagate units. The
a prediction for the next time step.                             update unit generates a new hidden state for the interacting
   GCN-GAN use a stacked DGNN as a generator and a               nodes. It is based on a Time-aware LSTM [108], which is
dense feed-forward networks as a discriminator [106] and         a modified LSTM that works on time-series with irregular
DynGraphGAN use a shallow generator and a GCN [75]               time intervals. The propagate unit updates the hidden states
stacked with a CNN as a discriminator [107].                     of the neighbouring nodes. It consists of an attention func-
                                                                 tion f , a time decay function g and a time based filter h.
D. CONTINUOUS DYNAMIC GRAPH NEURAL                               f estimates the importance between nodes, g gauges the
NETWORKS                                                         magnitude of the update based on how long ago it was and
Currently, there are three DGNN approaches to continuous         h is a binary function which filters out updates when the
modelling. RNN based approaches where node embeddings            receiving node has too old information. h has the effect of
are maintained by an RNN based architecture, temporal            removing noise as well as making the computation more
point based (TPP) approaches where temporal point pro-           efficient.
cesses are parameterized by a neural network and time               By first running the update component and afterwards
embedding approaches where positional embedding of the           propagating, information of the edge update is added to the
time is used to represent time as a vector.                      hidden states of the local neighbourhood.
                                                                    The second method is JODIE [77]. JODIE embeds nodes
1) RNN based models                                              in an interaction network. It is however targeted towards
These models use RNNs to maintain node embeddings in             recommender systems and built for user-item interaction
a continuous fashion. A common characteristic for these          networks. The intuition is that with minor modifications this
models is that as soon as an event occurs or there is a          model can work on general interaction networks.
change to the network, the embeddings of the interacting            JODIE uses an RNN architecture to maintain the embed-
nodes are updated. This enables the embeddings to stay           dings of each node. With one RNN for users (RNNu ) and
up to date continuously. There are two models in this            one RNN for items (RNNi ), the formula for each RNN is
category, Streaming graph neural networks (SGNN) [78]            identical except that they use different weights. When an
which encode directed strictly evolving networks and JODIE       interaction happens between a user and an item, each of the
[77] which encodes interaction networks.                         embeddings is updated according to equation 13.
VOLUME 9, 2021                                                                                                               15
                                                                        where hustruct is given by an attention mechanism that
     u(t) = σ (W1u u (t̄) + W2u i (t̄) + W3u f + W4u ∆u )            aggregates embeddings of neighbours of u. The attention
                                                              (13)   mechanism uses an attention matrix S which is calculated
     i(t) = σ W1i i (t̄) + W2i u (t̄) + W3i f + W4i ∆i
                                                       
                                                                     and maintained by the adjacency matrix A and the intensity
   where u(t) is the embedding of the interacting user, i(t)         function λ. In short, the λ parameterises the attention
the embedding of the interacting item, u(t̄) the embedding           mechanism used by the RNN which in turn is used to
of the user just before the interaction and similarly i(t̄) is       parameterise λ. Thus λ influences the parameterisation of
the embedding of the item just before the interaction. The           itself.
superscript on the weights indicates which RNN they are                 With λ well parameterised it serves as a model for
parameters of, so W1u is a parameter of RNNu . f is the              the dynamic network and its conditional intensity function
feature vector of the interaction and ∆u is the time since           can be used to predict link appearance and time of link
the user interacted with an item and similarly for Deltai .          appearance.
   An additional functionality of JODIE is the projection               Latent dynamic graph (LDG) [109] uses Kipf et al.’s
component of their architecture. It is used to predict the           Neural Relational Inference (NRI) model [110] to extend
trajectory of the dynamic embeddings. The model predicts             DyREP. The idea is to re-purpose NRI to encode the
the future position of the user or item embedding and is             interactions on the graph, generate a temporal attention
trained to improve this prediction.                                  matrix which is then used to improve upon self-attention
                                                                     originally used in DyREP.
2) Temporal point process based models                                  Graph Hawkes Network (GHN) [111] is another method
Know-Evolve [21] is the precursor to the rest of the dynamic         that parameterizes a TPP through a deep neural architecture.
graph temporal point process models discussed in this sec-           Similarly to Know-Evolve [21], it targets temporal knowl-
tion. It models knowledge graphs in the form of interaction          edge networks. A part of the architecture, the Graph Hawkes
networks by parameterizing a temporal point process (TPP)            Process, is an adapted continuous-time LSTM for Hawkes
by a modified RNN. With some minor modifications, the                processes [112].
model should be applicable to any interaction network, but
                                                                     3) Time embedding based models
since the original model is specifically for knowledge graphs
we will rather focus on its successor, DyREP [48].                   Some continuous models rely on time embedding methods.
   DyREP uses a temporal point process model which is                This includes using positional encoding to represent the
parameterised by a recurrent architecture [48]. The temporal         time dimension as introduced by Vaswani et al. [89]. An
point process can express both dynamics "of the network"             example of a time embedding method is time2vec [113].
(structural evolution) and "on the network" (node commu-             This is a positional encoding, similar to the transformer but
nication). By modelling this co-evolution of both dynamics           especially focused on encoding temporal patterns. Another
they achieve a richer representation than most embeddings.           example, is the functional time embedding introduced by
   The temporal point process (TPP) is modelled by events            Xu et al. [114] which converts learning temporal patterns to
(u, v, t, k) where u and v are the interacting nodes, t is           the kernel learning problem and learns the kernel function.
the time of the event and k ∈ {0, 1} indicates whether the           They apply classical functional analysis to enable functional
event is a structural evolution, k = 0 (edge added) or a             learning. These time embedding methods are particularly
communication k = 1.                                                 aimed at capturing temporal difference ti − tj , which is
                                                                     of substantial benefit when modelling interaction networks
   The conditional intensity function λ describes the prob-
                                                                     since it enables them to effectively capture inter-event time.
ability of an event happening. λ is parameterised by two
                                                                        Temporal Graph Attention (TGAT) [18] was the first
functions f and g.
                                                                     continuous DGNN to use a time embedding. The authors use
                                                                     the functional time embedding they introduced separately
                         λu,v    u,v
                          k fk (gk (t̄))                      (14)   [114], however when comparing different versions of the
    where t̄ is the time just before the current event, g is a       embedding they end up using a non-parametric version
weighted concatenation of node embeddings z, gku,v (t̄) =            (Equation 16) which is near identical to time2vec [113].
ω Tk · [z u (t̄); z v (t̄)]. f is a modified softplus, fk (x) =
ψk log (1 + exp (x/ψk )), ωk and ψk are four parameters                            
                                                                      Φd (t, t1 ) = cos (ω1 (t − t1 ) + ϕ1 ) , . . . , cos (ωd (t − t1 ) + ϕd )
                                                                                                                                                
                                                                                                                                                    (16)
which enable the temporal point process to be modelled
on two different time scales.                                           Where ωi and ϕi are learned weights and d is the size of
    The TPP is parameterised by an RNN. The RNN incor-               the time embedding.
porates aggregation of local node embeddings, the previous              A TGAT layer concatenates together the node features,
embedding of the given node and an exogenous drive.                  edge features (optional) and time features of each neighbour-
                                                                     ing node as well as the target node. It then applies masked-
            z v (tp ) = σ(W struct hustruct (t̄p )                   attention similar to the attention in GAT [88]. For each layer
                                                              (15)
                       + W rec z v (t̄vp )W t (tp − t̄vp ))          added an additional hop of neighbours is added. The authors
16                                                                                                                                         VOLUME 9, 2021
found 2 layers (2 hops) to be optimal, as additional hops                 TABLE 5: DGNN model types and network types. All
exponentially increase run-time.                                          continuous DGNNs work on specific types of networks,
                                                                          such as directed or knowledge networks, therefore there are
          h                                                               no continuous DGNNs for any general purpose dynamic
             (l−1)
    Z(t) = h̃0     (t) ke0,0 (t0 ) kΦdT (0),                              network.
                 (l−1)                                                       DGNN                            Network type
             h̃1         (t1 ) ke0,1 (t1 ) kΦdT (t − t1 ) ,
                                                                   (17)     model type   Interaction   Temporal   Evolving   Strictly evolving
             ...,                                                            Discrete        Yes         Yes        Yes             Yes
                                                              i>            Continuous       Yes         No         No               No
                 (l−1)
             h̃N         (tN ) ke0,N (tN ) kΦdT (t − tN )

   Z(t) is an entity-temporal feature matrix which include                been discussed by earlier works [19], to the best of our
features of nodes, edges and inter-event time. l is the layer.            knowledge, they have not been implemented in practice.
In line with self-attention Z(t) is linearly projected to obtain             Most methods focus on discrete graphs which enable
the ’query’, ’key’ and ’value’.                                           them to leverage recent advances in graph neural networks.
                                                                          This allows for modelling of diverse graphs, including node-
                          q(t) = [Z(t)]0 WQ                               dynamic graphs, dynamic labels on nodes and due to the
                          K(t) = [Z(t)]1:N WK                      (18)   use of snapshots, temporal networks can also be handled.
                          V (t) = [Z(t)]1:N WV                            Continuous models currently exist for strictly growing net-
                                                                          works and interaction networks. This leaves many classes of
   [Z(t)]0 is the features of the target node (the node we                dynamic graphs unexplored. Since continuous models have
want to compute the embedding for) and [Z(t)]1:N is the                   some inherent advantages over discrete graphs (see section
features of its neighbours. TGAT applies its attention to Z(t)            II-F), expanding the repertoire of dynamic network classes
to obtain h(t), the hidden representation of the node.                    for continuous models, is a promising future direction.
                                                                             All discrete DGNNs use a GNN to model graph topology
                                        q(t)K(t)                          and a deep time-series model, typically an RNN, to model
                   h(t) = softmax(        √      )V (t)            (19)
                                            dk                            the time dependency. Two types of architectures can be
   Finally, the hidden representation is concatenated with the            distinguished: (i) the stacked DGNN and (ii) the integrated
(static) node embedding of the target node, x0 , and passed               DGNN. Different stacked DGNNs only differ in which
to a feed-forward network.                                                spatial and temporal layers are used to stack (which GNN
                                                                          they use and which time series layer), while the integrated
                          (l)                                             DGNNs may differ not only by how they model spatial
                         h̃0 (t) = FFN (h(t)kx0 )                  (20)
                                                                          and temporal patterns but also in how they integrate the
   Temporal Graph Networks (TGN) [19] extends TGAT by                     spatial and temporal modules. Given the same graph, a
adding a memory module. The memory module embeds the                      stacked DGNN would generally have fewer parameters than
history of the node. The memory vector is added to Z(t) in                a typical integrated DGNN (such as GCRN-M2 [69]). Both
Equation 17.                                                              approaches offer great flexibility in terms of which GNN and
                                                                          RNN can be used. They also are rather flexible in that they
E. DISCUSSION AND SUMMARY                                                 can model networks with both appearing and disappearing
Deep learning on dynamic graphs is still a new field,                     edges as well as dynamic labels.
however, there are already promising methods that show                       Discrete models tend to treat every snapshot as a static
the capacity to encode dynamic topology. This section                     graph, thus the complexity of the model is proportional to
has provided a comprehensive and detailed survey of deep                  the size of the graph in each snapshot and the number
learning models for dynamic graph topology encoding.                      of snapshots. Whereas a continuous model complexity is
   The encoders are summarised and compared in Table 6.                   generally proportional to the number of changes in the
Models are listed based on their encoders and the encoders                graph. If a discrete approach creates snapshots using time-
capacity to model link and node dynamics. Any model                       windows, then it can trade off temporal granularity (and thus
which cannot model link deletion or link duration can only                theoretically modelling accuracy) for faster computation by
model strictly evolving networks or interaction networks                  using larger time-windows for each snapshot.
(see section II-B).                                                          Table 6 shows that every continuous DGNN is aimed at
   Table 6 list many models as not supporting link deletion,              a special type of continuous network. This is reflected in
it is possible to model link deletion by link deletion events             Table 5 which shows that there is, as of yet, no continuous
and thus an interaction network can model a persistent                    DGNN encoder for any general-purpose dynamic network.
link disappearing. Any continuous model should also be                       So which one should you chose? Converting the dy-
able to model node deletion by removing the node from                     namic network to an edge-weighted network is a simple,
node neighbourhood aggregation to effectively delete it.                  and depending on the application, possibly "good enough"
However, while these ways of modelling dynamics have                      approach. A practitioner only need to come up with some
VOLUME 9, 2021                                                                                                                                   17
scheme to weight edges, and then feed that to an optimized      diction of the future change to the network topology. Much
implementation of a standard GNN, e.g. GCN [75] or GAT          work has been done on the prediction of missing links in
[88]. TDGNN [81] shows a good example of such a scheme          networks, which can be thought of as an interpolation task.
by weighting the edges using an exponential distribution,       This section explores how dynamic graph neural networks
which weights more recent edges higher than old edges.          can be used for link prediction and deal exclusively with
   Another approach which should be considered before           the extrapolation (future link prediction) task.
trying any large DGNN model is whether applying a static           Predictions can be done in a time-conditioned or time-
GNN on a discrete representation might yield good enough        predicting manner [11]. Time-predicting means that a
results. Given the same number of features and layer size,      method predicts when an event will occur and time-
it will train faster and generally be a simpler model.          conditioned means that a method predicts whether an event
   The choice between discrete and continuous depends on        will occur at a given time t. For example, if the method
the data and the intended problem. If temporal granularity      predicts the existence of a link in the next snapshot, it is a
and performance is not a concern then one of the advanced       time-conditioned prediction. If it predicts when a new link
discrete approaches such as DySAT or EvolveGCN will             between nodes will appear, it is a time-predicting prediction.
likely be a great fit for most dynamic network problems.           Prediction of links often focuses only on the prediction of
Since they naturally support link deletion, node addition and   the appearance of a link. However, link disappearance is less
node deletion, they provide good general-purpose function-      explored but also important for the prediction of network
ality.                                                          topology. We refer to link prediction based on a dynamic
   The Discrete DGNNs covered in this work all iterate over     network as dynamic link prediction.
snapshots to encode, while the continuous DGNNs iterate            For embedding methods, what is predicted and how is
edge-by-edge. The continuous therefore tend to take longer      decided by the decoder. You can have both time-predicting
to train compared to the discrete models. This is especially    and time-conditioned decoders. The prediction capabilities
true if the network is rather dense.                            will depend on the information captured by the embeddings.
   Evolving networks are well served by any discrete ap-        Thus, an embedding that captures continuous-time infor-
proach, however, with the recent dominance of attention         mation has a higher potential to model temporal patterns.
architectures [89], we would expect DySAT to do well in a       Well modelled temporal and structural embeddings offer a
comparative test. EvolveGCN is expected to train fast on an     better foundation for a decoder and thus potentially better
evolving network with little change between snapshots. The      predictions.
discrete methods are also suited for temporal networks given       If dealing with discrete data and few timestamps, a time-
that the length of the time-windows covered by snapshots        conditioned decoder can be used for time prediction. This
is well selected.                                               can be done by applying the time-conditioned decoder to
   If node dynamics is an important feature of the network      every candidate timestamp t and then consider the t where
you wish to model, then you should choose a model that can      the link has the highest probability of appearing.
encode node dynamics such as DySAT [17], EvolveGCN                 The rest of this section is a description of how the
[16] or HDGNN [24].                                             surveyed models from the previous section can be used to
   If you have an interaction network with detailed times-      perform predictions. This includes mainly a discussion on
tamps, then TGAT [18] or TGN [19] are likely good fits. If      decoders and loss functions. Since the surveyed models aim
run-time complexity and time granularity are essential to the   to predict the time-conditioned existence of links, the focus
dynamic complex network at hand (for example in the case        will be on the dynamic link prediction task.
of a temporal network), then non-deep learning methods that        Autoencoders can use the same decoders and loss func-
are not covered by this survey are recommended. Those           tions as other methods. Their aim is typically a little
methods can be explored in the literature referred to in        different. The decoder is targeted at the already observed
section II-E.                                                   network and tries to recreate the snapshot. A prediction
                                                                for a snapshot at time t + 1 is marginally different from
IV. DEEP LEARNING FOR PREDICTION OF NETWORK                     the decoder of an autoencoder which is targeted at already
TOPOLOGY                                                        observed snapshots.
Any embedding method can be thought of as a concate-
nation of an encoder and a decoder [28]. Until now, we          A. DECODERS
have discussed encoders, but the quality of embeddings          Of the surveyed approaches which apply a predicting de-
depend on the decoder and the loss function as well. While      coder, almost all apply a time-conditioned decoder. A pre-
the encoders in Section III can be paired with a variety        diction is then often an adjacency matrix Âτ which indicates
of decoders and loss functions depending on the intended        the probabilities of an edge at time τ . Often τ = t + 1.
task, we focus in this section on one of the most commonly         We consider decoders to be the part of the architecture
tackled problems - link prediction.                             that produces Âτ from Z the dynamic graph embeddings.
   Prediction problems can be defined for many different           Since encoders make node embeddings and predicting a
contexts and settings. In this survey, we refer to the pre-     link involves two nodes decoders tend to aggregate two
18                                                                                                                VOLUME 9, 2021
TABLE 6: Deep encoders for dynamic network topology. While we note which GNNs are used in each of the discrete
models it is usually trivial to replace it with another GNN.
 Model type             Model name           Encoder                                              Link addition   Link deletion   Node addition   Node deletion   Network type
 Discrete networks
 Stacked DGNN           GCRN-M1 [69]         Spectral GCN [74] & LSTM                             Yes             Yes             No              No              Any
                        WD-GCN [76]          Spectral GCN [75] & LSTM                             Yes             Yes             No              No              Any
                        CD-GCN [76]          Spectral GCN [75] & LSTM                             Yes             Yes             No              No              Any
                        RgCNN [83]           Spatial GCN [84] & LSTM                              Yes             Yes             No              No              Any
                        DyGGNN [85]          GGNN [86] & LSTM                                     Yes             Yes             No              No              Any
                        DySAT [17]           GAT [88] & temporal attention [89]                   Yes             Yes             Yes             Yes             Any
                        TNDCN [25], [90]     Spectral GCN [25] & TCN [25]                         Yes             Yes             No              No              Any
                        StrGNN [93]          Spectral GCN [75] & GRU                              Yes             Yes             No              No              Any
                        HDGNN [24]           Spectral GCN [75] & A variety of RNNs                Yes             Yes             Yes             Yes             Heterogeneous
                        TeMP [22]            R-GCN [97] stacked with either GRU or attention      Yes             Yes             No              No              Knowledge
 Integrated DGNN        GCRN-M2 [69]         GCN [74] integrated in an LSTM                       Yes             Yes             No              No              Any
                        GC-LSTM [20]         GCN [74] integrated in an LSTM                       Yes             Yes             No              No              Any
                        EvolveGCN [16]       LSTM integrated in a GCN [75]                        Yes             Yes             Yes             Yes             Any
                        LRGCN [23]           R-GCN [97] integrated in an LSTM                     Yes             Yes             No              No              Any
                        RE-Net [95]          R-GCN [97] integrated in several RNNs                Yes             Yes             No              No              Knowledge
                        TNA [96]             GCN [75] stacked with a GRU and a linear layer       Yes             Yes             No              No              Any
 Continuous networks
 RNN based
                                                     Node embeddings maintained by
                        Streaming GNN [78]                                                        Yes             No              Yes             No              Directed strictly evolving
                                                 architecture consisting of T-LSTM [108]
                                                     Node embeddings maintained by
                        JODIE [77]                                                                Yes             No              No              No              Bipartite, interaction
                                                        an RNN based architecture
 TTP based
                        Know-Evolve [21]               TPP parameterised by an RNN                Yes             No              No              No              Interaction, knowledge network
                                                       TPP parameterised by an RNN
                        DyREP [48]                                                                Yes             No              Yes             No              Interaction combined with strictly evolving
                                                         aided by structural attention
                        LDG [109]                       TPP, RNN and self-attention               Yes             No              Yes             No              Interaction combined with strictly evolving
                                                           TPP parameterised by a
                        GHN [111]                                                                 Yes             No              No              No              Interaction, knowledge network
                                                        continuous time LSTM [112]
 Time embedding based
                                                    Temporal [113] and structural [88]
                        TGAT [18]                                                                 Yes             No              Yes             No              Directed or undirected interaction
                                                                attention
                                                    Temporal [113] and structural [88]
                        TGN [19]                                                                  Yes             No              Yes             No              Directed or undirected interaction
                                                        attention with memory


node embeddings to predict a link. The simplest way to                                                   and the architecture can be efficiently optimized with back-
aggregate is to apply an operator, e.g. the inner product [80]                                           propagation.
(shown in Equation 21), concatenation, mean or Hadamard                                                     The only surveyed method using a time-predicting de-
product [81]. This combines the node embeddings and gives                                                coder is DyRep [48]. DyRep uses the conditional intensity
a probability of a link appearing. These simple approaches                                               function of its temporal point process to model the dynamic
require that the encoder is able to embed the nodes in a                                                 network.
space such that nodes that are likely to connect are close to                                               While the focus in this section is on decoders that are used
each other or otherwise able to be decoded by the simple                                                 directly for the forecasting task, it is important to note that
decoder.                                                                                                 downstream learning can also be used. This is the DGNN
   Another simple decoder is to use a simple feed-forward                                                trained on a task and the node embeddings are used for a
network. The decoder as before receives two node em-                                                     different task. For example, the DGNN can be trained on
beddings and gives out a probability for whether the link                                                node classification and then the node embeddings are used
appeared or didn’t appear. This approach is used by several                                              later for link prediction. An example of this is seen in [17],
models for link prediction [16], [47], [101]. While this                                                 where a logistic regression classifier is trained on the node
requires more parameters, the decoder is can easily be                                                   embeddings of snapshot t to predict links at t + 1.
dwarfed in size by the encoder and it enables decoding of
non-linear relationships between node embeddings.                                                         B. LOSS FUNCTIONS
                                                                                                         The loss function is central to any deep learning method,
                     p Atij = 1|zit , zjt = σ (zit )> zjt                                                as it is the equation that is being optimized. Regarding loss
                                                         
                                                                                           (21)
                                                                                                         functions, we can make a distinction between (i) link pre-
   Where zk is the node embedding of node k. An inner                                                    diction optimizing methods; and (ii) autoencoder methods.
product decoder works well if we only want to predict or                                                 As the prediction methods optimize towards link prediction
reproduce the graph topology. If we would like to decode                                                 directly, an autoencoder optimizes towards the recreation
the feature matrix then a neural network should be used                                                  of the dynamic graph. Despite have slightly different aims,
[102].                                                                                                   both approaches have been used for link prediction and have
   Wu et al. [115] uses GraphRNN, a deep sequential                                                      been shown to perform well.
generative model as a decoder [43]. What is unique with
GraphRNN is that it reframes the graph generation problem                                                1) Link prediction
as a sequential problem. The GraphRNN authors claim                                                      Prediction of edges is seen as a binary classification task.
increased performance over feed-forward auto-encoders.                                                   Traditional link prediction is well known for being ex-
   In general, there are many options for how decoding can                                               tremely unbalanced [52], [116]. For predicting methods the
be done. A decoder might be viable as long as the prob-                                                  loss function is often simply the binary cross-entropy [16],
ability for each edge is produced from the latent variables                                              [17], [85].
VOLUME 9, 2021                                                                                                                                                                                              19
   Some models use negative sampling [16], [17]. This               where P is the set P  of observed
                                                                                                Pn events,    λ is the intensity
                                                                                            n
                                                                 function and Λ(τ ) = u=1 v=1 k∈{0,1} λu,v
                                                                                                       P
transforms the problem of link prediction from a multiple                                                          k (τ ) is the
output classification (a prediction for each link) to a binary   survival probability for all events that did not happen.
classification problem (is the link a "good" link or a "bad"     Survival probability indicates the probability of an event not
link). This speeds up computation and deals with the well-       happening [117]. The first term thus rewards a high intensity
known class imbalance problem in link prediction. The            when an event happens, whereas the second term rewards a
rate of negative samples used vary from work to work,            low intensity (high survival) of events that do not happen.
EvolveGCN [16] use 1 to 100 for training, while TGAT                Trivedi et al. [48] further identify that calculating the
[18] and TGN [19] use 1 to 1.                                    integral of Λ is intractable. They get around that by sampling
                          n X
                            n
                                                                 non-events and estimating the integral using Monte Carlo
                          X                                      estimation, this is done for each mini-batch.
                LCE =               Atij log(Âtij )      (22)
                          i=1 j=1
                                                                 4) Regularization
   Equation 22 is an example of a binary cross entropy loss      There are several different approaches for adding regular-
adapted from [20].                                               ization to loss functions to avoid overfitting. The total loss
   DySAT [17] sums the loss function only over nodes that        function (equation 25) is composed of the reconstruction
are in the same neighbourhood at time t. The neighbour-          loss and the regularization with an optional constant α
hoods are extracted by taking nodes that co-occur in random      to balance the terms. Here we cover the methods that
walks on the graph. The inner product is calculated as a part    use regularization, however many models chose to not use
of the summation in the loss function. This means that the       regularization as they find that they don’t have a problem
inner product will be calculated only for the node pairs that    with overfitting [16], [18], [19], [48].
the loss is computed on. Together it reduces the number of
nodes that are summed up and should result in a training                               Ltotal = L + αLreg                       (25)
speed up. Any accuracy trade-off is not discussed by the
authors.                                                            A common way to regularize is through summing up all
                                                                 the weights of the model, thus keeping the weights small and
2) Autoencoders                                                  the model less likely to overfit. The L2 norm is commonly
Autoencoder approaches [47], [98], [101] aim to reconstruct      used for this [20], [47].
the dynamic network. All surveyed autoencoders operate              The variational autoencoder methods use a different reg-
on discrete networks. Therefore the reconstruction of the        ularizer. They normalize the node embeddings compared to
network is reduced to the reconstruction of each snapshot.       a prior. In traditional variational autoencoders, this prior is a
This entails creating a loss function that penalizes wrong       Normal distribution with mean 0 and standard deviation 1.
reconstruction of the input graph. Variational autoencoder       In dynamic graph autoencoders [79], [102], the prior is still
approaches [79], [102] also aim to be generative models.         a Gaussian, but it is parameterised by previous observations.
To be generative, they need to enable interpolation in           Equation 26 is the regularization term from [102].
latent space. This is achieved by adding a term to the                            KL(q Z t |A≤t , X ≤t , Z <t
                                                                                                               
loss function which penalizes the learned latent variable                                                                    (26)
                                                                                       kp Z t |A<t , X <t , Z <t )
                                                                                                                 
distribution for being different from a normal distribution.
It is also common to add regularization to the loss functions       where q is the encoder distribution and p is the prior
to avoid overfitting.                                            distribution. KL is the Kullback-Leibler divergence which
                   n X
                     n 
                                                                 measures the difference between two distributions. The A<t
                                           
                                                                 indicate all adjacency matrices up to, but not including t and
                   X
              L=               Atij − Âtij ∗ Pij         (23)
                   i=1 j=1
                                                                 similarly for the other matrices. We can see that the prior
                                                                 is influenced by previous snapshots, but not by the current.
   Equation 23 is the reconstruction penalizing component        Whereas the encoder is influenced by the previous and the
of E-LSTM-D’s loss function [47]. P is a matrix which            current snapshot.
increases the focus on existing links. pij = 1 if Atij = 0
and pij = β > 1 if Atij = 1.                                     C. EVALUATION METRICS
                                                                 Link prediction is plagued by high class imbalance. It is
3) Temporal Point Processes                                      a binary classification, a link either exists or not and most
DyRep [48] models a dynamic network by parameterising            links will not exist. In fact, actual links tend to constitute less
a temporal point process. Its loss function influences how       than 1% of all possible links [118]. AUC and precision@k
the temporal point process is optimized.                         are two commonly used evaluation metrics in static link
                   P                      Z T                    prediction [116], [119]. If dynamic link prediction requires
                                                                 the prediction of both appearing and disappearing edges,
                   X
           L=−           log (λp (t)) +         Λ(τ )dτ   (24)
                   p=1                     0                     the evaluation metric needs to reflect that. Furthermore,
20                                                                                                                     VOLUME 9, 2021
traditional link prediction metrics have shortcomings when               also observed in dynamic link prediction [52]. Fixed-
used in a dynamic setting [52].                                          threshold metrics are not recommended unless the
   For a detailed discussion on the evaluation of link predic-           targeted problem has a natural threshold [116].
tion, we refer to Yang et al. [116] for static link prediction        4) Sum of absolute differences (SumD). Li et al. [118]
and Junuthula et al. [52] for dynamic link prediction evalu-             pointed out that models often have similar AUC scores
ation.                                                                   and suggested SumD as a stricter measurement of
                                                                         accuracy. It is simply, the number of mispredicted
  1) Area under the curve (AUC). The area under the                      links. The metric has different meanings depending
     curve (AUC) is used to evaluate a binary classification             on how many values are predicted since it is not
     and has the advantage of being independent of the                   normalized according to the total number of links.
     classification threshold. The AUC is the area under                 Chen et al. considers SumD misleading for this reason
     the receiver operating characteristic (ROC) curve. The              [47]. The metric strictly punishes false positives, since
     ROC is a plot of the true positive rate and the false               there are so many links not appearing, a slightly higher
     positive rate.                                                      rate of false positives will have a large impact on this
     The AUC evaluates predictions based on how well the                 metric.
     classifier ranks the predictions, this provides a mea-           5) Error rate. Since SumD suffers from several draw-
     sure that is invariant of the classification threshold.             backs an extension is suggested by Chen et al. [47].
     In link prediction, there has been little research into             Error rate normalizes SumD by the total number of
     finding the optimal threshold [120], using the AUC                  existing links.
     for evaluation avoids this problem.
     Yang et al. [116] note that AUC can show deceptively
     high performance in link prediction due to the extreme                                               Nfalse
                                                                                           Error Rate =                      (27)
     class imbalance. They recommend the use of PRAUC                                                     Ntrue
     instead.
  2) PRAUC. The PRAUC is similar to the AUC except                       where Nfalse is the number of mispredicted links and
     that it is the area under the precision-recall curve. The           Ntrue is the number of existing links. The error rate
     metric is often used in highly imbalanced information               is very similar to recall, except that recall focuses on
     retrieval problems [52].                                            true positives, where the error rate focuses on false
     PRAUC is recommended by Yang et al. [116] as a                      positives. Another difference between recall and error
     suitable metric for traditional (static) link prediction            rate is that recall is normalized between 0 and 1,
     due to the deceptive nature of the ROC curve and                    while the error rate may be above 1 if the number of
     because PRAUC shows a more discriminative view of                   mispredicted links outnumber the number of existing
     classification performance. And recommended for the                 links The error rate is a good metric if the number
     same reasons by Li et al. for dynamic link prediction               of false positives is a major concern. In dynamic link
     [118].                                                              prediction, false positives become a major issue due to
     One way of calculating the PRAUC is by using Mean                   the massive class imbalance of the prediction problem.
     Average Precision (MAP). MAP is the mean of the                  6) GMAUC. After a thorough investigation of evaluation
     average precision (AP) per node.                                    metrics for dynamic link prediction, Junuthula et
  3) Fixed-threshold metrics. One of the most common                     al. suggests GMAUC as an improvement over other
     fixed threshold metrics in traditional link prediction is           metrics [52]. The key insight is that dynamic link
     Precision@k. It is the ratio of items that are correctly            prediction can be divided into two sub-problems: (i)
     predicted. From the ranking prediction the top k                    predicting the disappearance of links that already exist
     predictions are selected, then precision is the ratio kkr ,         or the appearance of links that have once existed; and
     where kr is the number of correctly predicted links in              (ii) predicting links that have never been seen before.
     the top k predictions.                                              When the problem is divided in this way, each of the
     While a higher precision indicates a higher prediction              sub-problems takes on different characteristics.
     accuracy, it is dependent on the parameter k. k might               Prediction of links that have never been seen before
     be given on web-scale information retrieval, where                  is equivalent to traditional link prediction, for which
     we care about the accuracy of the highest k ranked                  PRAUC is a suitable metric [116]. Prediction of
     articles, in link prediction it is difficult to find the right      already existing links is both the prediction of once
     cut-off [120].                                                      seen links appearing and existing links disappearing.
     A fixed-threshold can be applied to other common                    This is a more balanced problem than traditional link
     metrics including accuracy, recall and F1 among oth-                prediction, thus AUC is a suitable measure. [52] note
     ers [116]. These methods suffer from instability in                 that both the mean and the harmonic mean will lead
     their predictions, where a change of thresholds can                 to either the AUC or the PRAUC to dominate, thus
     lead to contradictory results [116]. This problem is                the geometric mean is used to form a unified metric.
VOLUME 9, 2021                                                                                                                 21
                                                                 a reasonably balanced classification, this is not necessarily
           GMAUC =                                               true and depends on the data. An evaluation of new methods
             v
                            P
                                                                 should report the PRAUC of newly appearing links and the
                                                         (28)
             u
             u PRAUCnew − P +N
             t                 · 2 (AUCprev − 0.5)               PRAUC or AUC of reappearing links separately. The com-
                         P
                   1 − P +N                                      bined score should also be reported as either the PRAUC,
       P RAU Cnew is the PRAUC score of new links,               Error rate or GMAUC.
       AU Cprev is the AUC score of previously observed             Prediction on dynamic networks is in its infancy. Deep
       links.                                                    models are largely focused on unattributed time-conditioned
       The authors note the advantages of GMAUC:                 discrete link appearance prediction. This leaves opportuni-
                                                                 ties for future work in a large range of prediction tasks, with
         • Based on threshold curves, thus avoids the pitfall
                                                                 some types of prediction still unexplored. Prediction based
            of fixed-threshold metrics
                                                                 on continuous-time encoders is a particularly interesting
         • Accounts for differences between predicting new
                                                                 frontier due to the representations inherent advantages and
            and already observed edges without having the
                                                                 due to the limited amount of works in that area.
            metric to be dominated by either sub-problem.
         • Any predictor that predicts only new edges or
            only previously observed edges gets a score of       V. CHALLENGES AND FUTURE WORK
            0.                                                   There are plenty of challenges and multiple avenues for
   However, it does hinge on the assumption that reoccurring     the improvement of deep learning for both modelling and
edges is a balanced enough prediction problem that AUC           prediction of network topology.
is suitable. And that is not necessarily the case. Many real-       Expanding modelling and prediction repertoire. In
world networks are much more sparse than the two networks        this work we have exclusively focused on dynamic network
used by Junuthula et al. [52].                                   topology. However, complex networks are diverse and not
                                                                 only topology may vary. Topology dynamics can be repre-
D. DISCUSSION AND SUMMARY                                        sented as a 3-dimensional cube (Section II-D). However, real
In this section we have provided an overview of how,             networks can be much more complex. Complex networks
given a dynamic network encoder, one can perform network         may have dynamic node and edge attributes, they may have
topology prediction. The overview includes how methods           directed and/or signed edges, be heterogeneous in terms of
from section III use their embeddings for prediction. This       nodes and edges and be multilayered or multiplex. Each
completes the journey from establishing a dynamic network,       of these cases can be considered another dimension in the
to encoding the dynamic topology, to predicting changes in       dynamic network hypercube. Designing deep learning mod-
the topology.                                                    els for encoding these network cases expand the repertoire
   Prediction using a deep model requires decoding and           of tasks on which deep learning can be applied. Which
the use of a loss function that captures temporal and            types of networks can be encoded can be expanded as well
structural information. Prediction is largely focused on time-   as an expansion of what kind of predictions can be made
conditioned link prediction and the two main modelling           on those networks. For example, most DGNN models (and
approaches are (1) an architecture directly aimed at pre-        most GNN models) encode attributed dynamic networks but
diction; and (2) an architecture aimed at generating node        predict only graph topology without the node attributes.
embeddings which are then used for link prediction in a             Adoption of advances in closely related fields. Dy-
downstream step. Most dynamic network models surveyed            namic graph neural networks are based on GNNs and thus
fall into the second category, including all autoencoder         advances made to GNNs trickle down and can improve
approaches. All else being equal we would expect an              DGNNs. Challenges for GNNs include increasing modelling
architecture directly aimed at prediction to perform better      depth as GNNs struggle with vanishing gradients [121] and
than a two step architecture. This is because the first case     increasing scalability for large graphs [67]. As advance-
will allow the entire architecture to optimize itself towards    ments are made in deep neural networks for time series and
the prediction task.                                             in GNNs these advancements can be applied to dynamic
   The massive class imbalance makes the evaluation of           network modelling and prediction to improve performance.
dynamic link prediction is non-trivial. If the target problem    Similarly, improvements in deep time-series modelling can
has a natural fixed threshold, then adding a fixed threshold     easily be adapted to improve DGNNs.
to a common metric such as F1 is likely a good fit. PRAUC           Continuous DGNNs. Modelling temporal patterns is
(MAP) and Error rate are good metrics that avoids the class      what distinguishes modelling dynamic graphs from mod-
imbalance problem and are suitable for both link prediction      elling static graphs. Capturing these temporal patterns is key
and dynamic link prediction. The GMAUC metric incorpo-           to making accurate predictions. However, most models rely
rates the observation that reappearing and disappearing links    on snapshots which are coarse-grained temporal represen-
are not an imbalanced classification. Usage of GMAUC             tations. Methods modelling network change in continuous
however hinges on the assumption that reoccurring links are      time will offer fine-grained temporal modelling. Future work
22                                                                                                                 VOLUME 9, 2021
is needed for modelling and prediction of continuous-time                        [16] Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro
dynamic networks.                                                                     Suzumura, Hiroki Kanezashi, Tim Kaler, and Charles E. Leisersen.
                                                                                      Evolvegcn: Evolving graph convolutional networks for dynamic graphs.
   Scalability. Large scale datasets is a challenge for dy-                           CoRR, abs/1902.10191, 2019.
namic network modelling. Real-world datasets tend to be so                       [17] Aravind Sankar, Yanhong Wu, Liang Gou, Wei Zhang, and Hao Yang.
                                                                                      Dysat: Deep neural representation learning on dynamic graphs via self-
large that modelling becomes prohibitively slow. Dynamic                              attention networks. In James Caverlee, Xia (Ben) Hu, Mounia Lalmas,
networks either use a discrete representation in the form of                          and Wei Wang, editors, WSDM ’20: The Thirteenth ACM International
snapshots, in which case processing of each snapshot is the                           Conference on Web Search and Data Mining, Houston, TX, USA, Febru-
                                                                                      ary 3-7, 2020, pages 519–527. ACM, 2020.
bottleneck or continuous-time modelling which scales with                        [18] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan
the number of interactions. A snapshot model will need to                             Achan. Inductive representation learning on temporal graphs. arXiv
have frequent snapshots in order to achieve high temporal                             preprint arXiv:2002.07962, 2020.
                                                                                 [19] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard,
granularity. In addition, frequent snapshots might undermine                          Federico Monti, and Michael Bronstein. Temporal graph networks for
the capacity to model a temporal network. Improvements in                             deep learning on dynamic graphs. arXiv preprint arXiv:2006.10637,
continuous-time modelling are likely to improve the per-                              2020. tex.ids= rossiTemporalGraphNetworks2020.
                                                                                 [20] Jinyin Chen, Xuanheng Xu, Yangyang Wu, and Haibin Zheng. GC-
formance of deep learning modelling on dynamic networks
                                                                                      LSTM: graph convolution embedded LSTM for dynamic link prediction.
both in terms of temporal modelling capacity and ability to                           CoRR, abs/1812.04206, 2018.
handle large networks.                                                           [21] Rakshit Trivedi, Hanjun Dai, Yichen Wang, and Le Song. Know-evolve:
                                                                                      Deep temporal reasoning for dynamic knowledge graphs. In Doina
   Dynamic graph neural networks is a new exciting research                           Precup and Yee Whye Teh, editors, Proceedings of the 34th International
direction with a broad area of applications. With these                               Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia,
opportunities, the field is ripe with potential for future                            6-11 August 2017, volume 70 of Proceedings of Machine Learning
                                                                                      Research, pages 3462–3471. PMLR, 2017.
work.                                                                            [22] Jiapeng Wu, Meng Cao, Jackie Chi Kit Cheung, and William L Hamil-
                                                                                      ton. TeMP: Temporal message passing for temporal knowledge graph
                                                                                      completion. arXiv preprint arXiv:2010.03526, 2020.
REFERENCES                                                                       [23] Jia Li, Zhichao Han, Hong Cheng, Jiao Su, Pengyun Wang, Jianfeng
  [1] Steven H. Strogatz.         Exploring complex networks.          Nature,        Zhang, and Lujia Pan. Predicting Path Failure In Time-Evolving Graphs.
      410(6825):268, March 2001.                                                      In Proceedings of the 25th ACM SIGKDD International Conference on
  [2] Petter Holme and Jari Saramäki. Temporal networks. Physics reports,             Knowledge Discovery & Data Mining, KDD ’19, pages 1279–1289, New
      519(3):97–125, 2012.                                                            York, NY, USA, 2019. ACM.
  [3] Charu Aggarwal and Karthik Subbian. Evolutionary network analysis: A       [24] Fan Zhou, Xovee Xu, Ce Li, Goce Trajcevski, Ting Zhong, and Kunpeng
      survey. ACM Computing Surveys (CSUR), 47(1):10, 2014.                           Zhang. A heterogeneous dynamical graph neural networks approach to
  [4] Taisong Li, Bing Wang, Yasong Jiang, Yan Zhang, and Yonghong Yan.               quantify scientific impact. arXiv preprint arXiv:2003.12042, 2020.
      Restricted Boltzmann Machine-Based Approaches for Link Prediction in       [25] Yanbang Wang, Pan Li, Chongyang Bai, VS Subrahmanian, and Jure
      Dynamic Networks. IEEE Access, 6:29940–29951, 2018.                             Leskovec. Generic representation learning for dynamic social interaction.
  [5] Othon Michail and Paul G Spirakis. Elements of the theory of dynamic            In The 26th ACM SIGKDD international conference on knowledge
      networks. Communications of the ACM, 61(2):72–81, 2018.                         discovery & data mining, MLG workshop, 2020.
  [6] Petter Holme. Modern temporal network theory: A colloquium. The            [26] Ziwei Zhang, Peng Cui, and Wenwu Zhu. Deep Learning on Graphs: A
      European Physical Journal B, 88(9):234, September 2015.                         Survey. arXiv:1812.04202 [cs, stat], December 2018.
                                                                                 [27] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and P. S. Yu. A compre-
  [7] Kathleen M Carley, Jana Diesner, Jeffrey Reminga, and Maksim Tsveto-
                                                                                      hensive survey on graph neural networks. IEEE Transactions on Neural
      vat. Toward an interoperable dynamic network analysis toolkit. Decision
                                                                                      Networks and Learning Systems, pages 1–21, 2020.
      Support Systems, 43(4):1324–1347, 2007.
                                                                                 [28] William L Hamilton, Rex Ying, and Jure Leskovec. Representa-
  [8] Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu,
                                                                                      tion learning on graphs: Methods and applications. arXiv preprint
      Lifeng Wang, Changcheng Li, and Maosong Sun. Graph Neural Net-
                                                                                      arXiv:1709.05584, 2017.
      works: A Review of Methods and Applications. arXiv:1812.08434 [cs,
                                                                                 [29] Palash Goyal and Emilio Ferrara. Graph Embedding Techniques, Ap-
      stat], December 2018.
                                                                                      plications, and Performance: A Survey. Knowledge-Based Systems,
  [9] Naoki Masuda and Renaud Lambiotte. A Guide to Temporal Networks.                151:78–94, July 2018.
      World Scientific, 2016.                                                    [30] Giulio Rossetti and Rémy Cazabet. Community Discovery in Dynamic
 [10] Arnaud Casteigts, Paola Flocchini, Walter Quattrociocchi, and Nicola            Networks: A Survey. ACM Computing Surveys, 51(2):1–37, February
      Santoro. Time-Varying Graphs and Dynamic Networks. In Hannes                    2018.
      Frey, Xu Li, and Stefan Ruehrup, editors, Ad-Hoc, Mobile, and Wireless     [31] Bomin Kim, Kevin H Lee, Lingzhou Xue, and Xiaoyue Niu. A review
      Networks, pages 346–359, Berlin, Heidelberg, 2011. Springer Berlin              of dynamic network models with latent variables. Statistics surveys,
      Heidelberg.                                                                     12:105—135, 2018.
 [11] Seyed Mehran Kazemi, Rishab Goel, Kshitij Jain, Ivan Kobyzev,              [32] Jian Zhang. A survey on streaming algorithms for massive graphs. In
      Akshay Sethi, Peter Forsyth, and Pascal Poupart. Relational Rep-                Managing and Mining Graph Data, pages 393–420. Springer, 2010.
      resentation Learning for Dynamic (Knowledge) Graphs: A Survey.             [33] Cornelius Fritz, Michael Lebacher, and Göran Kauermann. Tempus volat,
      arXiv:1905.11485 [cs, stat], May 2019.                                          hora fugit: A survey of tie-oriented dynamic network models in discrete
 [12] Yu Xie, Chunyi Li, Bin Yu, Chen Zhang, and Zhouhua Tang. A survey on            and continuous time. Statistica Neerlandica, 2019.
      dynamic network embedding. arXiv preprint arXiv:2006.08093, 2020.          [34] Andrew McGregor. Graph stream algorithms: A survey. ACM SIGMOD
 [13] Claudio DT Barros, Matheus RF Mendonça, Alex B Vieira, and Artur                Record, 43(1):9–20, 2014.
      Ziviani. A survey on embedding dynamic graphs. arXiv preprint              [35] Saoussen Aouay, Salma Jamoussi, Faïez Gargouri, and Ajith Abraham.
      arXiv:2101.01229, 2021.                                                         Modeling dynamics of social networks: A survey. In CASoN, pages 49–
 [14] Ashesh Jain, Amir R Zamir, Silvio Savarese, and Ashutosh Saxena.                54. IEEE, 2014.
      Structural-rnn: Deep learning on spatio-temporal graphs. In Proceedings    [36] Giulio Rossetti. Social Network Dynamics. 2015.
      of the ieee conference on computer vision and pattern recognition, pages   [37] Mayur Datar, Aristides Gionis, Piotr Indyk, and Rajeev Motwani. Main-
      5308–5317, 2016.                                                                taining Stream Statistics over Sliding Windows. SIAM Journal on
 [15] Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph con-              Computing, 31(6):1794–1813, January 2002.
      volutional networks: A deep learning framework for traffic forecasting.    [38] Matteo Morini, Patrick Flandrin, Eric Fleury, Tommaso Venturini, and
      arXiv preprint arXiv:1709.04875, 2017.                                          Pablo Jensen. Revealing evolutions in dynamical networks. July 2017.

VOLUME 9, 2021                                                                                                                                              23
[39] S. Boccaletti, G. Bianconi, R. Criado, C. I. del Genio, J. Gómez-                prints. Journal of computer-aided molecular design, 30(8):595–608,
     Gardeñes, M. Romance, I. Sendiña-Nadal, Z. Wang, and M. Zanin.                   2016.
     The structure and dynamics of multilayer networks. Physics Reports,         [62] David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bom-
     544(1):1–122, November 2014.                                                     barell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Con-
[40] Daniel M Dunlavy, Tamara G Kolda, and Evrim Acar. Temporal link                  volutional networks on graphs for learning molecular fingerprints. In
     prediction using matrix and tensor factorizations. ACM Transactions on           Advances in Neural Information Processing Systems, pages 2224–2232,
     Knowledge Discovery from Data (TKDD), 5(2):10, 2011.                             2015.
[41] Albert-László Barabási and Réka Albert. Emergence of scaling in             [63] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L
     random networks. science, 286(5439):509–512, 1999.                               Hamilton, and Jure Leskovec. Graph convolutional neural networks
[42] Jure Leskovec, Jon Kleinberg, and Christos Faloutsos. Graphs over time:          for web-scale recommender systems. In Proceedings of the 24th ACM
     Densification laws, shrinking diameters and possible explanations. In            SIGKDD International Conference on Knowledge Discovery & Data
     Proceedings of the Eleventh ACM SIGKDD International Conference on               Mining, pages 974–983. ACM, 2018.
     Knowledge Discovery in Data Mining, pages 177–187. ACM, 2005.               [64] Federico Monti, Michael Bronstein, and Xavier Bresson. Geometric ma-
[43] Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, and Jure                  trix completion with recurrent multi-graph neural networks. In Advances
     Leskovec. GraphRNN: A Deep Generative Model for Graphs. CoRR,                    in Neural Information Processing Systems, pages 3697–3707, 2017.
     abs/1802.08773, 2018.                                                       [65] Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, and
[44] Mark Newman. Networks. Oxford university press, 2018.                            Jie Tang. DeepInf: Social Influence Prediction with Deep Learning.
[45] Albert-Laszlo Barabâsi, Hawoong Jeong, Zoltan Néda, Erzsebet Ravasz,             Proceedings of the 24th ACM SIGKDD International Conference on
     Andras Schubert, and Tamas Vicsek. Evolution of the social network               Knowledge Discovery & Data Mining - KDD ’18, pages 2110–2119,
     of scientific collaborations. Physica A: Statistical mechanics and its           2018.
     applications, 311(3-4):590–614, 2002.                                       [66] Yozen Liu, Xiaolin Shi, Lucas Pierce, and Xiang Ren. Characterizing
[46] Kevin Xu. Stochastic block transition models for dynamic networks. In            and forecasting user engagement with in-app action graph: A case study
     Artificial Intelligence and Statistics, pages 1079–1087, 2015.                   of snapchat. In Proceedings of the 25th ACM SIGKDD International
[47] Jinyin Chen, Jian Zhang, Xuanheng Xu, Chengbo Fu, Dan Zhang, Qing-               Conference on Knowledge Discovery & Data Mining, pages 2023–2031,
     peng Zhang, and Qi Xuan. E-LSTM-D: A Deep Learning Framework for                 2019.
     Dynamic Network Link Prediction. arXiv preprint arXiv:1902.08329,           [67] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation
     2019.                                                                            learning on large graphs. In Advances in Neural Information Processing
[48] Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan             Systems, pages 1024–1034, 2017.
     Zha. DyRep: Learning representations over dynamic graphs. In Interna-       [68] Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane
     tional Conference on Learning Representations, 2019.                             Idoumghar, and Pierre-Alain Muller. Deep learning for time series clas-
[49] Kevin S. Xu and Alfred O. Hero III. Dynamic stochastic blockmodels               sification: A review. Data Mining and Knowledge Discovery, 33(4):917–
     for time-evolving social networks. IEEE Journal of Selected Topics in            963, July 2019.
     Signal Processing, 8(4):552–562, August 2014.                               [69] Youngjoo Seo, Michaël Defferrard, Pierre Vandergheynst, and Xavier
[50] Akanda Wahid-Ul Ashraf, Marcin Budka, and Katarzyna Musial. Simu-                Bresson. Structured Sequence Modeling with Graph Convolutional
     lation and Augmentation of Social Networks for Building Deep Learning            Recurrent Networks. In Long Cheng, Andrew Chi Sing Leung, and
     Models. May 2019.                                                                Seiichi Ozawa, editors, Neural Information Processing, Lecture Notes
[51] Nicola Perra, Bruno Gonçalves, Romualdo Pastor-Satorras, and Alessan-            in Computer Science, pages 362–373. Springer International Publishing,
     dro Vespignani. Activity driven modeling of time varying networks.               2018.
     Scientific reports, 2(1):1–7, 2012. Publisher: Nature Publishing Group.     [70] Purnamrita Sarkar and Andrew Moore. Dynamic social network analysis
[52] Ruthwik R Junuthula, Kevin S Xu, and Vijay K Devabhaktuni. Evaluating            using latent space models. In Y. Weiss, B. Schölkopf, and J. Platt, editors,
     link prediction accuracy in dynamic networks with added and removed              Advances in neural information processing systems, volume 18. MIT
     edges. In 2016 IEEE International Conferences on Big Data and Cloud              Press, 2006.
     Computing (BDCloud), Social Computing and Networking (SocialCom),           [71] Tianbao Yang, Yun Chi, Shenghuo Zhu, Yihong Gong, and Rong Jin. De-
     Sustainable Computing and Communications (SustainCom)(BDCloud-                   tecting communities and their evolutions in dynamic social networks—a
     SocialCom-SustainCom), pages 377–384. IEEE, 2016.                                Bayesian approach. Machine learning, 82(2):157–189, 2011. Publisher:
[53] Tom A. B. Snijders, Gerhard G. van de Bunt, and Christian E. G. Steglich.        Springer.
     Introduction to stochastic actor-based models for network dynamics.         [72] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner,
     Social Networks, 32(1):44–60, January 2010.                                      and Gabriele Monfardini. The graph neural network model. IEEE
[54] Aswathy Divakaran and Anuraj Mohan. Temporal link prediction: A                  transactions on neural networks, 20(1):61–80, 2008. Publisher: IEEE.
     survey. New Generation Computing, pages 1–46, 2019.                         [73] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. DeepWalk: Online
[55] Xiaoyi Li, Nan Du, Hui Li, Kang Li, Jing Gao, and Aidong Zhang.                  Learning of Social Representations. CoRR, abs/1403.6652, 2014.
     A deep learning approach to link prediction in dynamic networks. In         [74] Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolu-
     Proceedings of the 2014 SIAM International Conference on Data Mining,            tional Neural Networks on Graphs with Fast Localized Spectral Filtering.
     pages 289–297. SIAM, 2014.                                                       In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett,
[56] Feng Liu, Bingquan Liu, Chengjie Sun, Ming Liu, and Xiaolong Wang.               editors, Advances in Neural Information Processing Systems 29, pages
     Deep learning approaches for link prediction in social network services.         3844–3852. Curran Associates, Inc., 2016.
     In International conference on neural information processing, pages 425–    [75] Thomas N. Kipf and Max Welling. Semi-supervised classification
     432, 2013. tex.organization: Springer.                                           with graph convolutional networks. In 5th International Conference
[57] Carter T. Butts and Christopher Steven Marcum. A relational event                on Learning Representations, ICLR 2017, Toulon, France, April 24-26,
     approach to modeling behavioral dynamics. In Andrew Pilny and                    2017, Conference Track Proceedings. OpenReview.net, 2017.
     Marshall Scott Poole, editors, Group processes: Data-driven compu-          [76] Franco Manessi, Alessandro Rozza, and Mario Manzo. Dynamic graph
     tational approaches, pages 51–92. Springer International Publishing,             convolutional networks. Pattern Recognition, 97:107000, January 2020.
     Cham, 2017.                                                                 [77] Srijan Kumar, Xikun Zhang, and Jure Leskovec. Predicting dynamic
[58] Steve Hanneke, Wenjie Fu, and Eric P. Xing. Discrete temporal models             embedding trajectory in temporal interaction networks. In Proceedings
     of social networks. Electronic Journal of Statistics, 4:585–605, 2010.           of the 25th ACM SIGKDD International Conference on Knowledge
[59] Per Block, Johan Koskinen, James Hollway, Christian Steglich, and                Discovery & Data Mining, pages 1269–1278, 2019.
     Christoph Stadtfeld. Change we can believe in: Comparing longitudinal       [78] Yao Ma, Ziyi Guo, Zhaochun Ren, Eric Zhao, Jiliang Tang, and Dawei
     network models on consistency, interpretability and predictive power.            Yin. Streaming Graph Neural Networks. arXiv:1810.10627 [cs, stat],
     Social Networks, 52:180–191, January 2018.                                       October 2018.
[60] Anna Goldenberg, Alice X. Zheng, Stephen E. Fienberg, and Edoardo M.        [79] Da Xu, Chuanwei Ruan, Kamiya Motwani, Evren Korpeoglu, Sushant
     Airoldi. A Survey of Statistical Network Models. Foundations and                 Kumar, and Kannan Achan. Generative Graph Convolutional Network
     Trends® in Machine Learning, 2(2):129–233, February 2010.                        for Growing Graphs. ICASSP 2019 - 2019 IEEE International Confer-
[61] Steven Kearnes, Kevin McCloskey, Marc Berndl, Vijay Pande, and                   ence on Acoustics, Speech and Signal Processing (ICASSP), pages 3167–
     Patrick Riley. Molecular graph convolutions: Moving beyond finger-               3171, May 2019.

24                                                                                                                                                VOLUME 9, 2021
 [80] Thomas N. Kipf and Max Welling. Variational graph auto-encoders.                   Tordai, and Mehwish Alam, editors, The Semantic Web, Lecture Notes
      CoRR, abs/1611.07308, 2016.                                                        in Computer Science, pages 593–607. Springer International Publishing,
 [81] Liang Qu, Huaisheng Zhu, Qiqi Duan, and Yuhui Shi. Continuous-                     2018.
      time link prediction via temporal dependent graph neural network. In          [98] Palash Goyal, Sujit Rokka Chhetri, Ninareh Mehrabi, Emilio Ferrara,
      Proceedings of the web conference 2020, WWW ’20, pages 3026–3032,                  and Arquimedes Canedo. DynamicGEM: A library for dynamic graph
      New York, NY, USA, 2020. Association for Computing Machinery.                      embedding methods. arXiv preprint arXiv:1811.10734, 2018.
      Number of pages: 7 Place: Taipei, Taiwan.                                     [99] Daixin Wang, Peng Cui, and Wenwu Zhu. Structural Deep Network
 [82] Felix A Gers, Nicol N Schraudolph, and Jurgen Schmidhuber. Learning                Embedding. In Proceedings of the 22Nd ACM SIGKDD International
      Precise Timing with LSTM Recurrent Networks. page 29.                              Conference on Knowledge Discovery and Data Mining, KDD ’16, pages
 [83] Apurva Narayan and Peter H. O’N Roe. Learning Graph Dynamics using                 1225–1234, New York, NY, USA, 2016. ACM.
      Deep Neural Networks. IFAC-PapersOnLine, 51(2):433–438, January              [100] Tianqi Chen, Ian J. Goodfellow, and Jonathon Shlens. Net2net: Acceler-
      2018.                                                                              ating learning via knowledge transfer. In Yoshua Bengio and Yann Le-
 [84] Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov. Learning                   Cun, editors, 4th International Conference on Learning Representations,
      convolutional neural networks for graphs. In Maria-Florina Balcan and              ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track
      Kilian Q. Weinberger, editors, Proceedings of the 33nd International               Proceedings, 2016.
      Conference on Machine Learning, ICML 2016, New York City, NY,                [101] Palash Goyal, Sujit Rokka Chhetri, and Arquimedes Canedo.
      USA, June 19-24, 2016, volume 48 of JMLR Workshop and Conference                   Dyngraph2vec: Capturing Network Dynamics using Dynamic
      Proceedings, pages 2014–2023. JMLR.org, 2016.                                      Graph Representation Learning. Knowledge-Based Systems, page
 [85] Aynaz Taheri, Kevin Gimpel, and Tanya Berger-Wolf. Learning to                     S0950705119302916, July 2019.
      Represent the Evolution of Dynamic Graphs with Recurrent Models.             [102] Ehsan Hajiramezanali, Arman Hasanzadeh, Krishna R. Narayanan, Nick
      In Companion Proceedings of The 2019 World Wide Web Conference,                    Duffield, Mingyuan Zhou, and Xiaoning Qian. Variational graph recur-
      WWW ’19, pages 301–307, New York, NY, USA, 2019. ACM.                              rent neural networks. In Hanna M. Wallach, Hugo Larochelle, Alina
 [86] Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard S. Zemel.                  Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett,
      Gated graph sequence neural networks. In Yoshua Bengio and Yann Le-                editors, Advances in Neural Information Processing Systems 32: Annual
      Cun, editors, 4th International Conference on Learning Representations,            Conference on Neural Information Processing Systems 2019, NeurIPS
      ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track                  2019, 8-14 December 2019, Vancouver, BC, Canada, pages 10700–
      Proceedings, 2016.                                                                 10710, 2019.
 [87] Sepp Hochreiter and Jürgen Schmidhuber. Long Short-Term Memory.              [103] Mingzhang Yin and Mingyuan Zhou. Semi-Implicit Variational Infer-
      Neural Comput., 9(8):1735–1780, November 1997.                                     ence. In International Conference on Machine Learning, pages 5660–
 [88] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero,              5669, July 2018.
      Pietro Liò, and Yoshua Bengio. Graph attention networks. In 6th Interna-     [104] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David
      tional Conference on Learning Representations, ICLR 2018, Vancouver,               Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Gen-
      BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings.                  erative Adversarial Nets. In Z. Ghahramani, M. Welling, C. Cortes, N. D.
      OpenReview.net, 2018.                                                              Lawrence, and K. Q. Weinberger, editors, Advances in Neural Informa-
 [89] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion                  tion Processing Systems 27, pages 2672–2680. Curran Associates, Inc.,
      Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention               2014.
      is All you Need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach,          [105] Zhengwei Wang, Qi She, and Tomas E Ward. Generative adversarial
      R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural            networks: A survey and taxonomy. arXiv preprint arXiv:1906.01529,
      Information Processing Systems 30, pages 5998–6008. Curran Asso-                   2019.
      ciates, Inc., 2017.                                                          [106] Kai Lei, Meng Qin, Bo Bai, Gong Zhang, and Min Yang. Gcn-gan:
 [90] Yanbang Wang, Pan Li, Chongyang Bai, and Jure Leskovec. TEDIC:                     A non-linear temporal link prediction model for weighted dynamic
      Neural modeling of behavioral patterns in dynamic social interaction               networks. In IEEE INFOCOM 2019-IEEE Conference on Computer
      networks. 2020.                                                                    Communications, pages 388–396. IEEE, 2019.
 [91] Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan,              [107] Yun Xiong, Yao Zhang, Hanjie Fu, Wei Wang, Yangyong Zhu, and
      Oriol Vinyals, Alexander Graves, Nal Kalchbrenner, Andrew Senior, and              S Yu Philip. Dyngraphgan: Dynamic graph embedding via generative
      Koray Kavukcuoglu. WaveNet: A generative model for raw audio. In                   adversarial networks. In International Conference on Database Systems
      Arxiv, 2016.                                                                       for Advanced Applications, pages 536–552. Springer, 2019.
 [92] Muhan Zhang and Yixin Chen. Link prediction based on graph neural            [108] Inci M. Baytas, Cao Xiao, Xi Zhang, Fei Wang, Anil K. Jain, and Jiayu
      networks. In Advances in neural information processing systems, pages              Zhou. Patient Subtyping via Time-Aware LSTM Networks. In Proceed-
      5165–5175, 2018.                                                                   ings of the 23rd ACM SIGKDD International Conference on Knowledge
 [93] Lei Cai, Zhengzhang Chen, Chen Luo, Jiaping Gui, Jingchao Ni, Ding                 Discovery and Data Mining - KDD ’17, pages 65–74, Halifax, NS,
      Li, and Haifeng Chen. Structural temporal graph neural networks for                Canada, 2017. ACM Press.
      anomaly detection in dynamic graphs. arXiv preprint arXiv:2005.07427,        [109] Boris Knyazev, Carolyn Augusta, and Graham W Taylor. Learning
      2020.                                                                              temporal attention in dynamic graphs with bilinear interactions. arXiv
 [94] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin                      preprint arXiv:1909.10367, 2019.
      Wong, and Wang-chun Woo. Convolutional LSTM network: A machine               [110] Thomas N. Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and
      learning approach for precipitation nowcasting. In Corinna Cortes,                 Richard S. Zemel. Neural relational inference for interacting sys-
      Neil D. Lawrence, Daniel D. Lee, Masashi Sugiyama, and Roman                       tems. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of
      Garnett, editors, Advances in Neural Information Processing Systems                the 35th International Conference on Machine Learning, ICML 2018,
      28: Annual Conference on Neural Information Processing Systems 2015,               Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of
      December 7-12, 2015, Montreal, Quebec, Canada, pages 802–810, 2015.                Proceedings of Machine Learning Research, pages 2693–2702. PMLR,
 [95] Woojeong Jin, He Jiang, Meng Qu, Tong Chen, Changlin Zhang, Pedro                  2018.
      Szekely, and Xiang Ren. Recurrent event network: Global structure            [111] Zhen Han, Yuyi Wang, Yunpu Ma, Stephan Günnemann, and Volker
      inference over temporal knowledge graph. arXiv: 1904.05530, 2019.                  Tresp. The graph hawkes network for reasoning on temporal knowledge
 [96] Stephen Bonner, Amir Atapour-Abarghouei, Philip T Jackson, John                    graphs. arXiv preprint arXiv:2003.13432, 2020.
      Brennan, Ibad Kureshi, Georgios Theodoropoulos, Andrew Stephen Mc-           [112] Hongyuan Mei and Jason M Eisner. The neural hawkes process: A
      Gough, and Boguslaw Obara. Temporal neighbourhood aggregation:                     neurally self-modulating multivariate point process. In Advances in
      Predicting future links in temporal graphs via recurrent variational graph         neural information processing systems, pages 6754–6764, 2017.
      convolutions. In 2019 IEEE international conference on big data (big         [113] Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan,
      data), pages 5336–5345, 2019. tex.organization: IEEE.                              Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart,
 [97] Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den                 and Marcus Brubaker. Time2vec: Learning a vector representation of
      Berg, Ivan Titov, and Max Welling. Modeling Relational Data with                   time. arXiv preprint arXiv:1907.05321, 2019.
      Graph Convolutional Networks. In Aldo Gangemi, Roberto Navigli,              [114] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan
      Maria-Esther Vidal, Pascal Hitzler, Raphaël Troncy, Laura Hollink, Anna            Achan. Self-attention with functional time representation learning. In

VOLUME 9, 2021                                                                                                                                                25
      H. Wallach, H. Larochelle, A. Beygelzimer, F. dAlché Buc, E. Fox, and                                 KATARZYNA MUSIAL received the M.Sc. de-
      R. Garnett, editors, Advances in neural information processing systems,                               gree in computer science from the Wrocław
      volume 32. Curran Associates, Inc., 2019.                                                             University of Science and Technology (WrUST),
[115] Changmin Wu, Giannis Nikolentzos, and Michalis Vazirgiannis. EvoNet:                                  Poland, the M.Sc. degree in software engineer-
      A neural network for predicting the evolution of dynamic graphs. arXiv                                ing from the Blekinge Institute of Technology,
      preprint arXiv:2003.00842, 2020.                                                                      Sweden, in 2006, and the Ph.D. from WrUST, in
[116] Yang Yang, Ryan N. Lichtenwalter, and Nitesh V. Chawla. Evaluating                                    November 2009.
      link prediction methods. Knowl. Inf. Syst., 45(3):751–782, 2015.
                                                                                                              In November 2009, she was appointed as a Se-
[117] Odd Aalen, Ornulf Borgan, and Hakon Gjessing. Survival and Event
                                                                                                            nior Visiting Research Fellow with Bournemouth
      History Analysis: A Process Point of View. Springer Science & Business
      Media, 2008.                                                                                          University (BU), where she has been a Lecturer in
[118] Taisong Li, Jiawei Zhang, S Yu Philip, Yan Zhang, and Yonghong Yan.        informatics, since 2010. In November 2011, she joined Kings as a Lecturer
      Deep dynamic network embedding for link prediction. IEEE Access,           in computer science. In September 2015, she returned to Bournemouth
      6:29219–29230, 2018.                                                       University as a Principal Academic in Computing, where she was a
[119] Linyuan Lü and Tao Zhou. Link prediction in complex networks: A            member of the Data Science Institute. In September 2017, she joined as
      survey. Physica A, 390(6):11501170, 2011.                                  an Associate Professor in network science with the School of Software,
[120] Víctor Martínez, Fernando Berzal, and Juan-Carlos Cubero. A Survey of      University of Technology Sydney, where she is currently a member of
      Link Prediction in Complex Networks. ACM Comput. Surv., 49(4):69:1–        the Advanced Analytics Institute. Her research interests include complex
      69:33, December 2016.                                                      networked systems, analysis of their dynamics and its evolution, adaptive
[121] Guohao Li, Matthias Müller, Ali K. Thabet, and Bernard Ghanem.             and predictive modeling of their structure and characteristics, as well as
      Deepgcns: Can gcns go as deep as cnns? In 2019 IEEE/CVF International      the adaptation mechanisms that exist within such systems are in the center
      Conference on Computer Vision, ICCV 2019, Seoul, Korea (South),            of her research interests.
      October 27 - November 2, 2019, pages 9266–9275. IEEE, 2019.


                          JOAKIM SKARDING received the M.Sc degree
                          in computer science from the Norwegian Univer-
                          sity of Science and Technology (NTNU) in 2015.
                             He worked as a software developer for Tom-
                          Tom in Berlin and since 2018 he has been pur-
                          suing his PhD at the University of Technology
                          Sydney. His research interest include modelling
                          and prediction of dynamic complex networks.


                          BOGDAN GABRYS received the M.Sc. degree
                          in electronics and telecommunication from Sile-
                          sian Technical University, Gliwice, Poland, in
                          1994, and the Ph.D. degree in computer science
                          from Nottingham Trent University, Nottingham,
                          U.K., in 1998.
                             Over the last 25 years, he has been working at
                          various universities and research and development
                          departments of commercial institutions. He is
                          currently a Professor of Data Science and the Di-
rector of the Advanced Analytics Institute at the University of Technology
Sydney, Sydney, Australia. His research activities have concentrated on
the areas of data science, complex adaptive systems, computational intelli-
gence, machine learning, predictive analytics, and their diverse applications.
He has published over 180 research papers, chaired conferences, work-
shops, and special sessions, and been on program committees of a large
number of international conferences with the data science, computational
intelligence, machine learning, and data mining themes. He is also a Senior
Member of the Institute of Electrical and Electronics Engineers (IEEE), a
Member of IEEE Computational Intelligence Society and a Fellow of the
Higher Education Academy (HEA) in the UK. He is frequently invited to
give keynote and plenary talks at international conferences and lectures
at internationally leading research centres and commercial research labs.
More details can be found at: http://bogdan-gabrys.com


26                                                                                                                                            VOLUME 9, 2021
                             TABLE 7: A summary of notation used in this work.
                 Notation              Description
                                       Element wise product
                 G                     Static graph
                 Gt                    Static graph at time t
                 DG                    Discrete dynamic graph
                 CG                    Continuous dynamic graph
                 V                     The set of nodes in a graph
                 E                     The set of edges in a graph
                 v                     A node v ∈ V
                 e                     An edge e ∈ E
                 t                     Time step / event time
                 t̄                    Time step just before time t
                 <t                    All time steps up until time t
                 ∆                     Duration
                 n                     Number of nodes
                 d                     Dimensions of a node feature vector
                 l                     Dimensions of a GNN produced hidden feature vector
                 k                     Dimensions of a RNN/self-attention produced hidden feature vector
                 Xt                    Feature matrix at time t
                 At                    Adjacency matrix at time t
                 Â                    Predicted adjacency matrix
                 zu t                  GNN produced hidden feature vector of node u at time t
                 htu                   RNN/self-attention produced hidden feature vector of node u at time t
                 Zt                    GNN produced hidden feature matrix at time t
                 Ht                    RNN/self-attention produced hidden feature matrix at time t
                 λ                     Conditional intensity function
                 σ                     The sigmoid function
                 W, U, w, b, ω, ψ, ϕ   Learnable model parameters


VOLUME 9, 2021                                                                                                 27
TABLE 8: A summary of abbreviations used in this work. Some model names in the text look like abbreviations but are
in fact simply the name of the model (or the authors do not explicitly state what the abbreviation stand for). These include:
PATCHY-SAN, DyGGNN, RgGNN, StrGNN, EvolveGCN, JODIE, GC-LSTM, GCN-GAN, DynGraphGAN and DyREP.
                               Abbreviation   Description
                               GNN            Graph neural network
                               DGNN           Dynamic graph neural network
                               RNN            Recurrent neural network
                               LSTM           Long-term short term memory
                               GRU            Gated recurrent unit
                               GAN            Generative adversarial network
                               CNN            Convolutional neural network
                               TPP            Temporal point process
                               RGM            Random graph model
                               ERGM           Exponential random graph model
                               TERGM          Temporal exponential random graph model
                               SAOM           Stochastic actor oriented model
                               REM            Relational event model
                               GCN            Graph Convolutional Network [75]
                               GAT            Graph Attention Network [88]
                               GGNN           Gated Graph Neural Network [86]
                               R-GCN          Relational Graph Convolutional Network [97]
                               convLSTM       Convolutional LSTM [94]
                               GraphRNN       Graph recurrent neural network [43]
                               G-GCN          Generative graph convolutional network [79]
                               VGAE           Variational graph autoencoder [80]
                               GCRN-M1        Graph convolutional recurrent network model 1 [69]
                               GCRN-M2        Graph convolutional recurrent network model 2 [69]
                               WD-GCN         Waterfall dynamic graph convolutional network [76]
                               CD-GCN         Concatenated dynamic graph convolutional network [76]
                               DySAT          Dynamic Self-Attention Network [17]
                               TNDCN          Temporal network-diffusion convolution networks [25] [90]
                               HDGNN          Heterogeneous Dynamical Graph Neural Network [24]
                               TeMP           Temporal Message Passing [22]
                               LRGCN          Long Short-Term Memory R-GCN [23]
                               RE-Net         Recurrent Event Network [95]
                               TNA            Temporal Neighbourhood Aggregation [96]
                               TDGNN          Temporal Dependent Graph Neural Network [81]
                               DynGEM         Dynamic graph embedding model [98]
                               E-LSTM-D       Encode-LSTM-Decode [47]
                               VGRNN          Variational graph recurrent neural network [103]
                               SI-VGRNN       Semi-implicit VGRNN [103]
                               SGNN           Streaming graph neural network [78]
                               LDG            Latent dynamic graph [109]
                               GHN            Graph Hawkes network [111]
                               TGAT           Temporal graph attention [18]
                               TGN            Temporal Graph Networks [19]


28                                                                                                               VOLUME 9, 2021
