import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)

from directed_graph.graph import Graph, visualize_graph, visualize_graph_with_equivalent_elements
from graph_metrics.metrics import GraphMetrics
# book = HigherDimGraph()
book = Graph()

# https://arxiv.org/pdf/2203.10308

# Thermodynamics studies energy transfer, storage, and
# usage. It started as a theory of heat engines, drove the Industrial Revolution, and matured at nearly the same time
# when evolutionary biology emerged. Nowadays, thermodynamics is perhaps the most general phenomenological
# theory in all of science that applies to all types of systems
# at all levels of organization. Several attempts have been
# made to represent various aspects of evolutionary biology,
# in particular, evolution of populations and ecosystems,
# within the framework of thermodynamics [1–11].

book.add_vertex("thermodynamics")
book.add_vertex("energy transfer")
book.add_vertex("energy storage")
book.add_vertex("energy usage")
book.add_vertex("theory of heat engines")
book.add_vertex("Industrial Revolution")
book.add_vertex("evolutionary biology")
book.add_vertex("evolution of populations")
book.add_vertex("ecosystems")

for concept in sorted(book.vertices.keys())[:4]:
    book.vertices[concept].vertex_type=0    

book.add_edge("thermodynamics", "energy transfer", "studies")
book.add_edge("thermodynamics", "energy storage", "studies")
book.add_edge("thermodynamics", "energy usage", "studies")
book.add_edge("thermodynamics", "theory of heat engines", "started as")
book.add_edge("Industrial Revolution", "theory of heat engines", "drove")
book.add_edge("thermodynamics", "evolutionary biology", "matured when")
book.add_edge("evolution of populations", "thermodynamics", "uses")
book.add_edge("ecosystems", "thermodynamics", "uses")
book.add_edge("evolutionary biology", "thermodynamics", "aspects of")


# # Here we develop a thermodynamic approach to selection. Its main premise is that once organisms (agents)
# # extract work (useful energy) and obey the laws of nonequilibrium thermodynamics in their metabolism [12],
# # they can be modeled as heat engines. Agents interact (compete) indirectly, if the extraction goes from the
# # same source. This competition can be represented via
# # game theory, and hence its outcome depends on workextraction strategies adopted by the agents. Such strategies depend on two parameters: efficiency and power.
# # The second law of thermodynamics states that the efficiency of any heat engine—defined as the ratio of useful extracted energy (work) to the total energy input
# # (heat)—is bound from above by Carnot’s efficiency [13–
# # 15]. But heat engines operating at the maximum efficiency yield zero work per unit of time (zero work power)
# # resulting in the well-known power-efficiency tradeoff [16–
# # 19]: the most efficient regime is not powerful, whereas
# # the most powerful regime is not efficient [20].

# book.add_vertex("thermodynamic approach to selection")
# book.add_vertex("organisms")
# book.add_vertex("agents")
# book.add_vertex("work")
# book.add_vertex("useful energy")
# book.add_vertex("heat engines")
# book.add_vertex("nonequilibrium thermodynamics")
# book.add_vertex("metabolism")
# book.add_vertex("game theory")
# book.add_vertex("work-extraction strategies")
# book.add_vertex("efficiency")
# book.add_vertex("power")
# book.add_vertex("Carnot’s efficiency")
# book.add_vertex("power-efficiency tradeoff")
# book.add_vertex("second law of thermodynamics")

# book.add_edge("thermodynamic approach to selection", "organisms", "applies to")
# book.add_edge("organisms", "work", "extract")
# book.add_edge("organisms", "useful energy", "extract")
# book.add_edge("organisms", "heat engines", "are modeled as") # more active
# book.add_edge("organisms", "agents", "are")
# book.add_edge("organisms", "nonequilibrium thermodynamics", "obey the laws")
# book.add_edge("organisms", "metabolism", "in")
# book.add_edge("game theory", "work-extraction strategies", "represents") # more active
# book.add_edge("work-extraction strategies", "efficiency", "depend on")
# book.add_edge("work-extraction strategies", "power", "depend on")
# book.add_edge("Carnot’s efficiency", "efficiency", "bounds")  # more active
# book.add_edge("second law of thermodynamics", "efficiency", "states")
# book.add_edge("power-efficiency tradeoff", "efficiency", "results in") # more active
# book.add_edge("power-efficiency tradeoff", "power", "results in")  # more active


# # The energy budget of an organism can be described as
# # three main energy currents: input, storage, and output
# # (waste) [4, 21–23]. The relationship between these three
# # currents are similar to that in a generalized heat engine:
# # input heat, work (storage), and output heat. Similar
# # to abiotic heat engines, organisms also face the power-efficiency (or speed-fidelity) trade-off. In particular, this
# # trade-off is seen in molecular machines of cells [24–27],
# # and also at the level of organism phenotypes [28–34]. The
# # power-efficiency trade-off is subject to selection and depends on available energy resources

# book.add_vertex("energy budget of an organism")
# book.add_vertex("energy currents")
# book.add_vertex("input")
# book.add_vertex("storage")
# book.add_vertex("output")
# book.add_vertex("waste")
# book.add_vertex("heat engine")
# book.add_vertex("input heat")
# book.add_vertex("output heat")
# book.add_vertex("abiotic heat engines")
# book.add_vertex("speed-fidelity tradeoff")
# book.add_vertex("molecular machines of cells")
# book.add_vertex("organism phenotypes")
# book.add_vertex("selection")
# book.add_vertex("energy resources")

# book.add_edge("energy budget of an organism", "energy currents", "includes") # more active
# book.add_edge("energy currents", "input", "include")
# book.add_edge("energy currents", "storage", "include")
# book.add_edge("energy currents", "output", "include")
# book.add_edge("energy currents", "waste", "include")
# book.add_edge("heat engine", "input heat", "similar to")
# book.add_edge("heat engine", "storage", "similar to")
# book.add_edge("heat engine", "output heat", "similar to")
# book.add_edge("organisms", "power-efficiency tradeoff", "face")
# book.add_edge("organisms", "speed-fidelity tradeoff", "face")
# book.add_edge("molecular machines of cells", "power-efficiency tradeoff", "show") # more active
# book.add_edge("organism phenotypes", "power-efficiency tradeoff", "show") # more active
# book.add_edge("selection", "power-efficiency tradeoff", "acts on") # more active
# book.add_edge("energy resources", "power-efficiency tradeoff", "depends on")


# # Hence, our goal is to explore a physical model for
# # the evolution of the metabolic power-efficiency trade-off,
# # where agents are modeled as heat engines. We do not
# # specify how the extracted work is utilized (reproduction,
# # metabolism, defense, or other functions). Instead, we focus on different strategies (phenotypes) that are available
# # to the agents to extract and store energy. The competition and selection emerge because at least two agents employ the same source (high-temperature bath). There are
# # two general scenarios for such competition, for effectively
# # infinite and for finite—and hence depletable—resources.
# # The quantities relevant for evolution in these two situations are, respectively, the power of work extraction and
# # the stored energy (=total extracted work).

# book.add_vertex("physical model")
# book.add_vertex("evolution of metabolic power-efficiency trade-off")
# book.add_vertex("high-temperature bath")
# book.add_vertex("reproduction")
# book.add_vertex("defense")
# book.add_vertex("work extraction")
# book.add_vertex("competition")
# book.add_vertex("infinite resources")
# book.add_vertex("finite resources")
# book.add_vertex("power of work extraction")
# book.add_vertex("stored energy")
# book.add_vertex("total extracted work")
# book.add_vertex("evolution")

# book.add_edge("physical model", "evolution of metabolic power-efficiency trade-off", "for")
# book.add_edge("agents", "heat engines", "model as") # more active
# book.add_edge("agents", "work extraction", "extract")
# book.add_edge("agents", "stored energy", "store")
# book.add_edge("competition", "high-temperature bath", "agents employ") # more active
# book.add_edge("selection", "high-temperature bath", "agents employ") # more active
# book.add_edge("competition", "infinite resources", "scenarios for")
# book.add_edge("competition", "finite resources", "scenarios for")
# book.add_edge("power of work extraction", "evolution", "influences") # more active
# book.add_edge("stored energy", "evolution", "influences") # more active
# book.add_edge("stored energy", "total extracted work", "is")


# # Competition for an infinite resource is analogous to the
# # competition of plants for light. Here the source, i.e. the
# # Sun, acts as a thermal bath providing high temperature
# # photons for the heat engine operation of the photosynthesis. It is not depletable, and yet, there is a competition for a limited energy current reaching the forest
# # surface [63–73]. Plants can behave differently when facing such competition, from confrontation to avoidance of
# # the competitor [71, 74, 75]. In section III we formalize
# # and examine these situations that can have more general
# # relevance in the context of nutrient allocation between
# # cells in multicellular organisms. In particular, we show
# # that the competition leads to increasing the efficiencies
# # consistently with observations.

# book.add_vertex("competition for infinite resource")
# book.add_vertex("competition of plants for light")
# book.add_vertex("Sun")
# book.add_vertex("thermal bath")
# book.add_vertex("high temperature photons")
# book.add_vertex("photosynthesis")
# book.add_vertex("limited energy current")
# book.add_vertex("forest surface")
# book.add_vertex("plants")
# book.add_vertex("confrontation")
# book.add_vertex("avoidance of the competitor")
# book.add_vertex("nutrient allocation")
# book.add_vertex("cells in multicellular organisms")
# book.add_vertex("efficiencies")

# book.add_edge("competition for infinite resource", "competition of plants for light", "analogous to")
# book.add_edge("Sun", "thermal bath", "acts as")
# book.add_edge("Sun", "high temperature photons", "provides") # more active
# book.add_edge("photosynthesis", "high temperature photons", "operates using") # more active
# book.add_edge("limited energy current", "forest surface", "reaches") # more active
# book.add_edge("plants", "confrontation", "exhibit") # more active
# book.add_edge("plants", "avoidance of the competitor", "exhibit") # more active
# book.add_edge("nutrient allocation", "cells in multicellular organisms", "between")
# book.add_edge("competition", "efficiencies", "increases") # more active


# # Exploitation of a finite source is a dynamical process,
# # since this source is depleted due to the functioning of
# # the agents themselves. We study this process in section
# # IV and show that competition favors heat engines with lower efficiencies. An example of this is the fermentation
# # (aerobic and anaerobic) and respiration pathways of ATP
# # production in yeasts [31, 32, 34, 83, 84, 86] and in solid
# # tumor cells [90–93]. Here the ATP production refers to
# # work-extraction and storage [47]. Respiratory ATP production is far more efficient than fermentation, but the
# # speed and hence the power of the fermentation path is
# # greater [76, 83, 85]. Given the available resources and
# # the presence of competition, cell choose one or the other
# # pathway of ATP production [31, 83, 86].

# book.add_vertex("exploitation of finite source")
# book.add_vertex("heat engines with lower efficiencies")
# book.add_vertex("fermentation")
# book.add_vertex("respiration")
# book.add_vertex("ATP production")
# book.add_vertex("yeasts")
# book.add_vertex("solid tumor cells")
# book.add_vertex("work-extraction")
# book.add_vertex("respiratory ATP production")
# book.add_vertex("fermentation path")
# book.add_vertex("available resources")
# book.add_vertex("cells")
# book.add_vertex("dynamical process")
# book.add_vertex("speed")

# book.add_edge("exploitation of finite source", "dynamical process", "is")
# book.add_edge("competition", "heat engines with lower efficiencies", "favors")
# book.add_edge("fermentation", "ATP production", "pathways of")
# book.add_edge("respiration", "ATP production", "pathways of")
# book.add_edge("ATP production", "work-extraction", "is") # more active
# book.add_edge("respiratory ATP production", "fermentation", "more efficient than")
# book.add_edge("fermentation path", "speed", "greater")
# book.add_edge("fermentation path", "power", "greater")
# book.add_edge("cells", "ATP production", "choose pathway of")
# book.add_edge("available resources", "cells", "given")


# # Agents competing for a depletable resource alter the
# # common environment similarly to what happens in niche
# # construction theories [98–100]. Thereby they shape the
# # selection process. Hence, we face a non-trivial gametheoretic situation, where the optimal values of power
# # and efficiency under competition are not unique. However, the environmental changes caused by the behavior
# # of competing agents are “myopic”, that is, the behavior
# # of the agents is not based on perception of the global
# # environmental state.

# book.add_vertex("agents competing for depletable resource")
# book.add_vertex("niche construction theories")
# book.add_vertex("selection process")
# book.add_vertex("game-theoretic situation")
# book.add_vertex("environmental changes")
# book.add_vertex("global environmental state")

# book.add_edge("agents competing for depletable resource", "niche construction theories", "similar to")
# book.add_edge("agents competing for depletable resource", "environmental changes", "induce") # more active
# book.add_edge("agents competing for depletable resource", "selection process", "shape")
# book.add_edge("game-theoretic situation", "power", "in")
# book.add_edge("game-theoretic situation", "efficiency", "in")
# book.add_edge("environmental changes", "agents", "cause") # more active
# book.add_edge("global environmental state", "agents", "base behaviour on") # more active

# # The common environment of competing agents changes
# # due to the very engine functioning. This fact poses the
# # problem of adaptive (i.e. structure adjusting) versus nonadaptive agents. This is analogous to the phenotype
# # adaptation that is observed in organisms [28, 77–81, 94–
# # 97]. As seen below, adaptation plays an important role
# # in selection process.

# book.add_vertex("common environment of competing agents")
# book.add_vertex("adaptive agents")
# book.add_vertex("non-adaptive agents")
# book.add_vertex("phenotype adaptation")
# book.add_vertex("engine functioning")
# book.add_vertex("adaptation")

# book.add_edge("common environment of competing agents", "engine functioning", "changes due to")  # more active
# book.add_edge("adaptive agents", "non-adaptive agents", "versus")
# book.add_edge("phenotype adaptation", "organisms", "occurs in") # more active
# book.add_edge("selection process", "adaptation", "plays role in") # more active

# # The rest of this paper is organized as follows. Next
# # section defined the heat engine model we employ. Section
# # III addresses competition for an infinite resources that
# # amounts to sharing a fixed energy current. Section IV
# # studies the competition for a finite resource. We explore
# # this situation via studying two competing agents that
# # can be adaptive or not. Sections III and IV can be read
# # independently from each other. Both sections employ
# # ideas and techniques from game theory, though no deep
# # preliminary knowledge on this subject is assumed, since
# # we provide the necessary background. We summarize in
# # the last section. All technical derivations are relegated
# # to Appendices

# book.add_vertex("heat engine model")
# book.add_vertex("competition for infinite resources")
# book.add_vertex("fixed energy current")
# book.add_vertex("competition for finite resource")
# book.add_vertex("competing agents")
# book.add_vertex("section")
# book.add_vertex("adaptive")
# book.add_vertex("non-adaptive")

# book.add_edge("heat engine model", "section", "defined in")
# book.add_edge("competition for infinite resources", "fixed energy current", "amounts to")
# book.add_edge("competition for finite resource", "section", "studied in")
# book.add_edge("competing agents", "adaptive", "can be")
# book.add_edge("competing agents", "non-adaptive", "can be")
# book.add_edge("section", "game theory", "employ") # more active


# # To model energy extraction and storage, we focused
# # on the minimal thermodynamically consistent model of a
# # heat engine [18, 52]. For further clarity, we start with the
# # explicit implementation of this model via three-energy
# # level Markov systems attached to different heat baths at
# # different temperatures; see Fig. 1 and Appendix A for details. Having an explicit model is essential for clarifying
# # the nature of the involved parameters and the extracted
# # work (stored energy). However, the model will be explored in the high-temperature (linear response) regime,
# # where the implementation details are not essential, and where it is equivalent to linear thermodynamic models
# # employed in biophysics [52].

# book.add_vertex("energy extraction")
# book.add_vertex("thermodynamically consistent model of a heat engine")
# book.add_vertex("three-energy level Markov systems")
# book.add_vertex("heat baths")
# book.add_vertex("high-temperature regime")
# book.add_vertex("linear thermodynamic models")
# book.add_vertex("biophysics")
# book.add_vertex("model")

# book.add_edge("energy extraction", "thermodynamically consistent model of a heat engine", "models")
# book.add_edge("energy storage", "thermodynamically consistent model of a heat engine", "models")
# book.add_edge("three-energy level Markov systems", "heat baths", "attached to")
# book.add_edge("high-temperature regime", "model", "explore")
# book.add_edge("linear thermodynamic models", "biophysics", "uses")


# # The engine has three states i = 1, 2, 3. This is the
# # minimal number of states a stationary heat engine can
# # have, because it should be in a non-equilibrium state (i.e.
# # to support one cyclic motion), and has to support three
# # external objects, one work-source and two thermal baths.
# # Each state i has energy Ei
# # , such that
# # E1 = 0 < E2 < E3. (1)

# book.add_vertex("engine")
# book.add_vertex("three states")
# book.add_vertex("non-equilibrium state")
# book.add_vertex("cyclic motion")
# book.add_vertex("external objects")
# book.add_vertex("work-source")
# book.add_vertex("thermal baths")
# book.add_vertex("energy")
# book.add_vertex("state i") # i = 1, 2, 3
# book.add_vertex("energy Ei")

# book.add_edge("engine", "three states", "has")
# book.add_edge("engine", "non-equilibrium state", "should be in")
# book.add_edge("non-equilibrium state", "cyclic motion", "supports")
# book.add_edge("engine", "external objects", "supports")
# book.add_edge("external objects", "work-source", "include")
# book.add_edge("external objects", "thermal baths", "include")
# book.add_edge("state i", "energy Ei", "has")
# book.add_edge("state i", "three states", "is one of")

# # Transitions between each pair of different states are
# # caused by the different thermal baths having different
# # temperatures (T_h, T_c, T) that accordingly provide or accept necessary energies; cf. Fig. 1. We assume that these
# # thermal baths are in thermal equilibrium states, which
# # means that the transition rates that drive Markov evolution of the engines obey the detailed-balance condition,
# # for example the transition rates between states {1, 3} satisfy the following relation
# # \rho_{1←3} e^(−\beta_hE_3) = \rho_{3←1} e^(−\beta_hE_1)
# # , (2)
# # where \beta_h = 1/T_h. Similar relation holds for the transition rates \rho{1←2} and \rho{2←3} caused by thermal baths with
# # temperature T and T_c, respectively.

# book.add_vertex("transitions between states")
# book.add_vertex("temperatures")
# book.add_vertex("energies")
# book.add_vertex("thermal equilibrium states")
# book.add_vertex("transition rates")
# book.add_vertex("Markov evolution of the engines")
# book.add_vertex("detailed-balance condition")

# book.add_vertex("temperature Th") #Temperature of hot bath
# book.add_vertex("temperature Tc") #Temperature of cold bath
# book.add_vertex("transition rate rho1<-3")
# book.add_vertex("transition rate rho3<-1")
# book.add_vertex("beta h")

# book.add_edge("transitions between states", "thermal baths", "result from")
# book.add_edge("thermal baths", "temperatures", "have")
# book.add_edge("thermal baths", "energies", "provide or accept")
# book.add_edge("thermal baths", "thermal equilibrium states", "are in")
# book.add_edge("transition rates", "Markov evolution of the engines", "drive")
# book.add_edge("transition rates", "detailed-balance condition", "obey")

# book.add_edge("temperature Th", "thermal baths", "is of the hot")
# book.add_edge("temperature Tc", "thermal baths", "is of the cold")
# book.add_edge("transition rate rho1<-3", "transitions between states", "is the rate of")
# book.add_edge("transition rate rho3<-1", "transitions between states", "is the rate of")
# book.add_edge("beta h", "temperature Th", "inverse of")


# # One temperature is assumed to be infinite [37]: \beta =
# # 1/T = 0. This bath is then a work-source. This key
# # point can be explained as follows. First, note that an
# # infinite temperature thermal bath exchanges dE energy
# # without changing its own entropy, dS = \betadE = 0, which
# # is a feature of mechanical device (a sources of work) [37].
# # Second, if the T = \inf-bath spontaneously interacts with
# # any (positive) temperature bath, then the former bath
# # always looses energy. Hence, its energy is freely convertible to any form of heat, as expected from work. Next, we
# # assume T_h > T_c, as necessary for heat engine operation.

# book.add_vertex("temperature")
# book.add_vertex("beta") # Inverse temperature
# book.add_vertex("infinite temperature thermal bath")
# book.add_vertex("dE energy")
# book.add_vertex("dS entropy")
# book.add_vertex("mechanical device")
# book.add_vertex("temperature T")
# book.add_vertex("zero")

# book.add_edge("beta", "temperature", "is the inverse of")
# book.add_edge("work-source", "infinite temperature thermal bath", "is")
# book.add_edge("infinite temperature thermal bath", "dE energy", "exchanges")
# book.add_edge("infinite temperature thermal bath", "dS entropy", "without changing")
# book.add_edge("dS entropy", "zero", "equals")
# book.add_edge("mechanical device", "infinite temperature thermal bath", "characteristic of")
# book.add_edge("infinite temperature thermal bath", "energy", "converts freely to heat")
# book.add_edge("temperature Th", "temperature Tc", "greater than")


# # In the stationary (but generally non-equilibrium) state
# # the average energy of the three-level system is constant
# # 3
# # i=1
# # dpi
# # dt
# # Ei = Jh + Jc + J = 0,
# # (3)
# # where pi is the probability of finding the system in en-
# # ergy level Ei. Here J and Jn with n = h, c are the av-
# # erage energy lost (for J, Jn > 0) or gain (J, Jn < 0) by
# # each bath per unit of time. Eq. (3) is the first law of
# # thermodynamics for a stationary state [44]. Now (3) in-
# # dicates on a perfect coupling between the thermal baths
# # and the three-level system: there is no an energy cur-
# # rent standing for irreversible losses within the system;
# # cf. [27, 32, 85, 87-89].

# book.add_vertex("stationary non-equilibrium state")
# book.add_vertex("average energy of three-level system")
# book.add_vertex("constant average energy")
# book.add_vertex("probability pi")
# book.add_vertex("energy level Ei")
# book.add_vertex("average energy lost J")
# book.add_vertex("average energy gain J")
# book.add_vertex("three-level system")
# book.add_vertex("first law of thermodynamics for stationary state")
# book.add_vertex("average energy lost Jh")
# book.add_vertex("average energy lost Jc")

# book.add_vertex("dpi/dt") #Time derivative of probability pi
# book.add_vertex("system")
# book.add_vertex("bath")
# book.add_vertex("equation (3)")

# book.add_edge("stationary non-equilibrium state", "average energy of three-level system", "has")
# book.add_edge("average energy of three-level system", "constant average energy", "is")
# book.add_edge("probability pi", "system", "finds in energy level")
# book.add_edge("energy level Ei", "system", "finds with probability")
# book.add_edge("average energy lost J", "bath", "loses")
# book.add_edge("average energy gain J", "bath", "gains")
# book.add_edge("first law of thermodynamics for stationary state", "equation (3)", "is")
# book.add_edge("thermal baths", "three-level system", "perfectly couple")
# book.add_edge("dpi/dt", "probability pi", "is the time derivative of")

# book.add_edge("average energy lost Jh", "bath", "loses")
# book.add_edge("average energy lost Jc", "bath", "loses")


# # In the stationary state, the energy currents hold
# # J = E2/Z p2←1p1←3p3←2(1 − e^(βc−βh)E3−βcE2), (4)
# # Jh = −E3J/E2, Jc = (E3 − E2)J/E2. (5)
# # where Z is the normalization factor defined by transition
# # rates (see Appendix A).
# # If the system functions as a heat engine, i.e. on aver-
# # age, pumps energy to the work-source, then
# # J < 0, Jh > 0, Jc < 0, (6)
# # Using Eq.(5,4) and the condition (6) one get the condi-
# # tion for the system to operate as a heat-engine
# # E2 [(1 − ϑ)E3 − E2] > 0, ϑ ≡ Tc/Th = βh/βc. (7)
# # The efficiency of any heat engine is defined as the result
# # (the extracted work) divided over the resource (the en-
# # ergy coming from the hot bath). Recalling E3 > E2, the
# # efficiency η reads from (7):
# # η ≡ −J/Jh = E2/E3 ≤ ηC ≡ 1 − ϑ. (8)
# # Hence, the efficiency η is bounded from the above by
# # the Carnot efficiency ηC. Eq. (8) is the second law of
# # thermodynamics for the heat engine efficiency [44].

# book.add_vertex("stationary state")
# book.add_vertex("normalization factor Z")
# book.add_vertex("extracted work")
# book.add_vertex("resource from hot bath")
# book.add_vertex("efficiency eta")
# book.add_vertex("Carnot efficiency etaC")
# book.add_vertex("second law of thermodynamics for heat engine efficiency")

# book.add_vertex("beta c") # Inverse temperature of cold bath
# book.add_vertex("rho 2<-1")
# book.add_vertex("rho 1<-3")
# book.add_vertex("rho 3<-2")
# book.add_vertex("Theta") #Temperature ratio between cold and hot baths

# book.add_edge("stationary state", "energy currents", "hold")
# book.add_edge("normalization factor Z", "transition rates", "defines")
# book.add_edge("system", "heat engine", "functions as")
# book.add_edge("system", "work-source", "pumps energy to")
# book.add_edge("extracted work", "heat engine", "result of")
# book.add_edge("resource from hot bath", "heat engine", "provides")
# book.add_edge("efficiency eta", "extracted work", "is ratio of")
# book.add_edge("efficiency eta", "resource from hot bath", "to")
# book.add_edge("efficiency eta", "Carnot efficiency etaC", "bounded by")
# book.add_edge("second law of thermodynamics for heat engine efficiency", "efficiency eta", "defines bound for")

# book.add_edge("Theta", "temperature Tc", "ratio with")
# book.add_edge("Theta", "temperature Th", "is")


# # Eqs. (4, 7, 8) demonstrate the power-efficiency tradeoff: at the maximum efficiency \eta = \eta_C the power −J of
# # the heat engine nullifies. This trade-off is also a general
# # feature of heat engines [19]. A clear understanding of this
# # trade-off is one pertinent reason for having an explicit
# # microscopic model of a heat engine

# book.add_vertex("maximum efficiency eta")
# book.add_vertex("power -J")
# book.add_vertex("explicit microscopic model of heat engine")
# book.add_vertex("equation 4")
# book.add_vertex("equation 7")
# book.add_vertex("equation 8")


# book.add_edge("equation 4", "power-efficiency tradeoff", "demonstrates")
# book.add_edge("equation 7", "power-efficiency tradeoff", "demonstrates")
# book.add_edge("equation 8", "power-efficiency tradeoff", "demonstrates")
# book.add_edge("maximum efficiency eta", "Carnot efficiency etaC", "equals")
# book.add_edge("power -J", "heat engine", "nullifies at maximum efficiency")
# book.add_edge("explicit microscopic model of heat engine", "power-efficiency tradeoff", "helps understand")


# # The work power −J depends on the specific form of the
# # transition rates that enter the detailed balance condition
# # (2) (see Appendix A); for example the Arrhenius form
# # of transition rates applies in chemical reaction dynamics
# # [43]. Here we work in the high-temperature limit, where
# # the details of rates are not important provided they hold
# # the detailed balance. Now E_i/T_c << 1 and E_i/T_h << 1,
# # but 0 ≤ ϑ ≤ 1 in (7) can be arbitrary. In this limit the
# # heat engine power reads via Eq.(4, 8):
# # −J = ρβcE2^3η(1 − ϑ − η), (9)

# # where ρ =1/Z(ρ2←1ρ1←3ρ3←2)|βh=βc=0 is a constant.
# # Eq. (9) shows that for a fixed ϑ the maximum power
# # of |J| of the engine is attained for
# # η = (1-ϑ)/2=ηc/2 (10)

# book.add_vertex("work power -J")
# book.add_vertex("detailed balance condition")
# book.add_vertex("Arrhenius form of transition rates")
# book.add_vertex("chemical reaction dynamics")
# book.add_vertex("high-temperature limit")
# book.add_vertex("Ei/Tc")
# book.add_vertex("Ei/Th")
# book.add_vertex("heat engine power")
# book.add_vertex("constant rho")
# book.add_vertex("maximum power |J| of engine")

# book.add_edge("work power -J", "transition rates", "depends on")
# book.add_edge("transition rates", "detailed balance condition", "enter")
# book.add_edge("Arrhenius form of transition rates", "chemical reaction dynamics", "applies in")
# book.add_edge("transition rates", "high-temperature limit", "details not important")
# book.add_edge("heat engine power", "Theta", "depends on")
# book.add_edge("maximum power |J| of engine", "efficiency eta", "attained for")
# book.add_edge("constant rho", "beta h", "depends on")
# book.add_edge("constant rho", "beta c", "depends on")


# # Below we shall heuristically apply the heat engine
# # model to ATP production, where the two thermal baths
# # refer to the e.g. glucose (the major reactant of the ATP
# # production), while the work corresponds to the energy
# # stored in ATP, which is metastable at physiological conditions. In this context let us discuss to which extent heatengine models can be applied to transformations of chemical energy. The standard understanding of the chemical
# # energy stored in certain molecular degrees of freedom is
# # that it is isothermal and is described by the (Gibbs) free
# # energy difference between reactants and products. This
# # coarse-grained description does not tell where (in which
# # degrees of freedom) the energy was stored and how it
# # came out. Detailed mechanisms of such processes are still
# # unclear, e.g. there is a long-standing and on-going debate
# # on how precisely ATP delivers the stored energy during
# # its hydrolysis and how this energy is employed for doing
# # work; see e.g. [45–47]. However, it is clear that at sufficiently microscopic level all types of stored energy should
# # be related to the fact that certain degrees of freedom are
# # not in the thermal equilibrium with the environment [47].
# # Indeed, if all degrees of freedom would be thermalized
# # at the same temperature, the second law will not allow
# # any work-extraction 1
# # . It is known that frequently such
# # non-thermalized degrees of freedom can be described by
# # different effective temperatures [48]. Moreover, even a
# # finite non-equilibrium system can (under certain conditions) play the role of a thermal bath, since the dynamics
# # of its subsystem obeys the detailed balance condition [49].
# # Thus when describing work-extraction from chemical energy, it is meaningful to assume two different thermal
# # baths, which is in fact the simplest situation of a nonequilibrium system. Modeling work-extraction through
# # different chemical potentials (a situation closer to the
# # standard understanding of the stored chemical energy) is
# # in fact structurally similar to heat-engines [50, 51], also
# # because we work in the high-temperature limit, where
# # many implementation details are irrelevant. In this limit
# # our model is fully consistent with linear equilibrium thermodynamics [18, 52]. Similar models have been widely
# # employed in bioenergetics for modeling coupled chemical reactions, where the passage of heat from higher to lower temperatures corresponds to the down-hill reaction, whereas work extraction corresponds to the up-hill
# # reaction [52].

# book.add_vertex("glucose")
# book.add_vertex("energy stored in ATP")
# book.add_vertex("transformations of chemical energy")
# book.add_vertex("isothermal chemical energy")
# book.add_vertex("Gibbs free energy difference")
# book.add_vertex("reactants")
# book.add_vertex("products")
# book.add_vertex("detailed mechanisms of chemical processes")
# book.add_vertex("ATP hydrolysis")
# book.add_vertex("microscopic level of stored energy")
# book.add_vertex("thermal equilibrium with the environment")
# book.add_vertex("non-thermalized degrees of freedom")
# book.add_vertex("effective temperatures")
# book.add_vertex("finite non-equilibrium system")
# book.add_vertex("dynamics of subsystem")
# book.add_vertex("work-extraction from chemical energy")
# book.add_vertex("chemical potentials")
# book.add_vertex("linear equilibrium thermodynamics")
# book.add_vertex("bioenergetics")
# book.add_vertex("coupled chemical reactions")
# book.add_vertex("down-hill reaction")
# book.add_vertex("up-hill reaction")
# book.add_vertex("ATP")

# book.add_edge("thermal baths", "glucose", "refer to")
# book.add_edge("work", "energy stored in ATP", "corresponds to")
# book.add_edge("Gibbs free energy difference", "isothermal chemical energy", "describes")
# book.add_edge("reactants", "Gibbs free energy difference", "related to")
# book.add_edge("products", "Gibbs free energy difference", "related to")
# book.add_edge("ATP hydrolysis", "ATP", "undergoes")
# book.add_edge("microscopic level of stored energy", "thermal equilibrium with the environment", "related to lacking")
# book.add_edge("non-thermalized degrees of freedom", "effective temperatures", "can be described by")
# book.add_edge("finite non-equilibrium system", "thermal bath", "can play role of")
# book.add_edge("dynamics of subsystem", "detailed balance condition", "obeys")
# book.add_edge("thermal baths", "work-extraction from chemical energy", "meaningful to assume")
# book.add_edge("chemical potentials", "work-extraction", "modeled through")
# book.add_edge("linear equilibrium thermodynamics", "heat engine model", "consistent with")
# book.add_edge("bioenergetics", "coupled chemical reactions", "employs models for")
# book.add_edge("coupled chemical reactions", "down-hill reaction", "correspond to passage of heat")
# book.add_edge("coupled chemical reactions", "up-hill reaction", "work extraction corresponds to")


visualize_graph_with_equivalent_elements(book)

metrics = GraphMetrics(book)
metrics.full_metrics_plot()

