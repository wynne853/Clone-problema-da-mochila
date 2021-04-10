# Imports necessários
import random
from deap import creator, base, tools, algorithms


def GenerateItems(size):
    itens = []

    for indice in range(size - 1):
        weight = random.randint(1, 5)
        value = random.randint(1, 20)
        itens.append({"Weight": weight, "Value": value})
    itens.append({"Weight": 0, "Value": 0})
    return itens


# ---------------------------Constantes----------------------------------------
INDIVIDUAL_SIZE = 15
BACKPACK_WEIGHT = 40
POPULATION_SIZE = 300
MAX_INTERACTION_COUNT = 10
MAX_VARIATION = 2
NUMBER_ITENS = 10
LIST_REFERENCES = GenerateItems(NUMBER_ITENS)
VARIATION_SIZE = 3

# -----------------------------------------------------------------------------
# Define o tipo fitness: Um objetivo com maximização
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Define o tipo indivíduo: indivíduo do tipo list (array) com
# a fitness definida anteriormente.
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox para inicialização de componentes do algoritmo
toolbox = base.Toolbox()

# ------------------------------------------------------------------------------

# Atributo booleano criado de forma aleatório
toolbox.register("attribute_int",
                 random.randint, 0, (NUMBER_ITENS - 1))

# Tipo de Individual
toolbox.register("individual",
                 tools.initRepeat, creator.Individual, toolbox.attribute_int, INDIVIDUAL_SIZE)

# Criação da população
toolbox.register("population",
                 tools.initRepeat, list, toolbox.individual)
# -------------------------------------------------------------------------------
# Função fitness


def evalOneMax(individual):
    fit = 0
    weight = 0
    for iten in individual:
        fit += LIST_REFERENCES[iten].get("Value", 0)
        weight += LIST_REFERENCES[iten].get("Weight", 0)
        if weight > BACKPACK_WEIGHT:
            return 0
    return fit


# Registra a função de fitness
toolbox.register("evaluate", evalOneMax)

# Crossover
toolbox.register("mate", tools.cxTwoPoint)
# Mutação def mutUniformInt(individual, low, up, indpb)
toolbox.register("mutate", tools.mutUniformInt, low=0,
                 up=(NUMBER_ITENS - 1), indpb=0.05)

# Registro do método de seleção
toolbox.register("select", tools.selTournament, tournsize=VARIATION_SIZE)

# Tamanho da população
population = toolbox.population(n=POPULATION_SIZE)

# ------------------------------------------------------------------------------

# Melhor item de todas as gerações
best = {"fit": 0, "ind": None}
# Contador
count = 0
# Iniciando o processo de evolução
while count < MAX_INTERACTION_COUNT:

    # Algoritmos evolucionários
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)

    # Avalia cada indivíduo
    fits = toolbox.map(toolbox.evaluate, offspring)

    # Melhor da população
    bestIndividual = {"fit": 0, "ind": None}

    # Associa cada indivíduo ao seu valor de fitness
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = (fit),
        if fit > bestIndividual.get("fit", 0):
            bestIndividual = {"fit": fit, "ind": ind}

    # Aplica a seleção para gerar a nova população
    population = toolbox.select(offspring, k=len(population))

    # Processar condição de parada
    bestIndividualFit = bestIndividual.get("fit")
    bestFit = best.get("fit")

    if ((bestIndividualFit >= bestFit and (bestIndividualFit <= bestFit + MAX_VARIATION)) or ((bestIndividualFit < bestFit and bestIndividualFit >= bestFit - MAX_VARIATION))):
        count += 1
    else:
        count = 0

    best = (bestFit > bestIndividualFit) and best or bestIndividual

# Imprime o melhor
print(best.get("ind"))
