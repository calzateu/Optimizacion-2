import numpy as np
import trimesh as tm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay as dl
import random
import matplotlib.pyplot as plt
from collections import defaultdict

def cargar_modelo(ruta):
    mesh = tm.load("Modelos_3D/face_1.obj", file_type='obj')
    v = mesh.vertices
    f = dl(v[:,(0,1)])
    mesh = tm.Trimesh(vertices=v, faces=f.simplices)

    return mesh

class NSGA_II:
    def __init__(self, vertices, num_population, maximo_porcentaje_puntos_a_quitar, num_generaciones, tasa_mutación):
        self.vertices = vertices
        self.num_population = num_population
        self.maximo_porcentaje_a_quitar = maximo_porcentaje_puntos_a_quitar
        self.num_generaciones = num_generaciones
        self.tasa_mutación = tasa_mutación

        self.numero_vertices = len(self.vertices)

        self.population = None
        self.fitness_values = None
        self.fronteras = None
        self.crowdingdistances = None

    def population_initialization(self):
        population = []
        for i in range(self.num_population):
            individuo = np.ones(len(self.vertices))
            numero_puntos_a_quitar = self._numero_puntos_a_quitar(len(self.vertices), self.maximo_porcentaje_a_quitar)
            individuo = self._quitar_puntos(individuo, numero_puntos_a_quitar)
            population.append(individuo)

        self.population = np.array(population)

    def _numero_puntos_a_quitar(self, numero_puntos_originales, maximo_porcentaje_a_quitar):
        maximo_puntos_a_quitar = int(maximo_porcentaje_a_quitar * numero_puntos_originales)
        puntos_a_quitar = np.random.randint(maximo_puntos_a_quitar, numero_puntos_originales)

        return puntos_a_quitar

    def _quitar_puntos(self, puntos, num_puntos_a_quitar):
        for i in range(num_puntos_a_quitar):
            puntos[random.randint(0, self.numero_vertices-1)] = 0

        return puntos

    def crear_offspring(self):
        offspring = np.zeros(self.population.shape)
        for i in range(self.population.shape[0]):
            for j in range(self.population.shape[1]):
                if random.random() < self.tasa_mutación:
                    offspring[i,j] = 1 if self.population[i,j] == 0 else 0

        self.population = np.concatenate((self.population, offspring), axis=0)
        return offspring

    def evaluation(self):
        fitness_values = np.zeros((self.population.shape[0], 2)) # because of 2 objective functions
        for i, chromosome in enumerate(self.population):
            faces = dl(self.vertices[chromosome == 1][:,(0,1)]) # Delaunay triangulation con los x y y
            mesh = tm.Trimesh(vertices=self.vertices[chromosome == 1], faces=faces.simplices)
            for j in range(2):
                if j == 0:      # objective 1
                    fitness_values[i,j] = mesh.area
                elif j == 1:     # objective 2
                    fitness_values[i,j] = len(faces.simplices)

        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a

        fitness_values = totuple(fitness_values)

        self.fitness_values = fitness_values

    def _dominates(self, obj1, obj2, sign=[-1, -1]):
        """Return true if each objective of *self* is not strictly worse than
                the corresponding objective of *other* and at least one objective is
                strictly better.
            **no need to care about the equal cases
            (Cuz equal cases mean they are non-dominators)
        :param obj1: a list of multiple objective values
        :type obj1: numpy.ndarray
        :param obj2: a list of multiple objective values
        :type obj2: numpy.ndarray
        :param sign: target types. positive means maximize and otherwise minimize.
        :type sign: list
        """
        indicator = False
        for a, b, sign in zip(obj1, obj2, sign):
            if a * sign > b * sign:
                indicator = True
            # if one of the objectives is dominated, then return False
            elif a * sign < b * sign:
                return False
        return indicator

    def sortNondominated(self, k=None, first_front_only=False):
        """Sort the first *k* *individuals* into different nondomination levels
            using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
            see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
            where :math:`M` is the number of objectives and :math:`N` the number of
            individuals.
            :param individuals: A list of individuals to select from.
            :param k: The number of individuals to select.
            :param first_front_only: If :obj:`True` sort only the first front and
                                        exit.
            :param sign: indicate the objectives are maximized or minimized
            :returns: A list of Pareto fronts (lists), the first list includes
                        nondominated individuals.
            .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
                non-dominated sorting genetic algorithm for multi-objective
                optimization: NSGA-II", 2002.
        """
        if k is None:
            k = len(self.fitness_values)

        # Use objectives as keys to make python dictionary
        map_fit_ind = defaultdict(list)
        for i, f_value in enumerate(self.fitness_values):  # fitness = [(1, 2), (2, 2), (3, 1), (1, 4), (1, 1)...]
            map_fit_ind[f_value].append(i)
        fits = list(map_fit_ind.keys())  # fitness values

        current_front = []
        next_front = []
        dominating_fits = defaultdict(int)  # n (The number of people dominate you)
        dominated_fits = defaultdict(list)  # Sp (The people you dominate)

        # Rank first Pareto front
        # *fits* is a iterable list of chromosomes. Each has multiple objectives.
        for i, fit_i in enumerate(fits):
            for fit_j in fits[i + 1:]:
                # Eventhougn equals or empty list, n & Sp won't be affected
                if self._dominates(fit_i, fit_j):
                    dominating_fits[fit_j] += 1
                    dominated_fits[fit_i].append(fit_j)
                elif self._dominates(fit_j, fit_i):
                    dominating_fits[fit_i] += 1
                    dominated_fits[fit_j].append(fit_i)
            if dominating_fits[fit_i] == 0:
                current_front.append(fit_i)

        fronts = [[]]  # The first front
        for fit in current_front:
            fronts[-1].extend(map_fit_ind[fit])
        pareto_sorted = len(fronts[-1])

        # Rank the next front until all individuals are sorted or
        # the given number of individual are sorted.
        # If Sn=0 then the set of objectives belongs to the next front
        if not first_front_only:  # first front only
            N = min(len(self.fitness_values), k)
            while pareto_sorted < N:
                fronts.append([])
                for fit_p in current_front:
                    # Iterate Sn in current fronts
                    for fit_d in dominated_fits[fit_p]: 
                        dominating_fits[fit_d] -= 1  # Next front -> Sn - 1
                        if dominating_fits[fit_d] == 0:  # Sn=0 -> next front
                            next_front.append(fit_d)
                            # Count and append chromosomes with same objectives
                            pareto_sorted += len(map_fit_ind[fit_d]) 
                            fronts[-1].extend(map_fit_ind[fit_d])
                current_front = next_front
                next_front = []

        self.fronteras = fronts

    def CrowdingDist(self):
        """
        :param fitness: A list of fitness values
        :return: A list of crowding distances of chrmosomes

        The crowding-distance computation requires sorting the population according to each objective function value 
        in ascending order of magnitude. Thereafter, for each objective function, the boundary solutions (solutions with smallest and largest function values) 
        are assigned an infinite distance value. All other intermediate solutions are assigned a distance value equal to 
        the absolute normalized difference in the function values of two adjacent solutions.
        """

        # initialize list: [0.0, 0.0, 0.0, ...]
        distances = [0.0] * len(self.fitness_values)
        crowd = [(f_value, i) for i, f_value in enumerate(self.fitness_values)]  # create keys for fitness values

        n_obj = len(self.fitness_values[0])

        for i in range(n_obj):  # calculate for each objective
            crowd.sort(key=lambda element: element[0][i])
            # After sorting,  boundary solutions are assigned Inf 
            # crowd: [([obj_1, obj_2, ...], i_0), ([obj_1, obj_2, ...], i_1), ...]
            distances[crowd[0][1]] = float("Inf")
            distances[crowd[-1][1]] = float("inf")
            if crowd[-1][0][i] == crowd[0][0][i]:  # If objective values are same, skip this loop
                continue
            # normalization (max - min) as Denominator
            norm = float(crowd[-1][0][i] - crowd[0][0][i])
            # crowd: [([obj_1, obj_2, ...], i_0), ([obj_1, obj_2, ...], i_1), ...]
            # calculate each individual's Crowding Distance of i th objective
            # technique: shift the list and zip
            for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
                distances[cur[1]] += (next[0][i] - prev[0][i]) / norm  # sum up the distance of ith individual along each of the objectives

        self.crowdingdistances = distances

    def graficar_pareto(self):
        fitness_values = np.array(self.fitness_values)
        plt.plot(fitness_values[:,0], fitness_values[:,1], 'o')
        plt.plot(fitness_values[self.fronteras[0],0], fitness_values[self.fronteras[0],1], 'ro')
        plt.show()

    def mostrar_individuos(self, num_individuos_a_mostrar):
        print(f'Cantidad real individuos {self.population.shape[0]}')
        for i in range(num_individuos_a_mostrar):
            puntos_filtrados = self.vertices[self.population[i] == 1]
            faces = dl(puntos_filtrados[:,(0,1)]) # Delaunay triangulation con los x y y
            tm.Trimesh(vertices=self.vertices[self.population[i] == 1], faces=faces.simplices).show()

    def run(self):
        for i in range(self.num_generaciones):
            self.population_initialization()
            self.crear_offspring()
            self.evaluation()
            self.sortNondominated()
            self.CrowdingDist()
            padres = []
            # while len(padres) < self.num_population:
            #     padres.append(self.population[self.fronteras[0][0]])
            print(len(self.crowdingdistances))


if __name__ == "__main__":
    mesh = cargar_modelo("Modelos_3D/face_1.obj")
    optimizador = NSGA_II(mesh.vertices, 100, 0.4, 10, 0.5)

    optimizador.run()



