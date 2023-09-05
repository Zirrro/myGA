# -*- coding: utf-8 -*-
# @Time    : 2023/7/1 9:25
# @Author  : Salieri
# @FileName: transfer.py
# @Software: PyCharm
# @Comment : GA algorithm

import pygad
import numpy
import matplotlib.pyplot as plt
from cores.algorithm.TriaxialSynthesisTotalField.CorrectTriaxial import CorrectTriaxial


def fitness_func(solution, solution_idx, function_inputs, desired_output):
    alpha = solution[0]
    beta = solution[1]
    gamma = solution[2]
    k01 = solution[3]
    k02 = solution[4]
    k03 = solution[5]
    offset01 = solution[6]
    offset02 = solution[7]
    offset03 = solution[8]
    dataresult = CorrectTriaxial(function_inputs, alpha, beta, gamma, k01, k02, k03, offset01, offset02, offset03)
    fitness = 1.0 / (numpy.sum((dataresult[:, 3] - desired_output) ** 2) / len(dataresult[:, 3]))

    return fitness


def callback_generation(ga_instance):
    print("Generation=%s,Fitness=%s" % (ga_instance.generations_completed, ga_instance.best_solution()[1]))


def runGA(time, function_inputs, desired_output, **kwargs):
    num_generations = kwargs.pop('num_generations')
    num_parents_mating = kwargs.pop('num_parents_mating')
    fitness_function = lambda solution, solution_idx: fitness_func(solution, solution_idx, function_inputs,
                                                                   desired_output)
    sol_per_pop = kwargs.pop('sol_per_pop')
    num_genes = kwargs.pop('num_genes')
    parent_selection_type = kwargs.pop('parent_selection_type')
    crossover_type = kwargs.pop('crossover_type')
    keep_parents = kwargs.pop('keep_parents')
    mutation_type = kwargs.pop('mutation_type')
    mutation_percent_genes = kwargs.pop('mutation_percent_genes')
    gene_space = kwargs.pop('gene_space')

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           gene_type=numpy.float32,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           keep_elitism=1,
                           gene_space=gene_space,
                           on_generation=callback_generation,
                           )

    ga_instance.run()
    # ga_instance.plot_result()
    ga_instance.plot_fitness()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    dataresult = CorrectTriaxial(function_inputs, solution[0], solution[1], solution[2], solution[3], solution[4],
                                 solution[5], solution[6], solution[7], solution[8])

    plt.subplots()
    plt.subplot(3, 1, 1)
    plt.plot(time, numpy.sqrt(numpy.sum(numpy.square(function_inputs.T), axis=0)))
    plt.plot(time, dataresult[:, 3])
    plt.xlabel('Time(s)')
    plt.ylabel('B(uT)')
    plt.gca().ticklabel_format(useOffset=False)
    plt.legend(['Original magnetic field', 'Corrected magnetic field'], loc=1)
    plt.subplot(3, 1, 2)
    plt.plot(time, desired_output)
    plt.plot(time, dataresult[:, 3])
    plt.xlabel('Time(s)')
    plt.ylabel('B(uT)')
    plt.gca().ticklabel_format(useOffset=False)
    plt.legend(['Target total field', 'Corrected magnetic field'], loc=1)
    plt.subplot(3, 1, 3)
    plt.plot(time, dataresult[:, 3] - desired_output)
    plt.xlabel('Time(s)')
    plt.ylabel('B(uT)')
    plt.gca().ticklabel_format(useOffset=False)
    plt.legend(['Residual magnetic field:std=%snT' % round(numpy.std(dataresult[:, 3] - desired_output) * 1000, 2)],
               loc=1)
    plt.show()
    res = dataresult[:, 3] - desired_output
    print('10Hz', numpy.std(res[::20]) * 1000)
    return solution, solution_fitness, dataresult
