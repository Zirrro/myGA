# -*- coding: utf-8 -*-
# @Time    : 2023/8/16 15:33
# @Author  : Salieri
# @FileName: mag629三轴校正.py
# @Software: PyCharm
# @Comment : 使用遗传算法的三轴合成程序，输入为5列的mat文件，分别为时间，三轴数据以及参考总场。

from cores.preprocessing.FileParsing.loadData import *
from PyQt5.QtCore import QSettings, QDateTime
from cores.algorithm.TriaxialSynthesisTotalField.GeneticAlgorithm import runGA
import cores.preprocessing.FileParsing.transfer as transfer
import matplotlib.pyplot as plt
import time as pytime
import datetime


def loadParaINI():
    filepath = r'./db/para.ini'
    settings = QSettings(filepath, QSettings.IniFormat)
    settings.setIniCodec('UTF-8')
    return settings


if __name__ == "__main__":
    fs = 10
    data_path = r'./data/mag690三轴校正数据-60s.mat'
    dsave = transfer.mat_to_nparray(data_path)
    # dsave[:, 4] = dsave[:, 4].mean()
    dsave = dsave[75 * fs:120 * fs]

    plt.subplots()
    plt.subplot(2, 1, 1)
    plt.plot(dsave[:, 0], dsave[:, 1:4])
    plt.xlabel('Time(s)')
    plt.ylabel('B(nT)')
    plt.legend(['Bx', 'By', 'Bz'], loc=1)
    plt.subplot(2, 1, 2)
    plt.plot(dsave[:, 0], np.sqrt(dsave[:, 1] ** 2 + dsave[:, 2] ** 2 + dsave[:, 3] ** 2))
    plt.plot(dsave[:, 0], dsave[:, 4])
    plt.xlabel('Time(s)')
    plt.ylabel('B(nT)')
    plt.show()

    print(">>三轴校正......")
    timestart = pytime.time()
    function_inputs = dsave[:, 1:4] / 1000
    desired_output = dsave[:, 4] / 1000
    time = [i / fs for i in range(len(function_inputs))]

    type = 'mag690'
    savename = '%s-%s' % (type, np.random.randint(0, 1000))
    num_generations = 500  # 世代数

    sol_per_pop = 500  # 种群中解（即染色体）的数量。
    num_genes = 9  # 染色体中的基因数量。

    num_parents_mating = 250  # 被选为父母的解决方案数量种群大小的一半

    parent_selection_type = "rws"  # 父选择类型。支持的类型是sss（用于稳态选择）、rws（用于轮盘选择）、sus（用于随机通用选择）、 rank（用于排名选择）random、（用于随机选择）和 tournament（用于锦标赛选择）。
    keep_parents = 250  # 当前人口中要保留的父母人数。-1（默认）意味着将所有父母保留在下一个种群中。0意味着在下一个种群中不保留父母。一个值意味着在下一个种群中保留指定数量的父母。

    crossover_type = "scattered"  # 交叉操作的类型。支持的类型有single_point（单点交叉）、 two_points（两点交叉）、uniform（均匀交叉）和scattered（分散交叉）。

    mutation_type = "random"  # 变异操作的类型。支持的类型是random（对于随机变异）、swap（对于交换变异）、inversion（对于反转变异）、scramble（对于争夺变异）和adaptive（对于自适应变异）。
    mutation_percent_genes = 20  # 突变基因的百分比。它默认为字符串"default"，稍后将转换为整数10，这意味着 10% 的基因将发生突变。

    # Mag629 使用的参数，可以达到0.99nT的精度
    # gene_space = [{'low': -1, 'high': 1}, {'low': -1, 'high': 1}, {'low': -1, 'high': 1},
    #               {'low': 0.9, 'high': 1.1}, {'low': 0.9, 'high': 1.1}, {'low': 0.9, 'high': 1.1},
    #               {'low': -0.1, 'high': 0.1}, {'low': -0.1, 'high': 0.1}, {'low': -0.1, 'high': 0.1}, ]  # 指定参数范围

    gene_space = [{'low': -10, 'high': 10}, {'low': -10, 'high': 10}, {'low': -10, 'high': 10},
                  {'low': 0.9, 'high': 1.2}, {'low': 0.9, 'high': 1.2}, {'low': 0.9, 'high': 1.2},
                  {'low': -40, 'high': 40}, {'low': -40, 'high': 40}, {'low': -40, 'high': 40}, ]  # 指定参数范围

    solution, solution_fitness, dataresult = runGA(time,  # 时间
                                                   function_inputs,  # 输入
                                                   desired_output,  # 输出
                                                   num_generations=num_generations,
                                                   num_parents_mating=num_parents_mating,
                                                   sol_per_pop=sol_per_pop,
                                                   num_genes=num_genes,
                                                   parent_selection_type=parent_selection_type,
                                                   keep_parents=keep_parents,
                                                   crossover_type=crossover_type,
                                                   mutation_type=mutation_type,
                                                   mutation_percent_genes=mutation_percent_genes,
                                                   gene_space=gene_space,
                                                   )

    np.savetxt('./result/' + savename + '.txt', dataresult, fmt='%8f')

    settings = loadParaINI()
    settings.beginGroup("%s" % savename)
    timenow = QDateTime.currentDateTime()
    settings.setValue('fileName', data_path)
    settings.setValue('datatime', timenow.toString())
    settings.setValue('sensorType', type)
    settings.setValue('num_generations', num_generations)
    settings.setValue('num_parents_mating', num_parents_mating)
    settings.setValue('sol_per_pop', sol_per_pop)
    settings.setValue('parent_selection_type', parent_selection_type)
    settings.setValue('keep_parents', keep_parents)
    settings.setValue('crossover_type', crossover_type)
    settings.setValue('mutation_type', mutation_type)
    settings.setValue('mutation_percent_genes', mutation_percent_genes)
    settings.setValue('solution_fitness', str(solution_fitness))
    settings.setValue('solution', ','.join([str(i) for i in solution]))
    settings.setValue('elapsed_time', str(pytime.time() - timestart))
    settings.endGroup()
