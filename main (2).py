import datetime
import numpy as np
import pyopencl as cl
import pyopencl.array
import random
import matplotlib.pyplot as plt
import time

from matplotlib import ticker


def cm_to_inch(value):
    return value / 2.54

#построение графиков
def plot_graphs_errorv(crit, fitness, iGeneration, N, dimensions):
    fig, ax = plt.subplots(figsize=(cm_to_inch(25), cm_to_inch(25)))
    for i in range(0, N):
        if fitness[i] == 1:
            ax.scatter(crit[2*i], crit[2*i+1], c='r')
        else:
            ax.scatter(crit[2*i], crit[2*i+1], c='b')
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid()
    ax.set_xlabel('v1')
    ax.set_ylabel('v2')
    plt.savefig('pareto' +str(iGeneration) + '.png')

def plot_graphs_error(crit, fitness, iGeneration, N, dimensions):
    fig, ax = plt.subplots(figsize=(cm_to_inch(25), cm_to_inch(25)))
    for i in range(0, N):
        ax.scatter(float(str(crit[2*i])), float(str(crit[2*i+1])), c='b')
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid()
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    plt.savefig('F' +str(iGeneration) + '.png')


#построение графиков
def plot_graphs_educt(crit, elit, iGeneration, num_of_outputs, freqplotEduct):
    #plt.figure(figsize=[60 * (iGeneration//freqplotEduct), 50])
    plt.figure(figsize=[150, 60])
    u1 = []
    u2 = []
    psi1 = []
    u1_elit = []
    u2_elit = []
    psi1_elit = []
    gens = []
    for i in range(iGeneration):
        u1.append(crit[(i * num_of_outputs) + 0])
        u2.append(crit[(i * num_of_outputs) + 1])
        psi1.append(crit[(i * num_of_outputs) + 2])
        u1_elit.append(elit[(i * num_of_outputs) + 0])
        u2_elit.append(elit[(i * num_of_outputs) + 1])
        psi1_elit.append(elit[(i * num_of_outputs) + 2])
        gens.append(i)
    u1 = np.array(u1)
    u2 = np.array(u2)
    psi1 = np.array(psi1)
    u1_elit = np.array(u1_elit)
    u2_elit = np.array(u2_elit)
    psi1_elit = np.array(psi1_elit)
    gens = np.array(gens)

    plt.subplot(2, 1, 1)
    plt.plot(gens, u1, 'g', label='Ошибка по u1')
    plt.plot(gens, u2, 'b',label='Ошибка по u2')
    plt.plot(gens, psi1, 'r',label='Ошибка по psi1')
    plt.legend(loc='upper right', fontsize=75)
    plt.scatter(gens, u1, color = 'g',s=500)
    plt.scatter(gens, u2, color = 'b',s=500)
    plt.scatter(gens, psi1, color = 'r',s=500)
    plt.title('Текущая ошибка в поколении\nЗначения ошибок на поколении №'+str(iGeneration)+': '+str(u1[iGeneration-1])+', '+str(u2[iGeneration-1])+', '+str(psi1[iGeneration-1])+'.', fontsize=75)
    plt.xlabel('Поколение', fontsize=75)
    plt.ylabel('Ошибка', fontsize=75)
    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(gens, u1_elit, 'g', label='Ошибка по u1')
    plt.plot(gens, u2_elit, 'b',label='Ошибка по u2')
    plt.plot(gens, psi1_elit, 'r',label='Ошибка по psi1')
    plt.legend(loc='upper right', fontsize=75)
    plt.scatter(gens, u1_elit, color = 'g',s=500)
    plt.scatter(gens, u2_elit, color = 'b',s=500)
    plt.scatter(gens, psi1_elit, color = 'r',s=500)
    plt.title('Наименьшие ошибки в процессе обучения\nНаименьшее значение ошибок на поколении №'+str(iGeneration)+': '+str(u1_elit[iGeneration-1])+', '+str(u2_elit[iGeneration-1])+', '+str(psi1_elit[iGeneration-1])+'.', fontsize=75)
    plt.xlabel('Поколение', fontsize=75)
    plt.ylabel('Ошибка', fontsize=75)
    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)
    plt.grid()

    plt.savefig('img/educt/educt_gen'+str(iGeneration) + '.png')
    plt.close()


#проверка критерия остановки
#на вход - текущее поколение, массив значений фитнесс-функции, объём популяции,
# максимальный процент точек, с фитнесс-функцией = 1, максимальное поколение
def isFitness(iGeneration, fitness, N,fitness_percent, nGeneration):
    fitCount = 0
    #подсчёт кол-ва точек, где фитнесс-функция = 1
    for i in range(N):
        if fitness[i]==1:
            fitCount=fitCount+1
    #подсчёт процента точек с фитнесс-функцией = 1
    fitCount = fitCount/N
    print("iGeneration "+str(iGeneration)+" - "+str(fitCount))
    #проверка соответствия критериям остановки
    if fitCount >= fitness_percent:
        return True
    return iGeneration >= nGeneration


if __name__ == '__main__':

    plotGraphs = True
    freqplotGraphs = 5

    N = 10000
    nGeneration = 10
    borders = [0, 2]
    layers = [2, 5, 5, 2]

    #params
    Q = 5
    fitness_percent = 0.9
    change_crossover = 0.95
    change_mutation = 0.2
    crossover_param = 5


    isFit = False
    iGeneration = 1

    maxLayer = max(layers)

    nLayers = len(layers)
    num_of_inputs = layers[0]
    num_of_outputs = layers[len(layers)-1]

    SizeOfWeigths = 0
    for i in range(nLayers-1):
        SizeOfWeigths = SizeOfWeigths + layers[i+1] + (layers[i]*layers[i+1])

    nNeurons = 0
    for i in range(nLayers-1):
        nNeurons = nNeurons + layers[i]

    nBias = 0
    for i in range(1, nLayers):
        nBias = nBias + layers[i]

    NeuralType = 0
    if nNeurons > (maxLayer*2):
        NeuralType = 2
    else:
        NeuralType = 1



    dimensions = 2

    borders_1d = np.asarray(borders).ravel()
    del borders

    if nGeneration % 2 != 0:
        nGeneration = nGeneration + 1

    x_line = []
    # for i in range(2*N):
    #     x_line.append(np.random.uniform(0.0, 2.0))
    # x_line = np.load("X.npy").flatten()
    x_line.append(1.0)
    x_line.append(1.0)

    z_line = []
    for i in range(2*N):
        z_line.append(np.random.uniform(0.0, 2.0))
    # z_line = np.load("Z.npy").flatten()
    num_z = len(z_line)/2


    # expect = np.loadtxt("EXC.txt")
    # expect_line = []
    # for i in range(num_z):
    #     for j in range(num_of_outputs):
    #         expect_line.append(expect[i][j])
    # del expect

    elit_crit = []
    elit_param = []

    f = open('main.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    defines = "#define N "+str(N)+"\n"+\
              "#define dimensions " + str(dimensions) + "\n"+\
              "#define fitness_percent " + str(np.float32(fitness_percent)) + "\n"+\
              "#define change_crossover " + str(np.float32(change_crossover)) + "\n"+\
              "#define change_mutation " + str(np.float32(change_mutation)) + "\n"+\
              "#define crossover_param " + str(np.int32(crossover_param)) + "\n"+\
              "#define Q " + str(Q)+"\n"+\
              "#define num_z " + str(num_z)+"\n"+\
              "#define SizeOfWeigths " + str(SizeOfWeigths)+"\n"+\
              "#define num_of_inputs " + str(num_of_inputs)+"\n"+\
              "#define num_of_outputs " + str(num_of_outputs)+"\n"+\
              "#define nLayers " + str(nLayers)+"\n"+\
              "#define nNeurons " + str(nNeurons)+"\n"+\
              "#define maxLayer " + str(maxLayer)+"\n"+\
              "#define NeuralType " + str(NeuralType)+"\n"+\
              "#define nBias " + str(np.int32(nBias))+"\n"

    kernels = defines + kernels

    prg = cl.Program(ctx, kernels).build()

    dev_x = cl.array.to_device(queue, np.array(x_line, dtype=np.double))

    dev_z = cl.array.to_device(queue, np.array(z_line, dtype=np.double))
    del x_line
    # f = [0]*2*pow(N,2)
    f = [0]*2*N
    dev_f = cl.array.to_device(queue, np.array(f, dtype=np.double))
    del f

    v = [0] * (num_of_outputs * N)
    dev_v = cl.array.to_device(queue, np.array(v, dtype=np.double))
    del v

    v_save = [0] * (2)
    dev_v_save = cl.array.to_device(queue, np.array(v_save, dtype=np.double))
    del v_save

    fit = [0] * N
    dev_fit = cl.array.to_device(queue, np.array(fit, dtype=np.double))
    del fit

    dev_borders = cl.array.to_device(queue, np.array(borders_1d, dtype=np.short))
    # dev_expect = cl.array.to_device(queue, np.array(expect_line, dtype=np.double))
    dev_layers = cl.array.to_device(queue, np.array(layers, dtype=np.uint))
    # dev_x веса и смещения - аналог x1x2
    # calculateerror - calculate f1f2
    evt = prg.CalculateF(queue, (N,), None,
                         dev_f.data,
                         dev_x.data,
                         dev_z.data,
                         dev_layers.data
                         )
    evt.wait()

    dev_v = dev_f

    #q = dev_q.get()

    evt = prg.minimum(queue, (N,), None,
                      dev_f.data,
                      dev_v_save.data
                      )
    evt.wait()
    with open("v1v2_" +str(iGeneration)+ ".txt", "w") as file:
        for item in dev_v_save:
            file.write("%s\n" % item)

    v = dev_v.get()

    evt = prg.paretoFitness(queue, (N,), None,
                            dev_fit.data,
                            dev_v.data,
                            np.int32(N)
                            )
    evt.wait()

    fit = dev_fit.get()

    for i in range(len(fit)):
        if fit[i] == 1:
            for j in range(dimensions):
                elit_param.append(z_line[(i * dimensions) + j])
            for j in range(num_of_outputs):
                elit_crit.append(v[(i * num_of_outputs) + j])
    del z_line

    isFit = isFitness(iGeneration, fit, N, fitness_percent, nGeneration)

    if plotGraphs:
        # plot_graphs_errorv(v, fit, iGeneration, N, num_of_outputs)
        plot_graphs_error(dev_f, fit, iGeneration, N, num_of_outputs)

    del v
    del fit

    while isFit != True:
        time.sleep(1)
        iGeneration = iGeneration + 1
        indexes = random.sample(range(N), N)+random.sample(range(N), N)
        parents = [0] * N

        dev_indexes = cl.array.to_device(queue, np.array(indexes, dtype=np.uint))
        del indexes

        dev_parents = cl.array.to_device(queue, np.array(parents, dtype=np.uint))
        del parents

        evt = prg.selectionTournament(queue, (N,), None,
                                dev_fit.data,
                                dev_parents.data,
                                dev_indexes.data
                                )
        evt.wait()

        #parents = dev_parents.get()

        rand_for_childs = [0] * ((int(N/2))*5)
        for i in range((int(N/2))*5):
            rand_for_childs[i] = random.random()
        dev_rand = cl.array.to_device(queue, np.array(rand_for_childs, dtype=np.double))
        del rand_for_childs

        rand_k = [0]*((int(N/2))*3)
        for i in range((int(N/2))*3):
            rand_k[i] = random.randint(0, dimensions)
        dev_rand_k = cl.array.to_device(queue, np.array(rand_k, dtype=np.ushort))
        del rand_k

        nextPopulation = [0] * N * dimensions
        dev_nextPopulation = cl.array.to_device(queue, np.array(nextPopulation, dtype=np.double))
        del nextPopulation

        evt = prg.newPopulation(queue, (int(N/2),), None,
                                dev_z.data,
                                dev_nextPopulation.data,
                                dev_parents.data,
                                dev_rand.data,
                                dev_rand_k.data,
                                dev_borders.data
                                )
        evt.wait()

        nextPopulation = dev_nextPopulation.get()

        z_line = []
        indexes = random.sample(range(N), N)
        for i in range(N):
            for j in range(dimensions):
                z_line.append(nextPopulation[(indexes[i]*dimensions)+j])
                if z_line[(i*dimensions)+j] < borders_1d[0]:
                    z_line[(i * dimensions) + j] = borders_1d[0]
                elif z_line[(i*dimensions)+j] > borders_1d[1]:
                    z_line[(i * dimensions) + j] = borders_1d[1]
        del nextPopulation

        dev_z = cl.array.to_device(queue, np.array(z_line, dtype=np.double))

        evt = prg.CalculateF(queue, (N,), None,
                             dev_f.data,
                             dev_x.data,
                             dev_z.data,
                             dev_layers.data
                             )
        evt.wait()

        dev_v = dev_f

        # q = dev_q.get()

        evt = prg.minimum(queue, (N,), None,
                          dev_f.data,
                          dev_v_save.data
                          )
        evt.wait()

        with open("v1v2_" +str(iGeneration)+ ".txt", "w") as file:
            for item in dev_v_save:
                file.write("%s\n" % item)

        evt = prg.paretoFitness(queue, (N,), None,
                                dev_fit.data,
                                dev_v.data,
                                np.int32(N)
                                )
        evt.wait()

        fit = dev_fit.get()

        v = dev_v.get()

        for i in range(len(fit)):
            if fit[i] == 1:
                for j in range(dimensions):
                    elit_param.append(z_line[(i * dimensions) + j])
                for j in range(num_of_outputs):
                    elit_crit.append(v[(i * num_of_outputs) + j])
        del z_line

        isFit = isFitness(iGeneration, fit, N, fitness_percent, nGeneration)

        if plotGraphs:
            if iGeneration % freqplotGraphs == 0 or isFit:
                # plot_graphs_errorv(v, fit, iGeneration, N, num_of_outputs)
                plot_graphs_error(dev_f, fit, iGeneration, N, num_of_outputs)

        del v
        del fit

        if not isFit:
            if (iGeneration%3==0):
                elit_fit = [0] * (int(len(elit_crit) / num_of_outputs))
                dev_elit_fit = cl.array.to_device(queue, np.array(elit_fit, dtype=np.double))
                dev_elit_crit = cl.array.to_device(queue, np.array(elit_crit, dtype=np.double))
                evt = prg.paretoFitness(queue, (int(len(elit_crit) / num_of_outputs),), None,
                                        dev_elit_fit.data,
                                        dev_elit_crit.data,
                                        np.int32(int(len(elit_crit) / num_of_outputs))
                                        )
                evt.wait()
                elit_fit = dev_elit_fit.get()
                temp_elit_param = []
                temp_elit_crit = []
                for i in range(len(elit_fit)):
                    if elit_fit[i] == 1:
                        for j in range(num_of_outputs):
                            temp_elit_crit.append(elit_crit[(i * num_of_outputs) + j])
                        for j in range(dimensions):
                            temp_elit_param.append(elit_param[(i * dimensions) + j])
                elit_param = temp_elit_param
                elit_crit = temp_elit_crit
                del temp_elit_param
                del temp_elit_crit
                del elit_fit

    elit_fit = [0]*(int(len(elit_crit)/num_of_outputs))

    dev_elit_fit = cl.array.to_device(queue, np.array(elit_fit, dtype=np.double))
    dev_elit_crit = cl.array.to_device(queue, np.array(elit_crit, dtype=np.double))

    evt = prg.paretoFitness(queue, (int(len(elit_crit)/num_of_outputs),), None,
                            dev_elit_fit.data,
                            dev_elit_crit.data,
                            np.int32(int(len(elit_crit)/num_of_outputs))
                            )
    evt.wait()

    elit_fit = dev_elit_fit.get()

    elit_elit = []
    for i in range(len(elit_fit)):
        if elit_fit[i] == 1:
            temp = [0]*dimensions
            for j in range(dimensions):
                temp[j]=elit_param[(i*dimensions)+j]
            elit_elit.append(temp)
    with open("output"+str(datetime.datetime.now().strftime("-Date-%Y-%m-%d-Time-%H-%M-%S"))+".txt", "w") as outfile:
        for item in elit_elit:
            outfile.write("%s\n" % item)

    '''
    if plotGraphs:
        plot_graphs_error(elit_crit, elit_fit, "Elit", int(len(elit_param)/dimensions), num_of_outputs)
    '''



# x(веса и смещения) z(первый слой нейронки) те же самые на вход на вход два значения это f1 а2
# у миши на выходе три параметра
# 1000 1000 матрица у меня
# 1000 на 256 на их персечении три значения
# далее идет операция мини
# векторный минимакс используется когда полученные ответы минимизируем не лучший и не худший, а по серединке
# максмин минимизируем ошибки
#
# у меня векторный максимин - даже в самых лучших слуачя получаются
#
# для кажого икса выделила два самых маленьких ф1 ф2 - поиск множества парето
# построила парето, проверяю критерий останова, вписываю поколения, если не ост, то запускаю процесс эволюции,
# z - неопределнный фактор, то, что извне влияет на
# какая цель при увеличении зет? если для увеличения ф1ф2 -
# изначально просто находим ф1ф2
# прогнали эволюцию по икс, потом по зет
# и новое поколоение
#