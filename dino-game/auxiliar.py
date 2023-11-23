import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#todo ==================[Graficas de train]==================
# # 0 = ventana
# vent_fit = np.array([2740,2916,3386,12418,12620,12805,12391,12478,12352,13324])

# # 1 = competencia
# comp_fit = np.array([1026 ,548 ,522 ,13338 ,12463 ,13384 ,13760 ,13400 ,13499 ,13520])

# # 2 = ruleta
# rule_fit = np.array([1436 ,5438 ,5339 ,5894 ,5345 ,5432 ,5453 ,6560 ,6299 ,5732])

# x = np.arange(len(vent_fit))
# fig, ax = plt.subplots()
# ax.plot(x, vent_fit, linewidth=2, linestyle='dotted', label="Ventana")
# ax.plot(x, comp_fit, linewidth=2, linestyle='dashed', label="Competencia")
# ax.plot(x, rule_fit, linewidth=2, linestyle='dashdot', label="Ruleta")
# ax.legend()
# ax.grid(True)
# ax.set(
#     xlabel='Generaciones',
#     ylabel='Fitness',
#     title='Entrenamiento por operadores de seleccion',
#     xticks=x
# )
# # ax.set_xticks(x, x+1)

# plt.show()

#todo ==================[Graficas de test]==================
# 663
struct_663 = np.array([1026 ,548 ,522 ,13338 ,12463 ,13384 ,13760 ,13400 ,13499 ,13519])

# 633
struct_633 = np.array([832 ,560 ,580 ,1206 ,914 ,1883 ,2536 ,5531 ,5165 ,5789])

# 683
struct_683 = np.array([539 ,786 ,507 ,545 ,5342 ,5444 ,5582 ,5699 ,5342 ,5366])

x = np.arange(len(struct_633))
fig, ax = plt.subplots()
ax.plot(x, struct_663, linewidth=2, linestyle='dotted', label="[6,6,3]")
ax.plot(x, struct_633, linewidth=2, linestyle='dashed', label="[6,3,3]")
ax.plot(x, struct_683, linewidth=2, linestyle='dashdot', label="[6,8,3]")
ax.legend()
ax.grid(True)
ax.set(
    xlabel='Generaciones',
    ylabel='Fitness',
    title='Entrenamiento con distintas arquitecturas',
    xticks=x
)
# ax.set_xticks(x, x+1)

plt.show()




#todo ==================[list concatenate]==================
# a = [1, 2, 3]
# b = [4, 5, 6]
# c = a + b
# print(c)
#todo ==================[list to string]==================
# my_list = [1, 2, 3]

# # 将列表转换为字符串，手动添加方括号
# my_string = str(my_list)

# print(my_string)


#todo ==================[randint]==================
# z = np.zeros(shape=(4, 4))
# a = np.random.randint(low=[0, 0], high=[1, 4], size=(4, 2))
# print(a)
# z[a[:, 0], a[:, 1]] += np.random.uniform(-1, 1, size=(4))
# print(z)

#todo ==================[check_file_existence]==================
# import os

# def check_file_existence(file_path):
#     return os.path.exists(file_path)

# # 用法示例
# file_path = "dino-game/GENETIC/brain"

# if check_file_existence(file_path):
#     print(f"The file {file_path} exists.")
# else:
#     print(f"The file {file_path} does not exist.")


#todo ==================[create_directory]==================
# import os

# def create_directory(directory_path):
#     os.makedirs(directory_path, exist_ok=True)

# # 用法示例
# directory_path = "dino-game/GENETIC/brain/"

# create_directory(directory_path)

#todo ==================[getBrainLink]==================

# def getBrainLink(genetic, neural_structure): 
#     link = 'neurWeightMLP.csv'
#     if(genetic):
#         link = 'dino-game/GENETIC/dinoBrain'
#         for num in neural_structure:
#             link += '_' + str(num)
#         link += '/'
    
#     return link

# g = True
# structure = [6, 6, 3]
# print(getBrainLink(g, structure))

#todo ==================[list copy]==================
# x = np.random.choice([True, False], size=10)
# print(x)
#todo ==================[list copy]==================

# a = [1, 2, 3]
# b = a.copy()
# b[0] = 0
# print(a)

#todo ==================[random choice]==================
# idxs = np.arange(10)
# a, b = np.random.choice(idxs, size=2, replace=False)
# # a, b = np.random.randint(0, 10, size=2)
# print(a, b)
#todo ==================[bla]==================

# x = (12, 14)
# y = (10, 10)
# z = (5, 50)
# li = [x, y, z]
# for a, b in li:
#     print(a)
#     print(b)
#     print("hola")


#todo ==================[enconteres in exp]==================
# alpha= 5
# Vi = np.arange(8)
# Y = 2/(1 + np.exp(-alpha * Vi)) - 1
# print(Vi.shape)
# print(Y)
#todo ==================[Probar rango de mutacion]==================
# x = np.linspace(0, 10_000, 1000)#!
# # # y = 2/np.log(x) - 0.000015*x
# # # # y = x**float(-0.01) - 0.000015*x
# # # y = 1/x**float(0.1) 
# # base = 5                        
# # y = 1/np.emath.logn(base, x)    
# alpha = 0.001#!
# y = 0.1 + np.exp(-alpha*x)#!

# y2 = -y
# plt.plot(x, y, label='Cota maxima')
# plt.plot(x, y2, label='Cota minima')
# plt.legend()
# plt.grid(True)
# plt.xlabel('Maximo puntaje')
# plt.ylabel('Cota de mutacion')
# plt.title(r'$\pm (0.1 + e^{-\alpha x }), \alpha = 0.001$')
# plt.show()

#todo ==================[Guardar pesos sinapticos]==================

# Wji = [np.array([
#         [ 0.20086339, -0.17032308, -0.09709814, -0.34608417,  0.01878807, 0.26692764],
#         [ 0.34951064, -0.14537308, -0.51034145,  0.12130056,  0.04974865, 0.35824235],
#         [-0.1001934 ,  0.38867493, -0.2652722 , -0.32913713, -0.15157116, -0.00981034],
#         [ 0.43922338,  0.10524793,  0.46903157, -0.46113214, -0.08514374, 0.09651354],
#         [ 0.46278005,  0.03152608, -0.11888204, -0.18585248, -0.33613728, -0.16309973]
#         ]), 
#     np.array([
#         [-0.26772695,  0.19257835,  0.22187682, -0.41641781, -0.35356841, -0.08471307],
#         [-0.25723259,  0.21676135, -0.20902567,  0.10298885,  0.31517318, -0.19743511],
#         [ 0.06580563,  0.30122432, -0.04365628, -0.22473899, -0.37062099, 0.22058742]
#         ])]


# with open("neurWeightMLP.csv", 'w') as file:
#     for weight in Wji:
#         np.savetxt(file, weight, delimiter=',')


#! Tiene problema
# data = np.genfromtxt('dataSetAux.csv', delimiter=',')
# print(data)

