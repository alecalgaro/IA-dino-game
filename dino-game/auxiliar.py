import numpy as np
import pandas as pd
# #! ===============[Origin]

# def main():
#     global game_speed, x_pos_bg, y_pos_bg, points, obstacles, check_interval
#     run = True
#     clock = pygame.time.Clock()
#     player = Dinosaur()
#     obstacles = deque([])      #? variable que nos interesa para la red neuronal

#     # cloud = Cloud()
#     game_speed = 14     #? variable que nos interesa para la red neuronal
#     x_pos_bg = 0
#     y_pos_bg = 380
#     points = 0      #? variable que nos interesa para el algoritmo evolutivo
#     font = pygame.font.Font('dino-game/assets/PressStart2P-Regular.ttf', 20)

#     # Cada n frame hace un check, para no sobrecargar la compu ir chequeando
#     # frame por frame
#     counter = 0
#     vel_check = 220
#     check_interval = vel_check//game_speed

#     time_prev = time.time()
#     time_next_obstacle = 10

#     def score():
#         global points, game_speed, check_interval
#         points += 1
#         if points % 100 == 0:   # aumenta la velocidad del juego cada 100 puntos
#             # Tener una velocidad maxima
#             game_speed = min(60, game_speed + 0.4)
#             check_interval = vel_check//game_speed

#         text = font.render("Points: " + str(points), True, (0, 0, 0))
#         textRect = text.get_rect()
#         textRect.center = (920, 40)
#         SCREEN.blit(text, textRect)

#     def background():
#         global x_pos_bg, y_pos_bg
#         image_width = BG.get_width()
#         SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
#         SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
#         if x_pos_bg <= -image_width:
#             SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
#             x_pos_bg = 0
#         x_pos_bg -= game_speed

#     #* ===============[Ciclo principal del juego]===============
#     while run:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False

#         SCREEN.fill((255, 255, 255))

#         #? ===========[Obstaculos]===========

#         #* Generacion de los obstaculos: 
#         if len(obstacles) == 0 or (time.time() - time_prev) > time_next_obstacle:
#             time_prev = time.time()
#             time_next_obstacle = random.uniform(0.6, 3)    # tiempo entre generacion de obstaculos
#             idx_obs = random.randint(0, 2)
#             match idx_obs:
#                 case 0:
#                     obstacles.append(SmallCactus(SMALL_CACTUS))
#                 case 1:
#                     obstacles.append(LargeCactus(LARGE_CACTUS))
#                 case 2:
#                     obstacles.append(Bird(BIRD))

        
#         #* Recorrer y analizar cada uno de los obstaculos
#         for obstacle in obstacles:

#             # Ajustar el área de colisión para el dinosaurio
#             # Inflate ajusta un rectangulo alrededor de su centro, que sera la caja de colision
#             dino_collision_rect = player.dino_rect.inflate(-50, -5)
#             #! check inflate
#             # pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(dino_collision_rect))

#             # Ajustar el área de colisión para las aves
#             if isinstance(obstacle, Bird):
#                 bird_collision_rect = obstacle.rect.inflate(-60, -5)
#             else:
#                 bird_collision_rect = obstacle.rect.inflate(-10, 0)

#             #! check inflate
#             # pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(bird_collision_rect))

#             # Verificar colisión entre el dinosaurio y el obstáculo
#             if dino_collision_rect.colliderect(bird_collision_rect):
#                 pygame.time.delay(200)
#                 return points

#             obstacle.draw(SCREEN)
#             obstacle.update()

#         # Chequear y eliminar el ultimo elemento
#         obs_data = obstacles[0].getObstacleData()
#         if obs_data[0] < -obs_data[3]:
#             obstacles.popleft()
#             # print("borrar")


#         #? ===========[DINO, input]===========

#         # lista de UP y DOWN
#         userInput = pygame.key.get_pressed()
#         userInput = [userInput[pygame.K_UP], userInput[pygame.K_DOWN], userInput[pygame.K_SPACE]]
#         # print(userInput)
#         player.draw(SCREEN)
#         player.update(userInput)

#         background()

#         # cloud.draw(SCREEN)
#         # cloud.update()

#         score()

#         #* Contador de frame, cada n frame hace una "captura", en vez de hacer todo el tiempo 
#         #* Se puede ver que a medida se aumenta la game_speed, la captura se hace cada vez mas
#         #* rapido
#         if counter >= check_interval:
#             counter = 0
#             pygame.draw.rect(SCREEN, (255, 0, 0), pygame.Rect(100, 100, 100, 100))
#         else:
#             counter += 1

#         clock.tick(50)
#         pygame.display.update()

# with open('dataSetAux.csv', 'a') as file:
#     for _ in range(10):
#         data = np.arange(np.random.randint(3, 8))[np.newaxis]
#         np.savetxt(file, data, delimiter=',')

import csv

# with open('dataSetAux.csv', 'r') as file:
#     csvreader = csv.reader(file)
#     for row in csvreader:
#         print(type(row))

# with open('dataSetAux.csv', 'a') as file:
#     datas = [[1, 2, 3], [6, 6], [10]] #! Todos son 1D, se complica la cosa
#     for data in datas:
#         np.savetxt(file, [data], delimiter=',') #! De toda manera pasarlo a 2D
        # np.savetxt(file, data, delimiter=',')


# with open('dataSetAux.csv', 'a') as file:
#     # data = [[1, 2, 3], [6, 6], [10]] #! No se puede por asarray interna
#     data = [[1, 2, 3]] #* si se puede, 2D!!!!
#     np.savetxt(file, data, delimiter=',')

# df = pd.read_csv('neurWeightMLP.csv', delimiter=',',dtype=float, na_values=0)

# maxStructure = 10
# df = pd.read_csv('neurWeightMLP.csv', delimiter=',', usecols=range(10), header=None) #!NOP


#? Perfecto
# structure = [10, 5, 5, 3]
# skrow = 0
# for n in structure:
#     df = pd.read_csv('neurWeightMLP.csv', delimiter=',',
#                       header=None, skiprows=skrow, nrows=n) 
#     skrow += n
#     matrix = df.to_numpy()
#     print(type(matrix))
#     print(matrix.shape[0])

a = np.arange(3)
b = [-1]
c = np.concatenate([b, a]) # a y b deben ser al menos 1D, no acepta constante que es 0D
print(c)

#! Tiene problema
# data = np.genfromtxt('dataSetAux.csv', delimiter=',')
# print(data)

