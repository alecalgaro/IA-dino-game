import pygame
import os
import random
import time
from collections import deque
import numpy as np
from Neuron.neuron import Neuron

pygame.init()

#! Detalle para pensar: el dinosaurio en realidad esta quieto, lo que se mueve es el fondo
#! con los obstaculos, asi que la distancia del dino al obstaculo se podria calcular
#! directo como la "x" del obstaculo tal vez.

# Definición de constantes globales
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Cargar imágenes para el juego
root = "dino-game"
RUNNING = [pygame.image.load(os.path.join(root, "assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join(root, "assets/Dino", "DinoRun2.png"))]

JUMPING = pygame.image.load(os.path.join(root, "assets/Dino", "DinoJump.png"))

DUCKING = [pygame.image.load(os.path.join(root, "assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join(root, "assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join(root, "assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join(root, "assets/Cactus", "SmallCactus2.png")),
                # pygame.image.load(os.path.join(root, "assets/Cactus", "SmallCactus3.png"))
                ]

LARGE_CACTUS = [pygame.image.load(os.path.join(root, "assets/Cactus", "LargeCactus1.png")),
                # pygame.image.load(os.path.join(root, "assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join(root, "assets/Cactus", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join(root, "assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join(root, "assets/Bird", "Bird2.png"))]

# CLOUD = pygame.image.load(os.path.join(root, "assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join(root, "assets/Other", "Track.png"))

#todo ======================================[Class Dino]======================================
class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8
    GRAVITY = 0.6

    def __init__(self, randStart = False, iPlay = True, initDinoBrain = False,
                 genetic=False, structure=[], bSigm=1, idxBrain=0):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()

        self.randN = randStart*np.random.uniform(-10, 220)
        self.dino_rect.x = self.X_POS + self.randN
        self.dino_rect.y = self.Y_POS

        #? Inicializar el "cerebro" del dino y que empiece caminando
        self.brain = Neuron()
        self.decision = [False, False, True]
        self.alpha = bSigm

        #? Truco, promocionar aquel dino que realiza varias acciones
        self.promosion = 0

        # Armo su celebro si NO juego yo
        #? Estructura de la neurona, HAY QUE ACLARAR AQUI
        if(not(iPlay)):

            # Si usa algoritmo genetico, admite inicializacion aleatoria 
            #! y rehace todo initDinoBrain = True
            # Sino simplemente arma su celebro en base a los datos de poblacion
            if(genetic):
                if(initDinoBrain):
                    #* Inicializar los pesos en el rango [-0.5, 0.5]
                    self.brain.initNeuralWeight(structure)
                else:
                    link = 'dino-game/GENETIC/dinoBrain'+ str(idxBrain) + '.csv'
                    self.brain.loadNeuralWeight(link, structure)

            else:
                #* Cargar pesos 
                link = 'neurWeightMLP.csv'
                self.brain.loadNeuralWeight(link, structure)

    #? =================[Actualizar en base a decision de neurona o del jugador]=================

    def updateDecision(self, neuralInput):
        # decisionPrev = np.argmax(self.decision)

        self.decision = self.brain.forwardPropagation(neuralInput, self.alpha)

        #! Ver si llega a ser necesario
        #? Cuando realiza diversas acciones distintas, lo promosiono
        #? para superar aquel que saca mayor punto por mantener presionado
        #? una sola tecla
        # if(decisionPrev != np.argmax(self.decision)):
        #     self.promosion += 20
        

    def updateNeuralInput(self):
        self.updateUserInput(self.decision)

    def updateUserInput(self, userInput):

        #* Cuando juegue la red neuronal habria que simular que presiona estas teclas
        #! si presiona para saltar y no esta saltando
        if userInput[0] and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
            # print("Saltar")

        #! si presiona para saltar y esta agachado (LO AGREGUE NUEVO)
        # elif userInput[0] and self.dino_duck:
        #     self.dino_duck = False
        #     self.dino_run = False
        #     self.dino_jump = True

        #? Bajar rapido
        elif userInput[1] and self.dino_jump:
            self.jump_vel -= 2
            # print("Bajar rapido")

        #! si presiona para agacharse y no esta saltando
        elif userInput[1] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
            # print("Agacharse")
        
        #! si no esta saltando ni presionando para agacharse o saltar
        elif not (self.dino_jump or userInput[0] or userInput[1]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False
            # print("Correr")

    def update(self):
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.dino_duck:
            self.duck()

        if self.step_index >= 10:
            self.step_index = 0

    #* Necesitamos solo su eje y
    def getDinoData(self):
        return (self.dino_rect.y)
    
    # Devolver su celebro, pesos sinapticos
    def getDinoBrain(self):
        return self.brain.getNeuralWeight()
    
    def getDinoPromosion(self):
        return self.promosion

#? =================================[Estados del Dino]=================================
    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()

        self.dino_rect.x = self.X_POS + self.randN

        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.width
        self.dino_rect.height

        self.dino_rect.x = self.X_POS + self.randN

        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= self.GRAVITY

        if self.dino_rect.y > self.Y_POS:
            self.dino_jump = False
            self.dino_rect.y = self.Y_POS
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

#todo ========================================[Class Cloud]========================================
# Clase para representar nubes en el juego
# class Cloud:
#     def __init__(self):
#         self.x = SCREEN_WIDTH + random.randint(800, 1000)
#         self.y = random.randint(50, 100)
#         self.image = CLOUD
#         self.width = self.image.get_width()

#     def update(self):
#         self.x -= game_speed
#         if self.x < -self.width:
#             self.x = SCREEN_WIDTH + random.randint(2500, 3000)
#             self.y = random.randint(50, 100)

#     def draw(self, SCREEN):
#         SCREEN.blit(self.image, (self.x, self.y))

# Clase base para obstáculos en el juego

#todo ========================================[Class Obstaculo]========================================
class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self, game_speed):
        self.rect.x -= game_speed

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getObstacleData(self):
        # Retorna el eje x, y, ancho y alto del obstaculo
        return (self.rect.x, self.rect.y, self.rect.width, self.rect.height)

#? ===================[Cactus]===================
# Clase para representar cactus chicos
class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 1)
        super().__init__(image, self.type)
        self.rect.y = 325

# Clase para representar cactus grandes
class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 1)
        super().__init__(image, self.type)
        self.rect.y = 300

#? ===================[Bird]===================
# Clase para representar aves
class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = np.random.choice([220, 270, 325], p=[0.2, 0.4, 0.4])
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0

        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1

#todo ========================================[MAIN]========================================

class Game:

    # Valores constantes en mayuscula
    CLOCK = pygame.time.Clock()
    FONT = pygame.font.Font('dino-game/assets/PressStart2P-Regular.ttf', 20)
    X_POS_BG = 0
    Y_POS_BG = 380
    VEL_CHECK = 150
    MAX_SPEED = 80

    #* ==================[Constructor, inicializacion]==================

    def __init__(self, nDino = 1, randStart=False, iPlay=True, 
                 initDinoBrain=False, genetic=False, structure=[], bSigm=1) :
        self.run = True
        self.iPlay = iPlay #? Juega el juador, sino ignora sus inputs

        # Parametros para dino
        self.player = []
        for idx in range(nDino):
            self.player.append(Dinosaur(randStart, iPlay, initDinoBrain, genetic,
                                        structure=structure, bSigm=bSigm, idxBrain=idx))

        self.numLive = nDino
        self.idxLive = np.arange(nDino)
        self.idxBoolLive = np.full(shape=(nDino), fill_value=True, dtype=bool) #? Control del muerto

        # Obstaculos
        self.obstacles = deque([])      #? variable que nos interesa para la red neuronal

        # self.cloud = Cloud()
        self.game_speed = 14     #? variable que nos interesa para la red neuronal
        self.points = 0     #? variable que nos interesa para el algoritmo evolutivo
        self.registPoints = []

        # n frame hace un check
        self.counter = 0
        self.check_interval = self.VEL_CHECK//self.game_speed

        # time
        self.time_prev = 10
        self.time_next_obstacle = 10
    
    #? ========================[Seconday]========================
    def updateSpeed(self):
        if self.points % 100 == 0:   # aumenta la velocidad del juego cada 100 puntos
            # Tener una velocidad maxima
            self.game_speed = min(self.MAX_SPEED, self.game_speed + 0.4)
            self.check_interval = self.VEL_CHECK//self.game_speed

    def drawScore(self):
        text = self.FONT.render("Points: " + str(self.points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (920, 40)
        SCREEN.blit(text, textRect)
    
    def drawBackground(self):
        image_width = BG.get_width()
        SCREEN.blit(BG, (self.X_POS_BG, self.Y_POS_BG))
        SCREEN.blit(BG, (image_width + self.X_POS_BG, self.Y_POS_BG))
        if self.X_POS_BG <= -image_width:
            SCREEN.blit(BG, (image_width + self.X_POS_BG, self.Y_POS_BG))
            self.X_POS_BG = 0
        self.X_POS_BG -= self.game_speed

    #? ========================[Obstacle]========================
    # Actualiza los obstaculos y retorna un booleano para indicar si salio o no de la pantalla
    def updateObstacle(self):

        #* Generacion de los obstaculos: 
        if len(self.obstacles) == 0 or (time.time() - self.time_prev) > self.time_next_obstacle:

            #* Crear nuevo obstaculo
            self.time_prev = time.time()
            self.time_next_obstacle = random.uniform(0.8, 2)    # tiempo entre generacion de obstaculos

            idx_obs = random.randint(0, 2)
            match idx_obs:
                case 0:
                    self.obstacles.append(SmallCactus(SMALL_CACTUS))
                case 1:
                    self.obstacles.append(LargeCactus(LARGE_CACTUS))
                case 2:
                    self.obstacles.append(Bird(BIRD))

        #* Eliminar el ultimo elemento si ya salio de la pantalla
        obs_data = self.obstacles[0].getObstacleData()
        out = obs_data[0] < -obs_data[3]
        if out:
            self.obstacles.popleft()

        #* Actualizar 
        for obstacle in self.obstacles:
            obstacle.update(self.game_speed)

        return out

    #? ========================[Collision]========================
    def collision(self):
        
        X_POS = 70

        if(len(self.obstacles) > 0):
            obstacle = self.obstacles[0]
        else:
            return

        # Si el primer obstaculo ya paso donde esta dino, chequeo con el siguiente (cuando existe)
        obstacleData = obstacle.getObstacleData()
        if (obstacleData[0] + obstacleData[3] < X_POS) and len(self.obstacles) > 1:
            obstacle = self.obstacles[1]

        # Armar el collision rect de obstaculo
        obstacle_collision_rect = obstacle.rect.inflate(-10, 0)
        if isinstance(obstacle, Bird):
            obstacle_collision_rect = obstacle.rect.inflate(-60, -5)

        #? Hitbox
        pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(obstacle_collision_rect))

        # Recorrer cada dino y verificar si esta en colision
        for idx in self.idxLive[self.idxBoolLive]:
            player = self.player[idx]
            dino_collision_rect = player.dino_rect.inflate(-50, -5)

            #? Hitbox
            pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(dino_collision_rect))

            # Haber colision, lo elimino de la lista y registro su indice con su puntaje 
            if dino_collision_rect.colliderect(obstacle_collision_rect):
                self.numLive -= 1
                self.idxBoolLive[idx] = False
                self.registPoints.append((self.points, player.getDinoBrain()))

    #? ========================[Player]========================
    def updatePlayer(self):

        userInput = [False, False, False]

        # Juega el jugador, actualiza en base a eso
        if(self.iPlay):
            userInput = pygame.key.get_pressed()
            userInput = [userInput[pygame.K_UP], userInput[pygame.K_DOWN], userInput[pygame.K_RIGHT]]

            for idx in self.idxLive[self.idxBoolLive]:
                self.player[idx].updateUserInput(userInput)
                self.player[idx].update()
        
        # Juega con la neurona
        else:
            # Si es el frame de tomar desicion, toma todas las entradas 
            # de todo los dinos vivos
            if self.counter == 0:
                # Es una lista de tuplas (idx, input)
                neuralInputs = self.getNeuronalInput()
                
                for idx, input in neuralInputs:
                    self.player[idx].updateDecision(input)
                    self.player[idx].updateNeuralInput()
                    self.player[idx].update()

            # Simplemente actualiza la grafica
            else:
                for idx in self.idxLive[self.idxBoolLive]:
                    self.player[idx].updateNeuralInput()
                    self.player[idx].update()
                
                # for dino in self.player:
                #     dino.update(userInput)

        return userInput
    
    #? ========================[Neuronal Input]========================
    def getNeuronalInput(self):
        inputs = []

        # Dino x, y 
        X_POS = 70

        if(len(self.obstacles) > 0):
            # Con getObstacleData sacamos (x, y, ancho, alto) del obstaculo
            obstacleData = self.obstacles[0].getObstacleData()
        else:
            return inputs

        # Si el primer obstaculo ya paso donde esta dino, chequeo con el siguiente (cuando existe)
        if (obstacleData[0] + obstacleData[2] < X_POS) and len(self.obstacles) > 1:
            obstacleData = self.obstacles[1].getObstacleData()

        # Normalizamos para tener distancia al obstaculo / velocidad del juego
        dist = obstacleData[0] - X_POS
        dist_norm = dist/self.game_speed   
        
        # for player in self.player:
        for idx in self.idxLive[self.idxBoolLive]:
            player = self.player[idx]
            # [
            # dist_norm
            # Y_DINO
            # Y_obstaculo
            # ancho_obstaculo
            # alto_obstaculo
            # ]
            #! RECORDAR QUE USE SOLOS dist Y NO dist/vel COMO ANTES
            #! PROBAR AGREGAR TAMBIEN LA VELOCIDAD COMO OTRA ENTRADA Y VER SI MEJORA O NO
            input = [dist,             # distancia/velocidad
                     player.getDinoData(),  # Y_DINO
                     obstacleData[1],       # Y_Obstaculo
                     obstacleData[2],       # ancho_obstaculo
                     obstacleData[3]        # alto_obstaculo
                     ]

            # Guardar las tuplas de inputs de dino vivo
            inputs.append((idx, input))

        return inputs

    #! ======================[DIBUJAR TODO]======================
    def updateScreen(self):
        #! Aqui deberia estar el screen.fill
        # SCREEN.fill((255, 255, 255))

        self.drawScore()
        self.drawBackground()

        for obstacle in self.obstacles:
            obstacle.draw(SCREEN)

        # Solo dibuja lo que estan vivo
        for idx in self.idxLive[self.idxBoolLive]: 
            self.player[idx].draw(SCREEN)

        # self.cloud.draw(SCREEN)
        # self.cloud.update()

        self.CLOCK.tick(50)
        pygame.display.update()
    
    def main(self):

        while self.run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False

            #! Dejar aqui screen.fill para hacer check de las cosas
            SCREEN.fill((255, 255, 255))

            self.points += 1

            userInput = self.updatePlayer()
            outScreen = self.updateObstacle()
            self.updateSpeed()

            self.collision()

            # #! Generar data set aux para agregarlo en el data set 
            # if(userInput[0] or userInput[1] or userInput[2]):
            #     input = self.getNeuronalInput()
            #     if(len(input) > 0):
            #         dataSet = np.concatenate([input[0][1], userInput])[np.newaxis]

            #         #* Guardar datos en un archivo auxiliar
            #         with open("dataSetAux.csv", 'a') as auxFile:
            #           np.savetxt(auxFile, dataSet, delimiter=',')
            #         # np.savetxt('dataSetAux.csv', dataSet, delimiter=',')

            # #! Actualizar el data set para entrenar MLP
            # if(outScreen):
            #     with open("dataSetAux.csv", 'r') as source_file, open("dataSet.csv", 'a') as target_file:
            #         content = source_file.read()
            #         target_file.write(content)
            #     with open("dataSetAux.csv", 'w') as file:
            #         file.truncate()

            #* Check dino vivo, sino sale del juego
            if self.numLive == 0:
                time.sleep(1)
                break

            if self.counter >= self.check_interval:
                self.counter = 0
                pygame.draw.rect(SCREEN, (255, 0, 0), pygame.Rect(100, 100, 100, 100))
            else:
                self.counter += 1

            #! Dibujar todo
            self.updateScreen()

        return self.registPoints
                

# Función que muestra el menú inicial y maneja reinicios
def menu():
    #! Parametros principales
    run = True

    IPLAY = False           #* Juega el jugador

    # Configuracion de dino
    N_DINO = 1               # Numero de dinos
    RAND_START = False       # Empezar en una posicion aleatoria
    
    # Parametros de algoritmo genetico
    GENETIC = True
    INIT_DINO_BRAIN = True   #! Inicializacion al azar de los pesos, SINO LEE DE UNA CARPETA
    UPDATE_POPULATION = True #! Actualizar la poblacion por medio de mutacion y cruza
    
    # Estructura de la red neuronal
    bSigm = 5
    NEURAL_STRUCTURE = [5, 3]

    start_menu()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                # game = Game()
                game = Game(nDino=N_DINO, randStart=RAND_START, iPlay=IPLAY,
                            initDinoBrain=INIT_DINO_BRAIN, genetic=GENETIC, structure=NEURAL_STRUCTURE, bSigm=bSigm)
                dataPopulation = game.main()
                points = dataPopulation[-1][0]
            
                # INIT_DINO_BRAIN = False

                restart_menu(points)

def start_menu():
    SCREEN.fill((255, 255, 255))
    font = pygame.font.Font('dino-game/assets/PressStart2P-Regular.ttf', 30)

    # Menu principal, cuando recien empieza el juego
    text = font.render("Press any Key to Start", True, (0, 0, 0))
    textRect = text.get_rect()
    textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    SCREEN.blit(text, textRect)
    SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
    pygame.display.update()

def restart_menu(points):
    # Menu de restart
    SCREEN.fill((255, 255, 255))
    font = pygame.font.Font('dino-game/assets/PressStart2P-Regular.ttf', 30)
    text = font.render("Press any Key to Restart", True, (0, 0, 0))
    score = font.render("Your Score: " + str(points), True, (0, 0, 0))
    scoreRect = score.get_rect()
    scoreRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
    SCREEN.blit(score, scoreRect)
    textRect = text.get_rect()
    textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    SCREEN.blit(text, textRect)
    SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
    pygame.display.update()

menu()