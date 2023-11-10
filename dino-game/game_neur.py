import pygame
import os
import random
import time
import numpy as np
from collections import deque
from Neuron.neuron import Neuron
from GENETIC.Operators.selectionOperator import *
from GENETIC.Operators.reproductionOperator import *

pygame.init()

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
                pygame.image.load(os.path.join(root, "assets/Cactus", "SmallCactus3.png"))
                ]

LARGE_CACTUS = [
                pygame.image.load(os.path.join(root, "assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join(root, "assets/Cactus", "LargeCactus1.png"))
                # pygame.image.load(os.path.join(root, "assets/Cactus", "LargeCactus2.png")),
                # pygame.image.load(os.path.join(root, "assets/Cactus", "LargeCactus3.png"))
                ]

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
                 genetic=False, structure=[], bSigm=1, idxBrain=0, link=''):
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
        if(not(iPlay)):
            if(genetic):
                if(initDinoBrain):
                    #* Inicializar los pesos en el rango [-0.5, 0.5]
                    self.brain.initNeuralWeight(structure)
                else:
                    link += 'brain_' + str(idxBrain) + '.csv'
                    self.brain.loadNeuralWeight(link, structure)

            else:
                #* Cargar pesos 
                # link = 'neurWeightMLP_2.3error.csv'
                link = 'neurWeightMLP_1.3error.csv'     # EL MEJOR
                # link = 'neurWeightMLP.csv'
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

        #! Bajar rapido
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
        self.rect.y = np.random.choice([200, 270, 325], p=[0.55, 0.35, 0.1])
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
    FONT2 = pygame.font.Font('dino-game/assets/PressStart2P-Regular.ttf', 15)
    X_POS_BG = 0
    Y_POS_BG = 380
    VEL_CHECK = 20
    MAX_SPEED = 80

    #* ==================[Constructor, inicializacion]==================

    def __init__(self, nDino = 1, randStart=False, iPlay=True, 
                 initDinoBrain=False, genetic=False, structure=[], 
                 bSigm=1, link='', geneticRecord=[]):
        self.run = True
        self.iPlay = iPlay #? Juega el juador, sino ignora sus inputs

        # Parametros para dino
        self.player = []
        for idx in range(nDino):
            self.player.append(Dinosaur(randStart, iPlay, initDinoBrain, genetic,
                                        structure=structure, bSigm=bSigm, idxBrain=idx, link=link))

        self.numLive = nDino
        self.idxLive = np.arange(nDino)
        self.idxBoolLive = np.full(shape=(nDino), fill_value=True, dtype=bool) #? Control del muerto

        # Obstaculos
        self.obstacles = deque([])      #? variable que nos interesa para la red neuronal

        # self.cloud = Cloud()
        self.game_speed = 14     #? variable que nos interesa para la red neuronal
        self.points = 0     #? variable que nos interesa para el algoritmo evolutivo
        self.registPoints = []

        # Datos para mostrar en la pantalla
        self.structure = structure
        self.genetic = genetic
        self.geneticRecord = geneticRecord

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
            self.game_speed = min(self.MAX_SPEED, self.game_speed + 0.2)
            self.check_interval = self.VEL_CHECK//self.game_speed

    def drawGeneticRecord(self) -> None:


        txt0 = "Structure: " + str(self.structure)
        txt1 = "Generation: " + str(int(self.geneticRecord[0]))
        txt2 = "Max Score: " + str(int(self.geneticRecord[1]))
        txt3 = "Alive:" + str(self.numLive)

        x = 720
        y = 75
        linespacing = 22

        # Structure
        text = self.FONT2.render(txt0, True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.topleft = (x, y)
        SCREEN.blit(text, textRect)
        y += linespacing

        # Generation
        text = self.FONT2.render(txt1, True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.topleft = (x, y)
        SCREEN.blit(text, textRect)
        y += linespacing

        # Max Score
        text = self.FONT2.render(txt2, True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.topleft = (x, y)
        SCREEN.blit(text, textRect)
        y += linespacing

        # Alive
        text = self.FONT2.render(txt3, True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.topleft = (x, y)
        SCREEN.blit(text, textRect)



    def drawScore(self):
        text = self.FONT.render("Points: " + str(self.points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.topleft = (750, 40)
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
            self.time_next_obstacle = random.uniform(0.6, 2)    # tiempo entre generacion de obstaculos

            idx_obs = np.random.choice([0, 1, 2], p=[0.2, 0.2, 0.6])
            match idx_obs:
                case 0:
                    self.obstacles.append(SmallCactus(SMALL_CACTUS))
                case 1:
                    self.obstacles.append(LargeCactus(LARGE_CACTUS))
                case 2:
                    self.obstacles.append(Bird(BIRD))

        #* Eliminar el ultimo elemento si ya salio de la pantalla
        obs_data = self.obstacles[0].getObstacleData()
        # out = obs_data[0] < -obs_data[2]
        out = obs_data[0] < (-obs_data[2] + 20)
        # out = obs_data[0] == -10
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
            obstacle_collision_rect = obstacle.rect.inflate(-30, -5)

        #? Hitbox
        # pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(obstacle_collision_rect))

        # Recorrer cada dino y verificar si esta en colision
        for idx in self.idxLive[self.idxBoolLive]:
            player = self.player[idx]
            dino_collision_rect = player.dino_rect.inflate(-50, -5)

            #? Hitbox
            # pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(dino_collision_rect))

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
        X_POS = 80

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
        # dist_norm = dist/self.game_speed 
        
        # for player in self.player:
        for idx in self.idxLive[self.idxBoolLive]:
            player = self.player[idx]
            # [
            # dist_norm
            # velocidad del juego
            # Y_DINO
            # Y_obstaculo
            # ancho_obstaculo
            # alto_obstaculo
            # ]
            
            input = [dist,             # distancia/velocidad
                     self.game_speed,        # velocidad_juego
                     player.getDinoData(),  # Y_DINO
                     obstacleData[1],       # Y_Obstaculo
                     obstacleData[2],       # ancho_obstaculo
                     obstacleData[3]       # alto_obstaculo
                     ]
            
            # Guardar las tuplas de inputs de dino vivo
            if(dist > 0):    # para que no guarde distancias negativas que son en los frames cuando el ostaculo pasa al dino 
                inputs.append((idx, input))
                # print("dist:", f'{input[0]:.2f}')
                # print(f'{valor:.2f}')

        return inputs

    #! ======================[DIBUJAR TODO]======================
    def updateScreen(self):
        #! Aqui deberia estar el screen.fill
        # SCREEN.fill((255, 255, 255))

        self.drawScore()
        self.drawBackground()
        
        if(self.genetic):
            self.drawGeneticRecord()

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

            #! ---- PARA GUARDAR LOS DATOS DURANTE EL JUEGO PARA USAR EN EL ENTRENAMIENTO ----
            #! Generar data set aux para agregarlo en el data set
            #Primero guarda en el dataSetAux y con el otro "if" guarda solo si fue exitoso 
            # if(userInput[0] or userInput[1] or userInput[2]):
            #     input = self.getNeuronalInput()
            #     if(len(input) > 0):
            #         input_data = [int(value) for value in input[0][1]]  # Convierte los valores a enteros
            #         user_input = [int(value) for value in userInput]  # Convierte los valores a enteros
            #         dataSet = np.concatenate([input_data, user_input])[np.newaxis]

            #         #* Guardar datos en un archivo auxiliar
            #         with open("dataSetAux.csv", 'a') as auxFile:
            #             np.savetxt(auxFile, dataSet, delimiter=',', fmt='%d')  # Utiliza fmt='%d' para guardar enteros

            # #! Actualizar el data set para entrenar MLP
            # # Si el obstáculo sale de la pantalla significa que fue exitoso, entonces ahí lo guarda
            # # en el dataSet. Por ejemplo, cuando pierda no lo va a guardar.
            # if(outScreen):
            #     with open("dataSetAux.csv", 'r') as source_file, open("dataSet.csv", 'a') as target_file:
            #         content = source_file.read()
            #         target_file.write(content)
            #     with open("dataSetAux.csv", 'w') as file:
            #         file.truncate()
            #! --------
            
            #* Check dino vivo, sino sale del juego
            if self.numLive == 0:
                time.sleep(1)
                break

            if self.counter >= self.check_interval:
                self.counter = 0
                # pygame.draw.rect(SCREEN, (255, 0, 0), pygame.Rect(100, 100, 100, 100))
            else:
                self.counter += 1

            #! Dibujar todo
            self.updateScreen()

        return self.registPoints
                

# Función que muestra el menú inicial y maneja reinicios
def menu():
    #! ===============[Parametros principales]===============
    IPLAY = False               #? True = Juega el jugador, False = buscar/generar celebro

    # Configuracion de dino
    N_DINO = 80                 #? Numero de dinos
    RAND_START = False           #? Empezar en una posicion aleatoria
    
    # Estructura de la red neuronal
    bSigm = 1
    NEURAL_STRUCTURE = [6, 8, 3]


    # Parametros de algoritmo genetico
    GENETIC = True              #? False = MLP
    INIT_DINO_BRAIN = False      #? Inicializacion al azar de los pesos, SINO LEE DE UNA CARPETA
    UPDATE_POPULATION = True   #? Actualizar o no la poblacion por medio de mutacion y cruza


    #* ===============[Cuando UPDATE_POPULATION = True]===============
    # Parametros de SELECCION 
    SELECT_OPER = 0             #? Operador de seleccion (0 = ventana, 1 = competencia, 2 = ruleta)
    NUM_PARENT = 0.5            #? Cantidad de padres deseados. Admite flotante de rango [0, 1]
    REPLACE = False             #? Admitir o no repeticion de individuos

    # Parametros de REPRODUCCION
    PROB_CROSS = 0.8
    PROB_MUTA = 0.15
    #* ===============================================================


    #! ======================================================
    run = True
    if(not(GENETIC)):
        NEURAL_STRUCTURE = [6, 6, 3]

    # Generar el string para ubicar/generar la carpeta donde estan los celebros
    link = getBrainLink(genetic=GENETIC, neural_structure=NEURAL_STRUCTURE)
    elite = []

    # Graficar el menu del comienzo
    start_menu()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:

                # (0 -> generation, 1 -> maxScore)
                reg = getPopulationRegister(GENETIC, INIT_DINO_BRAIN, link)
                # Sacar los mejores puntos
                generation = reg[0]
                maxScore = reg[1]

                #! game = Game()
                game = Game(nDino=N_DINO, randStart=RAND_START, iPlay=IPLAY,
                            initDinoBrain=INIT_DINO_BRAIN, genetic=GENETIC, 
                            structure=NEURAL_STRUCTURE, bSigm=bSigm, link=link, 
                            geneticRecord=reg)
                dataPopulation = game.main()
                points = dataPopulation[-1][0]

                if(GENETIC and UPDATE_POPULATION):
                    if(len(elite) == 0):
                        elite = dataPopulation[0][1].copy()

                    if(maxScore <= points):
                        # Elitismo
                        elite = dataPopulation[-1][1].copy()
                        maxScore = points
                    
                    generation += 1

                    reg = [generation, maxScore]

                    updatePopulation(SELECT_OPER, dataPopulation, NUM_PARENT, REPLACE, NEURAL_STRUCTURE,
                                     N_DINO, PROB_CROSS, PROB_MUTA, maxScore, elite, link, reg)
           
                    INIT_DINO_BRAIN = False

                restart_menu(points)

def updatePopulation(SELECT_OPER, dataPopulation, NUM_PARENT, REPLACE,
                     NEURAL_STRUCTURE, N_DINO, PROB_CROSS, PROB_MUTA, 
                     points, elite, link, register):
    parent = []
    match(SELECT_OPER):
        case 0:
            parent = window(dataPopulation, numParent=NUM_PARENT, replace=REPLACE)
        case 1:
            parent = competition(dataPopulation, numParent=NUM_PARENT, replace=REPLACE)
        case _:
            parent = roulette(dataPopulation, numParent=NUM_PARENT, replace=REPLACE)
    
    parent[0] = elite
    # Cruza
    child = crossover(structure=NEURAL_STRUCTURE, parent=parent, 
                        nDino=N_DINO, probability=PROB_CROSS)
    # Mutacion
    child = mutation(structure=NEURAL_STRUCTURE, childs=child, 
                        probability=PROB_MUTA, score=points)

    # Unir los padres hijos
    population = parent + child

    # Almacenar poblacion en las carpetas que le correspondan
    savePopulation(population, link)
    saveRegister(register, link)




#* Almacenar Poblacion
def savePopulation(brains, link):
    for idx, Wji in enumerate(brains):
        path = link + 'brain_' + str(idx) + '.csv'

        with open(path, 'w') as file:
            for weight in Wji:
                np.savetxt(file, weight, delimiter=',')

#* Almacenar el registro de poblacion
def saveRegister(register, link) -> None:
    np.savetxt(link + 'register.csv', register, delimiter=',')


#* Extraer la ubicacion del archivo de los pesos
def getBrainLink(genetic, neural_structure): 
    link = 'neurWeightMLP.csv'
    if(genetic):
        link = 'dino-game/GENETIC/dinoBrain'
        for num in neural_structure:
            link += '_' + str(num)
        link += '/'

        os.makedirs(link, exist_ok=True)
    
    return link

# Extraer registro de la poblacion
def getPopulationRegister(GENETIC, INIT_DINOBRAIN, link):
    """
    When it IS genetic, it returns
    [0] -> generation 
    [1] -> maxScore
    """
    register = [int(0), int(0)]
    if(GENETIC and not(INIT_DINOBRAIN)):
        path = link + "register.csv"
        
        # Si existe el registro, extrae los datos. 
        if(os.path.exists(path)):
            register = np.genfromtxt(path, delimiter=',')

        # Sino crea el registro.  
        else:
            np.savetxt(path, register, delimiter=",")
    
    return register



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