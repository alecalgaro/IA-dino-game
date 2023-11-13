import pygame
import os
import random
import time
import numpy as np
from collections import deque
from Neuron.neuron import Neuron
from EVOLUTIONARY.Operators.selectionOperator import *
from EVOLUTIONARY.Operators.reproductionOperator import *
from EVOLUTIONARY.algEvolutionary import *

pygame.init()

# Definicion de constantes globales
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Cargar imagenes para el juego
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
                 EVOLUTIONARY=False, structure=[], bSigm=1, idxBrain=0, link=''):
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

        # Se arma el cerebro del dino si no juega el jugador
        if(not(iPlay)):
            if(EVOLUTIONARY):    # si se usa algoritmo EVOLUTIONARYo
                if(initDinoBrain):
                    #* Inicializar los pesos en el rango [-0.5, 0.5]
                    self.brain.initNeuralWeight(structure)
                else:
                    link += 'brain_' + str(idxBrain) + '.csv'
                    self.brain.loadNeuralWeight(link, structure)

            else:       # si se usa solo la red neuronal
                #* Cargar pesos 
                link = 'neurWeightMLP_1.3error.csv'     # EL MEJOR
                # link = 'neurWeightMLP_2.3error.csv'
                # link = 'neurWeightMLP.csv'
                self.brain.loadNeuralWeight(link, structure)

    #? =================[Actualizar en base a decision de neurona o del jugador]=================

    def updateDecision(self, neuralInput):
        # decisionPrev = np.argmax(self.decision)

        self.decision = self.brain.forwardPropagation(neuralInput, self.alpha)
        
    def updateNeuralInput(self):
        self.updateUserInput(self.decision)

    #* Acciones del dino (saltar, agacharse o correr)
    def updateUserInput(self, userInput):

        #* si presiona para saltar y no esta saltando
        if userInput[0] and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
            # print("Saltar")

        #* Bajar rapido
        elif userInput[1] and self.dino_jump:
            self.jump_vel -= 2
            # print("Bajar rapido")

        #* si presiona para agacharse y no esta saltando
        elif userInput[1] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
            # print("Agacharse")
            
        #* si no esta saltando ni presionando para agacharse o saltar
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

    # Obtener eje "y" del dino
    def getDinoData(self):
        return (self.dino_rect.y)
    
    # Devolver el cerebro del dino (pesos sinapticos)
    def getDinoBrain(self):
        return self.brain.getNeuralWeight()

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

#todo ========================================[Class Obstaculo]========================================
class Obstacle:     # Clase base para obstaculos en el juego
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
        self.rect.y = np.random.choice([240, 270, 325], p=[0.55, 0.35, 0.1])
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0

        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1

#todo ========================================[MAIN]========================================

class Game:

    # Variables constantes
    CLOCK = pygame.time.Clock()
    FONT = pygame.font.Font('dino-game/assets/PressStart2P-Regular.ttf', 20)
    FONT2 = pygame.font.Font('dino-game/assets/PressStart2P-Regular.ttf', 15)
    FONT3 = pygame.font.Font('dino-game/assets/PressStart2P-Regular.ttf', 35)
    X_POS_BG = 0
    Y_POS_BG = 380
    VEL_CHECK = 60    # velocidad que se usa luego para registrar datos del juego
    MAX_SPEED = 80

    #* ==================[Constructor, inicializacion]==================

    def __init__(self, nDino = 1, randStart=False, iPlay=True, 
                 initDinoBrain=False, EVOLUTIONARY=False, structure=[], 
                 bSigm=1, link='', EVOLUTIONARYRecord=[]):
        self.run = True
        self.iPlay = iPlay  #? Si es True juega el juador, sino ignora sus inputs de teclado

        # Parametros para dino
        self.player = []
        for idx in range(nDino):
            self.player.append(Dinosaur(randStart, iPlay, initDinoBrain, EVOLUTIONARY,
                                        structure=structure, bSigm=bSigm, idxBrain=idx, link=link))

        self.numLive = nDino
        self.idxLive = np.arange(nDino)
        self.idxBoolLive = np.full(shape=(nDino), fill_value=True, dtype=bool) #? Control del muerto

        # Obstaculos
        self.obstacles = deque([])    

        # self.cloud = Cloud()
        self.game_speed = 14     # velocidad inicial del juego
        self.points = 0          # puntos iniciales en el juego
        self.registPoints = []

        # Datos para mostrar en la pantalla
        self.structure = structure
        self.EVOLUTIONARY = EVOLUTIONARY
        self.EVOLUTIONARYRecord = EVOLUTIONARYRecord

        # cada cierta cantidad de frames se registran los datos del juegos y se guardan o se envian 
        # a la red, para eso usamos estas variables counter y check_interval
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

    # Mostrar informacion en la pantalla
    def drawEVOLUTIONARYRecord(self) -> None:
        txt0 = "Structure: " + str(self.structure)
        txt1 = "Generation: " + str(int(self.EVOLUTIONARYRecord[0]))
        txt2 = "Max Score: " + str(int(self.EVOLUTIONARYRecord[1]))
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
            self.time_next_obstacle = random.uniform(0.6, 2)  # tiempo entre generacion de obstaculos

            # se le da mas probabilidad a Bird para que aparezcan mas aves
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
        if (obstacleData[0] + obstacleData[2] < X_POS) and len(self.obstacles) > 1:
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

        # Actualizacion si juega el jugador
        if(self.iPlay):
            userInput = pygame.key.get_pressed()
            userInput = [userInput[pygame.K_UP], userInput[pygame.K_DOWN], userInput[pygame.K_RIGHT]]

            for idx in self.idxLive[self.idxBoolLive]:
                self.player[idx].updateUserInput(userInput)
                self.player[idx].update()
        
        # Actualizacion si juega con la neurona
        else:
            # Si es el frame de tomar una decision, toma todas las entradas de todo los dinos vivos
            if self.counter == 0:
                # Es una lista de tuplas (idx, input)
                neuralInputs = self.getNeuronalInput()
                
                for idx, input in neuralInputs:
                    self.player[idx].updateDecision(input)
                    self.player[idx].updateNeuralInput()
                    self.player[idx].update()

            # Actualiza la grafica
            else:
                for idx in self.idxLive[self.idxBoolLive]:
                    self.player[idx].updateNeuralInput()
                    self.player[idx].update()
                
                # for dino in self.player:
                #     dino.update(userInput)

        return userInput
    
    #? ========================[Neuronal Input]========================
    # Obtener datos del juego que nos interesan registrar o enviar a la red
    def getNeuronalInput(self):
        inputs = []

        # Dino x, y 
        X_POS = 80

        if(len(self.obstacles) > 0):
            # Con getObstacleData obtenemos (x, y, ancho, alto) del obstaculo
            obstacleData = self.obstacles[0].getObstacleData()
        else:
            return inputs

        # Si el primer obstaculo ya paso donde esta el dino, chequeo con el siguiente (cuando existe)
        if (obstacleData[0] + obstacleData[2] < X_POS) and len(self.obstacles) > 1:
            obstacleData = self.obstacles[1].getObstacleData()

        # Distancia del dino al obstaculo
        dist = obstacleData[0] - X_POS
        
        # for player in self.player:
        for idx in self.idxLive[self.idxBoolLive]:
            player = self.player[idx]
            
            input = [dist,                  # distancia
                     self.game_speed,       # velocidad_juego
                     player.getDinoData(),  # Y_DINO
                     obstacleData[1],       # Y_Obstaculo
                     obstacleData[2],       # ancho_obstaculo
                     obstacleData[3]        # alto_obstaculo
                     ]
            
            # Guardar las tuplas de inputs del dino vivo
            if(dist > 0):    # para que no guarde distancias negativas que son en los frames cuando el obstaculo pasa al dino 
                inputs.append((idx, input))

        return inputs

    #! ======================[DIBUJAR TODO]======================
    # Actualizar pantalla con los textos y graficos
    def updateScreen(self):
        # Aqui deberia estar el screen.fill (lo pasamos al main)
        # SCREEN.fill((255, 255, 255))

        self.drawScore()
        self.drawBackground()
        
        txt = "MLP"
        color = (255, 0, 0)

        if(self.EVOLUTIONARY and not(self.iPlay)):
            self.drawEVOLUTIONARYRecord()
            txt = "EVOLUTIONARY"
            color = (0, 255, 0)

        if(self.iPlay):
            txt = "PLAYER"
            color = (0, 0, 255)

        text = self.FONT3.render(txt, True, color)
        textRect = text.get_rect()
        textRect.center = (300, 50)
        SCREEN.blit(text, textRect)

        for obstacle in self.obstacles:
            obstacle.draw(SCREEN)

        # Solo dibuja los dinos que estan vivo
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

            # userInput y outScreen se usan en el bloque para guardar datos en csv
            userInput = self.updatePlayer()     
            outScreen = self.updateObstacle()
            self.updateSpeed()

            self.collision()

            #? ---- Bloque para guardar los datos durante el juego en csv ----
            #* Primero guarda en el dataSetAux y con el otro "if" guarda solo si fue exitoso el movimiento 
            # if(userInput[0] or userInput[1] or userInput[2]):
            #     input = self.getNeuronalInput()
            #     if(len(input) > 0):
            #         input_data = [int(value) for value in input[0][1]]  # Convierte los valores a enteros
            #         user_input = [int(value) for value in userInput]  # Convierte los valores a enteros
            #         dataSet = np.concatenate([input_data, user_input])[np.newaxis]

            #         # Guardar datos en un archivo auxiliar
            #         with open("dataSetAux.csv", 'a') as auxFile:
            #             np.savetxt(auxFile, dataSet, delimiter=',', fmt='%d')  # Utiliza fmt='%d' para guardar enteros

            # #* Actualizar el data set para entrenar el MLP
            # # Si el obstaculo sale de la pantalla significa que fue exitoso, entonces ahi lo guarda
            # # en el dataSet, en cambio cuando pierda no se va a guardar ese movimiento
            # if(outScreen):
            #     with open("dataSetAux.csv", 'r') as source_file, open("dataSet.csv", 'a') as target_file:
            #         content = source_file.read()
            #         target_file.write(content)
            #     with open("dataSetAux.csv", 'w') as file:
            #         file.truncate()
            #? ----------------
            
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
                
# Funcion que muestra el menu inicial y maneja reinicios
# Aca estan todos los parametros para saber si quiere jugar el jugador, la red neurona o el 
# algoritmo evolutivo, la estructura de la red y parametros del algoritmo evolutivo
def menu():
    #! ========================[Parametros principales]========================
    IPLAY = False               #? True = Juega el jugador, False = buscar/generar cerebro

    # Configuracion de dino
    N_DINO = 80                 #? Numero de dinos
    RAND_START = False          #? Empezar en una posicion aleatoria
    
    # Estructura de la red neuronal
    bSigm = 1
    NEURAL_STRUCTURE = [6, 6, 3]

    #!=================
    EVOLUTIONARY = False              #? (True -> EVOLUTIONARY, False -> MLP)
    #!=================

    #* ===============[Parametros de EVOLUTIONARY]===============
    # Parametros del algoritmo evolutivo
    INIT_DINO_BRAIN = True      #? Inicializacion al azar de los pesos, SINO LEE DE UNA CARPETA
    UPDATE_POPULATION = True    #? Actualizar o no la poblacion por medio de mutacion y cruza

    #* ==========[Cuando UPDATE_POPULATION = True]==========
    # Parametros de SELECCION 
    SELECT_OPER = 0             #? Operador de seleccion (0 = ventana, 1 = competencia, 2 = ruleta)
    NUM_PARENT = 0.4            #? Cantidad de padres deseados. Admite flotante de rango [0, 1]
    REPLACE = False             #? Admitir o no repeticion de individuos

    # Parametros de REPRODUCCION
    PROB_CROSS = 0.9            #? probabilidad de cruza
    PROB_MUTA = 0.1             #? probabilidad de mutacion por cromosoma
    #* ====================================================

    #! ========================================================================
    run = True
    
    # Parametros para jugar solo con la red neuronal
    if(not(IPLAY) and not(EVOLUTIONARY)):
        N_DINO = 1
        RAND_START = False
        NEURAL_STRUCTURE = [6, 6, 3]

    # Generar el string para ubicar/generar la carpeta donde estan los cerebros
    link = getBrainLink(EVOLUTIONARY=EVOLUTIONARY, neural_structure=NEURAL_STRUCTURE)
    elite = []

    # Graficar el menu del comienzo
    start_menu()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:

                # Obtener cantidad de generacions y puntuacion maxima (0 -> generation, 1 -> maxScore)
                reg = getPopulationRegister(EVOLUTIONARY, INIT_DINO_BRAIN, link)
                generation = reg[0]
                maxScore = reg[1]

                # Se crea una instancia del juego con todos los parametros necesarios
                game = Game(nDino=N_DINO, randStart=RAND_START, iPlay=IPLAY,
                            initDinoBrain=INIT_DINO_BRAIN, EVOLUTIONARY=EVOLUTIONARY, 
                            structure=NEURAL_STRUCTURE, bSigm=bSigm, link=link, 
                            EVOLUTIONARYRecord=reg)
                
                # Se ejecuta el juego y se obtiene la informacion de la poblacion resultante
                dataPopulation = game.main()
                # Se obtienen los puntos del ultimo elemento (ultimo dino)
                points = dataPopulation[-1][0]

                # Logica de evolucion de la poblacion
                if(not(IPLAY) and EVOLUTIONARY and UPDATE_POPULATION):
                    if(len(elite) == 0):
                        elite = dataPopulation[0][1].copy()

                    if(maxScore <= points):
                        # Elitismo
                        elite = dataPopulation[-1][1].copy()
                        maxScore = points
                    
                    generation += 1

                    reg = [generation, maxScore]

                    # Se actualiza la poblacion
                    updatePopulation(SELECT_OPER, dataPopulation, NUM_PARENT, REPLACE, NEURAL_STRUCTURE,
                                     N_DINO, PROB_CROSS, PROB_MUTA, maxScore, elite, link, reg)
           
                    INIT_DINO_BRAIN = False

                restart_menu(points)

#* Extraer la ubicacion del archivo de los pesos
def getBrainLink(EVOLUTIONARY, neural_structure): 
    link = 'neurWeightMLP.csv'
    if(EVOLUTIONARY):
        link = 'dino-game/EVOLUTIONARY/dinoBrain'
        for num in neural_structure:
            link += '_' + str(num)
        link += '/'

        os.makedirs(link, exist_ok=True)
    
    return link

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