import pygame
import os
import random
import time
from collections import deque

pygame.init()

#! Detalle para pensar: el dinosaurio en realidad esta quito, lo que se mueve es el fondo
#! con los obstaculos, asi que la distancia del dino al obstaculo se podria calcular
#! directo como la "x" del obstaculo tal vez.

# Definición de constantes globales
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Cargar imágenes para el juego
RUNNING = [pygame.image.load(os.path.join("assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("assets/Dino", "DinoRun2.png"))]

JUMPING = pygame.image.load(os.path.join("assets/Dino", "DinoJump.png"))

DUCKING = [pygame.image.load(os.path.join("assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("assets/Cactus", "SmallCactus2.png")),
                # pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))
                ]

LARGE_CACTUS = [pygame.image.load(os.path.join("assets/Cactus", "LargeCactus1.png")),
                # pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("assets/Cactus", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join("assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("assets/Bird", "Bird2.png"))]

# CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

# Clase para representar al dinosaurio del juego
#todo ===================[Dino]===================
class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8
    GRAVITY = 0.6

    def __init__(self):
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
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        #? =====[Teclas que se presionan]=====

        #* Cuando juegue la red neuronal habria que simular que presiona estas teclas
        if userInput[0] and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True

        #? Bajar rapido
        elif userInput[1] and self.dino_jump:
            self.jump_vel -= 6

        elif userInput[1] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput[1]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    #* Necesitamos solo su eje y
    def getDinoData(self):
        return (self.dino_rect.y)

#? ===================[Estados del Dino]===================
    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.width
        self.dino_rect.height
        self.dino_rect.x = self.X_POS
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
        #! Hitbox check
        # pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(self.dino_rect))
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

#todo ===================[Cloud]===================
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

#todo ===================[Obstaculos]===================
class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed

    def draw(self, SCREEN):
        #! Hitbox check
        # pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(self.rect))
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
        self.rect.y = random.choice([220, 270, 325])
        # self.rect.y = random.choice([270])
        # print(f"bird.self.rect = {self.rect}")
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        #! Hitbox check
        # pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(self.rect))
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1


#todo ===================[MAIN]===================
def main():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    obstacles = deque([])      #? variable que nos interesa para la red neuronal

    # cloud = Cloud()
    game_speed = 14     #? variable que nos interesa para la red neuronal
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0      #? variable que nos interesa para el algoritmo evolutivo
    font = pygame.font.Font('assets/PressStart2P-Regular.ttf', 20)

    time_prev = time.time()
    time_next_obstacle = 10

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:   # aumenta la velocidad del juego cada 100 puntos
            game_speed += 0.4

        text = font.render("Points: " + str(points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (920, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    #* ===============[Ciclo principal del juego]===============
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        SCREEN.fill((255, 255, 255))

        #? ===========[Obstaculos]===========

        #* Generacion de los obstaculos: 
        if len(obstacles) == 0 or (time.time() - time_prev) > time_next_obstacle:
            time_prev = time.time()
            time_next_obstacle = random.uniform(0.5, 3)
            idx_obs = random.randint(0, 2)
            match idx_obs:
                case 0:
                    obstacles.append(SmallCactus(SMALL_CACTUS))
                case 1:
                    obstacles.append(LargeCactus(LARGE_CACTUS))
                case 2:
                    obstacles.append(Bird(BIRD))

        
        #* Recorrer y analizar cada uno de los obstaculos
        for obstacle in obstacles:

            # Ajustar el área de colisión para el dinosaurio
            # Inflate ajusta un rectangulo alrededor de su centro, que sera la caja de colision
            dino_collision_rect = player.dino_rect.inflate(-50, -5)
            #! check inflate
            # pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(dino_collision_rect))

            # Ajustar el área de colisión para las aves
            if isinstance(obstacle, Bird):
                bird_collision_rect = obstacle.rect.inflate(-60, -5)
            else:
                bird_collision_rect = obstacle.rect.inflate(-10, 0)

            #! check inflate
            # pygame.draw.rect(SCREEN, (0, 0, 0), pygame.Rect(bird_collision_rect))

            # Verificar colisión entre el dinosaurio y el obstáculo
            if dino_collision_rect.colliderect(bird_collision_rect):
                pygame.time.delay(200)
                return points

            obstacle.draw(SCREEN)
            obstacle.update()

        # Chequear y eliminar el ultimo elemento
        obs_data = obstacles[0].getObstacleData()
        if obs_data[0] < -obs_data[3]:
            obstacles.popleft()
            # print("borrar")


        #? ===========[DINO, input]===========

        # lista de UP y DOWN
        userInput = pygame.key.get_pressed()
        userInput = [userInput[pygame.K_UP], userInput[pygame.K_DOWN]]
        # print(userInput)
        player.draw(SCREEN)
        player.update(userInput)

        background()

        # cloud.draw(SCREEN)
        # cloud.update()

        score()

        clock.tick(50)
        pygame.display.update()


# Función que muestra el menú inicial y maneja reinicios
def menu():
    global points
    run = True

    SCREEN.fill((255, 255, 255))
    font = pygame.font.Font('assets/PressStart2P-Regular.ttf', 30)

    # Menu principal, cuando recien empieza el juego
    text = font.render("Press any Key to Start", True, (0, 0, 0))
    textRect = text.get_rect()
    textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    SCREEN.blit(text, textRect)
    SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
    pygame.display.update()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main()

                # Menu de restart
                SCREEN.fill((255, 255, 255))
                font = pygame.font.Font('assets/PressStart2P-Regular.ttf', 30)
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
