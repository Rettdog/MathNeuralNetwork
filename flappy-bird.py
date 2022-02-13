import sys
import pygame
import numpy as np
import time

pygame.init()

size = width, height = 500, 600

isRunning = True

black = 0, 0, 0

screen = pygame.display.set_mode(size)

gravity = 0.2
jumpVel = -7

jumpBuffer = 20

buffer = 0

pipeWidth = 100
pipeGap = 200

pipeBuffer = 150
pipeTimer = pipeBuffer
pipeSpeed = -2

minPipeHeight = 150
maxPipeHeight = 450

score = 0
scoreBuffer = 50
scoreTimer = scoreBuffer


class GameObject():

    def __init__(self, p_x, p_y, p_width, p_height):
        self.x = p_x
        self.y = p_y
        self.width = p_width
        self.height = p_height

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), self.x,
                         self.y, self.width, self.height)


class Bird(GameObject):

    def __init__(self, p_x, p_y, p_radius):
        self.x = p_x
        self.y = p_y
        self.radius = p_radius
        self.velocity = 0

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0),
                           (self.x, self.y), self.radius)

    def update(self):
        self.velocity += gravity
        self.y += self.velocity


class Pipe(GameObject):

    def __init__(self, p_x, p_y, p_width):
        self.x = p_x
        self.y = p_y
        self.width = p_width
        self.velocity = pipeSpeed

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(
            self.x, 0, self.width, self.y-pipeGap/2))
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(
            self.x, self.y+pipeGap/2, self.width, height-self.y-pipeGap/2))

    def update(self):
        self.x += self.velocity
        if self.x + self.width + self.velocity < 0:
            return True
        return False


def checkCollision(pipe, bird):
    if pipe.x < bird.x + bird.radius and pipe.x + pipe.width > bird.x - bird.radius and (pipe.y - pipeGap/2 > bird.y - bird.radius or pipe.y + pipeGap/2 < bird.y + bird.radius):
        return True
    return False


# Begin Program
bird = Bird(100, 100, 25)

pipes = []

pipes.append(Pipe(500, 300, pipeWidth))

# clock = pygame.time.Clock()

while(isRunning):

    # clock.tick(60)

    if buffer > 0:
        buffer -= 1

    if pipeTimer <= 0:
        pipeTimer = pipeBuffer
        pipes.append(Pipe(500, np.random.randint(
            minPipeHeight, maxPipeHeight), pipeWidth))

    pipeTimer -= 1

    # Check exit button
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False

    # Update Positions

    bird.update()

    if bird.y > height-bird.radius:
        bird.y = height-bird.radius
        bird.velocity = 0
    if bird.y < bird.radius:
        bird.y = bird.radius
        bird.velocity = 0

    for pipe in pipes:
        if pipe.update():
            pipes.remove(pipe)

    # Check Collision with first pipe

    if checkCollision(pipes[0], bird):
        isRunning = False

    # Increase Score

    if scoreTimer <= 0:
        scoreTimer = scoreBuffer
        score += 1

    scoreTimer -= 1

    # Check key presses
    keys = pygame.key.get_pressed()

    if keys[pygame.K_x]:
        isRunning = False

    # jump
    if keys[pygame.K_SPACE] and buffer == 0:
        bird.velocity = jumpVel
        buffer = jumpBuffer

    # Draw screen

    screen.fill(black)

    for pipe in pipes:
        pipe.draw(screen)

    bird.draw(screen)

    pygame.display.flip()

    time.sleep(10 / 1000)

print(f"Final Score: {score}")
