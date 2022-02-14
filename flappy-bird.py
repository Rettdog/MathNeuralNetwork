import sys
import pygame
import numpy as np
import time


class FlappyBirdGame():

    def __init__(self):

        pygame.init()

        self.size = self.width, self.height = 500, 600

        self.isRunning = True

        self.black = 0, 0, 0

        self.screen = pygame.display.set_mode(self.size)

        self.gravity = 0.2
        self.jumpVel = -7

        self.jumpBuffer = 20

        self.buffer = 0

        self.pipeWidth = 100
        self.pipeGap = 200

        self.pipeBuffer = 150
        self.pipeTimer = self.pipeBuffer
        self.pipeSpeed = -2

        self.minPipeHeight = 150
        self.maxPipeHeight = 450

        self.score = 0
        self.scoreBuffer = 50
        self.scoreTimer = self.scoreBuffer

    def play(self):
        # Begin Program
        bird = Bird(100, 100, 25)

        pipes = []

        pipes.append(Pipe(500, 300, self.pipeWidth,
                     self.pipeSpeed, self.pipeGap, self.height))

        # clock = pygame.time.Clock()

        while(self.isRunning):

            # clock.tick(60)

            if self.buffer > 0:
                self.buffer -= 1

            if self.pipeTimer <= 0:
                self.pipeTimer = self.pipeBuffer
                pipes.append(Pipe(500, np.random.randint(
                    self.minPipeHeight, self.maxPipeHeight), self.pipeWidth, self.pipeSpeed, self.pipeGap, self.height))

            self.pipeTimer -= 1

            # Check exit button
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.isRunning = False

            # Update Positions

            bird.update(self.gravity)

            if bird.y > self.height-bird.radius:
                bird.y = self.height-bird.radius
                bird.velocity = 0
            if bird.y < bird.radius:
                bird.y = bird.radius
                bird.velocity = 0

            for pipe in pipes:
                if pipe.update():
                    pipes.remove(pipe)

            # Check Collision with first pipe

            if self.checkCollision(pipes[0], bird):
                self.isRunning = False

            # Increase Score

            if self.scoreTimer <= 0:
                self.scoreTimer = self.scoreBuffer
                self.score += 1

            self.scoreTimer -= 1

            # Check key presses
            keys = pygame.key.get_pressed()

            if keys[pygame.K_x]:
                self.isRunning = False

            # jump
            if keys[pygame.K_SPACE] and self.buffer == 0:
                bird.velocity = self.jumpVel
                self.buffer = self.jumpBuffer

            # Draw screen

            self.screen.fill(self.black)

            for pipe in pipes:
                pipe.draw(self.screen)

            bird.draw(self.screen)

            pygame.display.flip()

            time.sleep(10 / 1000)

        print(f"Final Score: {self.score}")
        return self.score

    def checkCollision(self, pipe, bird):
        if pipe.x < bird.x + bird.radius and pipe.x + pipe.width > bird.x - bird.radius and (pipe.y - pipe.pipeGap/2 > bird.y - bird.radius or pipe.y + pipe.pipeGap/2 < bird.y + bird.radius):
            return True
        return False


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

    def update(self, gravity):
        self.velocity += gravity
        self.y += self.velocity


class Pipe(GameObject):

    def __init__(self, p_x, p_y, p_width, p_speed, p_gap, p_screenHeight):
        self.x = p_x
        self.y = p_y
        self.width = p_width
        self.velocity = p_speed
        self.pipeGap = p_gap
        self.screenHeight = p_screenHeight

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(
            self.x, 0, self.width, self.y-self.pipeGap/2))
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(
            self.x, self.y+self.pipeGap/2, self.width, self.screenHeight-self.y-self.pipeGap/2))

    def update(self):
        self.x += self.velocity
        if self.x + self.width + self.velocity < 0:
            return True
        return False


game = FlappyBirdGame()
score = game.play()
