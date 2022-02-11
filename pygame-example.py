import sys
import pygame
import numpy as np
import math
pygame.init()

size = width, height = 800, 600
speed = np.array([0.1, 0.0])
black = 0, 0, 0

screen = pygame.display.set_mode(size)

ballX = 180
ballY = 100
ballRadius = 40

maxHeights = np.empty(1)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    ballX += speed[0]
    ballY += speed[1]
    if ballX-ballRadius < 0 or ballX+ballRadius > width:
        speed[0] = -speed[0]

        speed = 1.1*speed
        speed[1] = -abs(speed[1])
    if ballY-ballRadius < 0 or ballY+ballRadius > height:
        speed[1] = -0.95*speed[1]


    speed[1] += 0.0001

    if speed[0] > 10:
        speed[0] = 10
    if speed[1] > 10:
        speed[1] = 10

    screen.fill(black)
    pygame.draw.circle(screen, (255, 0, 0), (ballX, ballY), ballRadius)
    pygame.display.flip()
