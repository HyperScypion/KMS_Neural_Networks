import os
import pygame
import numpy as np
from dataset import Dataset



# initialize the pygame module
pygame.init()

# initialize screen 
screen = pygame.display.set_mode((300, 300))
pygame.display.set_caption("Hand written digits predictor")

# width and height of each of cells in the grid
width, height = 20, 20

# margin of the grid
margin = 5

# grid 
grid = np.zeros((7, 5))

# main loop
while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			exit(0)
		elif event.type == pygame.MOUSEBUTTONDOWN:
			pos = pygame.mouse.get_pos()
			print('Current possition of the mouse {}'.format(pos))
			column = pos[0] // (width + margin)
			row = pos[1] // (height + margin)
			print('Column={}, Row={}'.format(column, row))
			
			# set position on grid
			try:
				if grid[row, column] == 0:
					grid[row, column] = 1
				else:
					grid[row, column] = 0
				print('Grid values {}'.format(grid.flatten()))
			except:
				print('Out of bound')

	for row in range(7):
		for column in range(5):
			color = (255, 255, 255)
			if grid[row, column] == 1:
				color = (0, 255, 0)
			pygame.draw.rect(screen, color,
						[(margin + width) * column + margin,
						(margin + height) * row + margin,
						width,
						height])
	pygame.display.flip()

pygame.quit()

