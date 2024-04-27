# 0 : ismeretlen cellák
# 1 : hajót tartalmazó cella
# 2 : találat
# 3 : mellé
# 4 : elsüllyedt hajó
# 5 : ismert hajóhoz közeli találat

import pygame
import random
import torch
import numpy as np
from training import DQN

pygame.init()
pygame.display.set_caption("BattleShip with AI")
pygame.font.init()

WIDTH = 900
HEIGHT = 800
BLUE = (31,47,69)
BACKGROUND = (56,93,141)
SHIPS = (207, 193, 189)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (134,239,172)
RED =  (239, 68, 68)
ORANGE = (253,186,116)
COLORS = {'0': BLUE, '2': ORANGE, '3': RED, '4': GREEN, '5': BLUE}
display = pygame.display.set_mode((WIDTH, HEIGHT))

CELLSIZE = 30
ROWS = 10
COLS = 10

class Ship:
    def __init__(self, size):
        self.x = random.randrange(0, 9)
        self.y = random.randrange(0, 9)
        self.size = size
        self.position = random.choice(["horizontal", "vertical"])
        self.coordinates = self.shipCoordinates()

    def shipCoordinates(self):
        firstCoord = self.y * 10 + self.x
        if self.position == 'horizontal':
            return [firstCoord + i for i in range(self.size)]
        elif self.position == 'vertical':
            return [firstCoord + i * 10 for i in range(self.size)]
        
class Player:
    def __init__(self):
        self.ships = []
        self.ocean = ["0" for i in range(100)]
        shipSize = [5, 4, 3, 2, 2]
        self.shipsOnBoard(shipSize)
        coordInList = [ship.coordinates for ship in self.ships]
        self.shipsList = [i for l in coordInList for i in l]

    def shipsOnBoard(self, shipSize):
        for s in shipSize:
            placed = False
            while not placed:
                ship = Ship(s)

                canBePlaced = True
                for i in ship.coordinates:
                    if i >= 100:
                        canBePlaced = False
                        break

                    for placedShip in self.ships:
                        if i in placedShip.coordinates:
                            canBePlaced = False
                            break

                    new_x = i // 10
                    new_y = i % 10
                    if new_x != ship.x and new_y != ship.y:
                        canBePlaced = False
                        break

                if canBePlaced:
                    self.ships.append(ship)
                    placed = True

    def printShips(self):
        coordinates = ["0" if i not in self.shipsList else "1" for i in range(100)]
        for x in range(10):
            print(" ".join(coordinates[(x-1)*10:x*10]))

class BattleShip:
    def __init__(self, model_path):
        self.player1 = Player()
        self.player2 = BattleShipAI(model_path)
        self.player_turn = True
        self.gameOver = False
        self.sunkShips = []
        self.taken_actions_player = set()
        self.taken_actions_opponent = set()

    def playersTurn(self, i):
        player = self.player1 if self.player_turn else self.player2
        opponent = self.player2 if self.player_turn else self.player1
        hit = False

        if i in (self.taken_actions_player if self.player_turn else self.taken_actions_opponent):
            print(f"Cell {i} has been already chosen. Try again!")
            return

        if self.player_turn:
            self.taken_actions_player.add(i)
        else:
            self.taken_actions_opponent.add(i)

        if i in opponent.shipsList:
            player.ocean[i] = "2"
            self.updateOnHit(player.ocean, i)
            sunk_ship = self.checkSunkShips(player, opponent)
            hit = True
            if sunk_ship:
                self.updateSunk(player.ocean, sunk_ship)
                self.sunkShips.append(sunk_ship)

        else:
            player.ocean[i] = "3"

        self.isGameOver(opponent)

        if not hit:
            self.player_turn = not self.player_turn

    def markNearby(self, ocean, index):
        board_size = 10
        nearby_coords = []
        row = index // board_size
        col = index % board_size

        if row > 0:
            nearby_coords.append(index - board_size)
        if row < board_size - 1:
            nearby_coords.append(index + board_size)
        if col > 0:
            nearby_coords.append(index - 1)
        if col < board_size - 1:
            nearby_coords.append(index + 1)

        for index in nearby_coords:
            if ocean[index] == "0":
                ocean[index] = "5"

    def updateOnHit(self, ocean, index):
        ocean[index] = "2"
        self.markNearby(ocean, index)

    def updateSunk(self, ocean, ship):
        for coord in ship.coordinates:
            ocean[coord] = "4"
            self.resetCoords(ocean, coord)

    def checkSunkShips(self, player, opponent):
        for ship in opponent.ships:
            if ship not in self.sunkShips and all(player.ocean[coord] == "2" for coord in ship.coordinates):
                return ship
        return None

    def resetCoords(self, ocean, index):
        board_size = 10
        nearby_coords = []

        if index % board_size > 0:
            nearby_coords.append(index - 1)
        if index % board_size < board_size - 1:
            nearby_coords.append(index + 1)
        if index >= board_size:
            nearby_coords.append(index - board_size)
        if index < board_size * (board_size - 1):
            nearby_coords.append(index + board_size)

        for index in nearby_coords:
            if ocean[index] == "5":
                if not any(ocean[index + offset] == "2" for offset in [-1, 1, -board_size, board_size] if 0 <= index + offset < 100):
                    ocean[index] = "0"

    def isGameOver(self, opponent):
        if all(ship in self.sunkShips for ship in opponent.ships):
            self.gameOver = True
            self.result = 'Player' if self.player_turn else 'AI'

    def printFullGrid(self, ocean):
        print("Board:")
        for row in range(10):
            for col in range(10):
                cell = ocean[row * 10 + col]
                print(cell, end=" ")
            print()

    def getState(self):
        state_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        state = [state_mapping[cell] for cell in self.player2.ocean]

        state = np.array(state, dtype=np.float32)

        state = state.flatten()

        return state

class Button:
	def __init__(self, img, pos, input, font, color, hover_color):
		self.img = img
		self.x = pos[0]
		self.y = pos[1]
		self.font = font
		self.color, self.hover_color = color, hover_color
		self.input = input
		self.text = self.font.render(self.input, True, self.color)
		self.rect = self.img.get_rect(center=(self.x, self.y))
		self.text_rect = self.text.get_rect(center=(self.x, self.y))

	def updateButton(self, screen):
		screen.blit(self.img, self.rect)
		screen.blit(self.text, self.text_rect)

	def buttonClick(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			return True
		return False

	def hoverColor(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			self.text = self.font.render(self.input, True, self.hover_color)
		else:
			self.text = self.font.render(self.input, True, self.color)

class BattleShipAI(Player):
    def __init__(self, model_path):
        super(BattleShipAI, self).__init__()
        self.model = DQN(100, 128, 100)
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def selectAction(self, state, valid_actions):
        state = [int(cell) for row in self.ocean for cell in row]
        state_tensor = torch.tensor(np.array(state, dtype=np.float32), dtype=torch.float32).flatten().unsqueeze(0)

        with torch.no_grad():
            predicted_q_values = self.model(state_tensor)

        masked_q_values = predicted_q_values.clone().squeeze()

        for i in range(len(masked_q_values)):
            if i not in valid_actions:
                masked_q_values[i] = float('-inf')

        for idx, value in enumerate(state):
            if value == 5:
                masked_q_values[idx] += 1
            elif value == 0:
                masked_q_values[idx] -= 0.5

        action = torch.argmax(masked_q_values).item()

        return action

def drawBoard(player, marginLeft = 0, marginTop = 0, search = False):
    for i in range(100):
        x = marginLeft + i % 10 * CELLSIZE
        y = marginTop + i // 10 * CELLSIZE
        cell = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
        pygame.draw.rect(display, BLUE, cell)
        pygame.draw.rect(display, WHITE, cell, 1)
        if search:
            x += CELLSIZE // 2
            y += CELLSIZE // 2
            rect_width = CELLSIZE // 2
            rect_height = CELLSIZE // 2
            pygame.draw.rect(display, COLORS[player.ocean[i]], (x - rect_width // 2, y - rect_height // 2, rect_width, rect_height))

def drawShips(player, marginLeft = 0, marginTop = 0):
    for ship in player.ships:
        x = marginLeft + ship.y * CELLSIZE + 7
        y = marginTop + ship.x * CELLSIZE + 7
        if ship.position == "horizontal":
            width = ship.size * CELLSIZE - 14
            height = CELLSIZE - 14
        elif ship.position == "vertical":
            width = CELLSIZE - 14
            height = ship.size * CELLSIZE - 14
        cell = pygame.Rect(x, y, width, height)
        pygame.draw.rect(display, SHIPS, cell)

def getFont(fontsize):
    return pygame.font.Font("assets/PixelifySans-VariableFont_wght.ttf", fontsize)

def game(model_path):
    battleship = BattleShip(model_path)

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if battleship.player_turn:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    if event.button == 1:
                        x1 = x - 100
                        y1 = y - 50

                        if 0 <= x1 < 10 * CELLSIZE and 0 <= y1 < 10 * CELLSIZE:
                            row = x1 // CELLSIZE
                            col = y1 // CELLSIZE
                            coord = col * 10 + row
                            battleship.playersTurn(coord)
                            print(coord)

            else:
                state = battleship.getState()
                valid_actions = [i for i in range(100) if i not in battleship.taken_actions_opponent]
                action = battleship.player2.selectAction(state, valid_actions)
                battleship.playersTurn(action)
                print(action)

        display.fill(BACKGROUND)
        drawBoard(battleship.player1, 100, 50, search = True)
        drawBoard(battleship.player2, 100, 450)
        drawBoard(battleship.player1, 500, 50)
        drawBoard(battleship.player2, 500, 450, search = True)

        drawShips(battleship.player1, 100, 450)
        # drawShips(battleship.player2, 500, 50) 

        if battleship.gameOver:
            text = battleship.result + ' wins! Press right mouse button to play again.'
            RESULT = getFont(30).render(text, False, WHITE)
            RESULT_RECT = RESULT.get_rect(center=(450, 400))
            display.blit(RESULT, RESULT_RECT)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    battleship = BattleShip(model_path)

        pygame.display.flip()

def main_menu():
    while True:
        display.fill(BACKGROUND)
        
        MENU_TEXT = getFont(75).render("WELCOME TO BATTLESHIP", True, WHITE)

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_RECT = MENU_TEXT.get_rect(center=(450, 100))

        SHIP_IMG = pygame.image.load("assets/ship.png")

        SHIP_RECT = SHIP_IMG.get_rect(center=(450, 300))

        PLAY_BUTTON = Button(pygame.image.load("assets/button1.png"), (450, 500), "PLAY", getFont(50), BLACK, RED)
        
        QUIT_BUTTON = Button(pygame.image.load("assets/button1.png"), (450, 650), "QUIT", getFont(50), BLACK, RED)
        
        display.blit(MENU_TEXT, MENU_RECT)
        display.blit(SHIP_IMG, SHIP_RECT)

        for button in [PLAY_BUTTON, QUIT_BUTTON]:
            button.hoverColor(MENU_MOUSE_POS)
            button.updateButton(display)

        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.buttonClick(MENU_MOUSE_POS):
                    game(model_path)
                if QUIT_BUTTON.buttonClick(MENU_MOUSE_POS):
                    pygame.quit()

if __name__ == "__main__":
    model_path = 'models/modell.pth'
    main_menu()