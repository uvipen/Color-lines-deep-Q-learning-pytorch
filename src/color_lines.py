from pygame import init, display, image, font, Surface, transform
from pygame.surfarray import array3d
from random import randint
from math import sqrt, sin
import numpy as np
import torch
import copy
import cv2

TILES_COUNT = 9
STARTING_AMOUNT_OF_BALLS = 5
NEW_PENDING_BALLS_PER_TURN = 3
WINDOW_CAPTION = 'Color Lines'

# BALL_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255))
BALL_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
BALL_COLOR_COUNT = len(BALL_COLORS)
BACKGROUND_COLOR = 200, 200, 200
SCORE_PATTERN = {"2": 0, "3": 0, "4": 0, "5": 0}


# From gameState
def get_empty_square_matrix(size):
    return [[0 for _ in range(size)] for i in range(size)]


def get_reward_score(sequence_length):
    "0 points for less than 5 balls, 5 for 5, 7 for 6, 10 for 7, 14 for 8"
    if sequence_length < 5:
        return 0
    rewardable_balls = sequence_length - 4
    return 4 + (1 + rewardable_balls) * rewardable_balls // 2


def add_diagonal_offset(coords, value):
    return (coords[0] + value, coords[1] + value)


def generate_ball_image(tile_size, ball_radius, ball_diagonal_offset, color):
    red, green, blue = color
    img = Surface((tile_size, tile_size))
    trans_color = img.get_at((0, 0))
    img.set_colorkey(trans_color)
    ball_top = ball_left = tile_size // 2 - ball_radius + ball_diagonal_offset

    for x in range(2 * ball_radius):
        for y in range(2 * ball_radius):
            dx = (x - ball_radius) ** 2
            dy = (y - ball_radius) ** 2
            if dx + dy < ball_radius * ball_radius:
                brightness = 1 - sqrt((x * x + y * y) / 8) / ball_radius
                img.set_at((ball_left + x, ball_top + y), (brightness * red, brightness * green, brightness * blue))
    return img


def generate_shadow_image(tile_size, ball_radius):
    img = Surface((tile_size, tile_size))
    trans_color = img.get_at((0, 0))
    img.set_colorkey(trans_color)
    ball_top = ball_left = 1 + tile_size // 2 - ball_radius

    for x in range(2 * ball_radius):
        for y in range(2 * ball_radius):
            dx = (x - ball_radius) ** 2
            dy = (y - ball_radius) ** 2
            distance_from_center = sqrt(dx + dy) / ball_radius
            if distance_from_center <= 1:
                shadow_color = 0 + 75 * distance_from_center
                img.set_at((ball_left + x, ball_top + y), (shadow_color, shadow_color, shadow_color))
    return img


class ColorLines(object):
    init()
    score = 0
    gameover = False
    empty_tile = image.load("src/Square.png")
    tile_size = empty_tile.get_width()

    # Score text
    font.init()
    scoreFont = font.SysFont("Georgia", 30)
    scoreColor = (43, 91, 132)
    scoreCoords = (30, 5)

    # Grid and screen sizes
    grid_left = 30
    grid_top = 50

    grid_right_margin = 30
    grid_bottom_margin = 30

    ball_radius = tile_size // 3
    tile_dimentions = (tile_size, tile_size)

    ball_diagonal_offset = - tile_size // 15
    shadow_animation_movement_range = tile_size // 7

    pending_ball_image_size = tile_size // 2
    pending_ball_image_offset = (tile_size - pending_ball_image_size) // 2 - ball_diagonal_offset

    grid_right = grid_left + tile_size * TILES_COUNT
    grid_bottom = grid_top + tile_size * TILES_COUNT

    screen_size = grid_right + grid_right_margin, grid_bottom + grid_bottom_margin

    screen = display.set_mode(screen_size)
    display.set_caption(WINDOW_CAPTION)

    def __init__(self):
        self.balls = get_empty_square_matrix(TILES_COUNT)
        self.pending_balls = get_empty_square_matrix(TILES_COUNT)
        self.add_random_balls(STARTING_AMOUNT_OF_BALLS, NEW_PENDING_BALLS_PER_TURN)
        self.ball_images = [generate_ball_image(self.tile_size, self.ball_radius, self.ball_diagonal_offset, color) for
                            color in BALL_COLORS]
        self.pending_ball_images = []

        for imageIndex in range(BALL_COLOR_COUNT):
            newSize = (self.pending_ball_image_size, self.pending_ball_image_size)
            self.pending_ball_images.append(transform.scale(self.ball_images[imageIndex], newSize))

        self.tile_image = transform.smoothscale(self.empty_tile, self.tile_dimentions)
        self.ball_shadow_image = generate_shadow_image(self.tile_size, self.ball_radius)
        self.render()

    def add_random_balls(self, quantity, pendingBallsQuantity):
        freePositions = []
        for col in range(TILES_COUNT):
            for row in range(TILES_COUNT):
                if self.balls[col][row] == 0:
                    freePositions.append((col, row))
        # Adding balls
        for i in range(min(quantity, len(freePositions))):
            posIndex = randint(0, len(freePositions) - 1)
            position = freePositions.pop(posIndex)
            self.balls[position[0]][position[1]] = randint(1, BALL_COLOR_COUNT)

        # Adding pending balls
        for i in range(min(pendingBallsQuantity, len(freePositions))):
            posIndex = randint(0, len(freePositions) - 1)
            position = freePositions.pop(posIndex)
            self.pending_balls[position[0]][position[1]] = randint(1, BALL_COLOR_COUNT)

        return len(freePositions)

    def tile_coordinates(self, col, row):
        return self.grid_left + self.tile_size * col, self.grid_top + self.tile_size * row

    def draw_tile(self, col, row, ball_color, shadowOffset, isPendingBall=False):
        coord = self.tile_coordinates(col, row)
        self.screen.blit(self.tile_image, coord)
        if ball_color > 0:
            if not isPendingBall:
                self.screen.blit(self.ball_shadow_image, add_diagonal_offset(coord, shadowOffset))
                self.screen.blit(self.ball_images[ball_color - 1], coord)
            else:
                self.screen.blit(self.pending_ball_images[ball_color - 1],
                                 add_diagonal_offset(coord, self.pending_ball_image_offset))

        return (*coord, *self.tile_dimentions)

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)
        for col in range(TILES_COUNT):
            for row in range(TILES_COUNT):
                ball_color = self.balls[col][row]
                pending_ball_color = self.pending_balls[col][row]
                if pending_ball_color > 0:
                    self.draw_tile(col, row, pending_ball_color, 2, True)
                else:
                    self.draw_tile(col, row, ball_color, 2, False)

        scoreText = 'Score: ' + str(self.score)
        textSurface = self.scoreFont.render(scoreText, False, self.scoreColor)
        self.screen.blit(textSurface, self.scoreCoords)
        display.flip()
        return np.transpose(cv2.cvtColor(array3d(display.get_surface()), cv2.COLOR_RGB2BGR), (1, 0, 2))

    def animateSelectedBall(self, balls, coords, timeCounter):
        color = balls[coords[0]][coords[1]]
        shadowOffset = self.ball_diagonal_offset + self.shadow_animation_movement_range * (
                1 + sin(timeCounter / 2.5)) / 2
        updateArea = self.draw_tile(*coords, color, shadowOffset)
        display.update(updateArea)

    def check_board_click(self, x, y):
        if x >= self.grid_left and x < self.grid_right and y >= self.grid_top and y < self.grid_bottom:
            col = (x - self.grid_left) // self.tile_size
            row = (y - self.grid_top) // self.tile_size
            self.on_tile_click(col, row)

    def get_next_states(self):
        states = {}
        source_balls = np.nonzero(np.array(self.balls))
        self.mix_balls = np.array(self.balls) + np.array(self.pending_balls)
        curr_total_score, curr_col_score, curr_row_score, curr_tl_br_score, curr_bl_tr_score = self.get_current_score()
        num_free_tiles = np.count_nonzero(self.mix_balls == 0)
        for s_col, s_row in zip(*source_balls):
            destination_balls = np.nonzero(
                np.array(self.get_reachable_tile_map((s_col, s_row))) * (np.array(self.pending_balls) == 0).astype(
                    np.int))

            for d_col, d_row in zip(*destination_balls):
                if s_col == d_col and s_row == d_row:
                    continue
                score = copy.deepcopy(SCORE_PATTERN)
                col_score, row_score, tl_br_score, bl_tr_score = self.get_next_score(s_col, s_row, d_col, d_row)
                for key in score.keys():
                    score[key] = curr_total_score[key] - curr_col_score[str(s_col)][key] - curr_row_score[str(s_row)][
                        key] + col_score[str(s_col)][key] + row_score[str(s_row)][key]
                    if s_col != d_col:
                        score[key] -= curr_col_score[str(d_col)][key]
                        score[key] += col_score[str(d_col)][key]
                    if s_row != d_row:
                        score[key] -= curr_row_score[str(d_row)][key]
                        score[key] += row_score[str(d_row)][key]
                    for k in tl_br_score.keys():
                        if k in curr_tl_br_score.keys():
                            score[key] -= curr_tl_br_score[k][key]
                            score[key] += tl_br_score[k][key]

                    for k in bl_tr_score.keys():
                        if k in curr_bl_tr_score.keys():
                            score[key] -= curr_bl_tr_score[k][key]
                            score[key] += bl_tr_score[k][key]

                states[(s_col, s_row, d_col, d_row)] = torch.FloatTensor(
                    list(score.values()) + [num_free_tiles + 5 * score["5"]])

        return states

    def consume_consequent_balls(self, startPosition, getNextPosition, balls_to_remove):
        sequenceColor = 0
        sequence = []
        position = startPosition
        isLastStep = False

        while not isLastStep:
            currentColor = self.balls[position[0]][position[1]]
            if currentColor == sequenceColor and sequenceColor != 0:
                sequence.append(position)

            nextPosition = getNextPosition(position)
            isLastStep = nextPosition[0] >= TILES_COUNT or nextPosition[1] >= TILES_COUNT or nextPosition[0] < 0 or \
                         nextPosition[1] < 0
            if currentColor != sequenceColor or isLastStep:
                reward = get_reward_score(len(sequence))
                if reward > 0:
                    self.score += reward
                    for col, row in sequence:
                        balls_to_remove[col][row] = 1

            if currentColor != sequenceColor:
                sequenceColor = currentColor
                sequence.clear()
                sequence.append(position)

            position = nextPosition

    def remove_aligned_balls(self):
        balls_to_remove = get_empty_square_matrix(TILES_COUNT)
        # print (balls_to_remove)
        # Vertical rows
        for col in range(TILES_COUNT):
            self.consume_consequent_balls((col, 0), lambda position: (position[0], position[1] + 1), balls_to_remove)

        # Horizontal rows
        for row in range(TILES_COUNT):
            self.consume_consequent_balls((0, row), lambda position: (position[0] + 1, position[1]), balls_to_remove)

        # Down-Right diagonal
        getNextPosition = lambda position: (position[0] + 1, position[1] + 1)
        for col in range(TILES_COUNT):
            self.consume_consequent_balls((col, 0), getNextPosition, balls_to_remove)
        for row in range(1, TILES_COUNT):
            self.consume_consequent_balls((0, row), getNextPosition, balls_to_remove)

        # Down-Left diagonal
        getNextPosition = lambda position: (position[0] - 1, position[1] + 1)
        for col in range(TILES_COUNT):
            self.consume_consequent_balls((col, 0), getNextPosition, balls_to_remove)
        for row in range(1, TILES_COUNT):
            self.consume_consequent_balls((TILES_COUNT - 1, row), getNextPosition, balls_to_remove)
        # print(balls_to_remove)
        totalConsumed = 0
        for col in range(TILES_COUNT):
            for row in range(TILES_COUNT):
                if balls_to_remove[col][row] == 1:
                    self.balls[col][row] = 0
                    totalConsumed += 1
        # print (totalConsumed)
        return totalConsumed

    def get_next_score(self, s_col, s_row, d_col, d_row):
        col_score = {}
        row_score = {}
        tl_br_score = {}
        bl_tr_score = {}
        mix_balls = copy.deepcopy(self.mix_balls)
        mix_balls[d_col][d_row] = mix_balls[s_col][s_row]
        mix_balls[s_col][s_row] = 0
        for curr_col, curr_row in [[s_col, s_row], [d_col, d_row]]:
            best_length = 0
            col_score[str(curr_col)] = copy.deepcopy(SCORE_PATTERN)
            for row in range(TILES_COUNT // 2 + 1):
                values, counts = np.unique(mix_balls[curr_col, row:row + TILES_COUNT // 2 + 1], return_counts=True)
                if len(counts) == 1 and 0 not in list(values):
                    best_length = 5
                    break
                if len(counts) != 2 or 0 not in list(values):
                    continue
                else:
                    ball_count = counts[1]
                    if ball_count == 1:
                        continue
                    else:
                        if ball_count > best_length:
                            best_length = ball_count
            if best_length != 0:
                col_score[str(curr_col)][str(best_length)] += 1

            best_length = 0
            row_score[str(curr_row)] = copy.deepcopy(SCORE_PATTERN)
            for col in range(TILES_COUNT // 2 + 1):
                values, counts = np.unique(mix_balls[col:col + TILES_COUNT // 2 + 1, curr_row], return_counts=True)
                if len(counts) == 1 and 0 not in list(values):
                    best_length = 5
                    break
                if len(counts) != 2 or 0 not in list(values):
                    continue
                else:
                    ball_count = counts[1]
                    if ball_count == 1:
                        continue
                    else:
                        if ball_count > best_length:
                            best_length = ball_count
            if best_length != 0:
                row_score[str(curr_row)][str(best_length)] += 1

            # Diagonal. Top left -> bottom right
            columns, rows = zip(
                *[[curr_col + i, curr_row + i] for i in range(-TILES_COUNT + 1, TILES_COUNT) if
                  0 <= min(curr_col + i, curr_row + i) <= max(curr_col + i, curr_row + i) < TILES_COUNT])
            diagonal = mix_balls[columns, rows]
            best_length = 0
            tl_br_score[(columns[0], rows[0])] = copy.deepcopy(SCORE_PATTERN)
            for first_element in range(len(diagonal) - TILES_COUNT // 2):
                values, counts = np.unique(diagonal[first_element:first_element + TILES_COUNT // 2 + 1],
                                           return_counts=True)
                if len(counts) == 1 and 0 not in list(values):
                    best_length = 5
                    break
                if len(counts) != 2 or 0 not in list(values):
                    continue
                else:
                    ball_count = counts[1]
                    if ball_count == 1:
                        continue
                    else:
                        if ball_count > best_length:
                            best_length = ball_count
            if best_length != 0:
                tl_br_score[(columns[0], rows[0])][str(best_length)] += 1

            # Diagonal. Bottom left -> top right
            min_index = min(curr_col, TILES_COUNT - 1 - curr_row)
            columns, rows = zip(
                *[[curr_col + i, curr_row - i] for i in range(-TILES_COUNT + 1, TILES_COUNT) if
                  0 <= min(curr_col + i, curr_row - i) <= max(curr_col + i, curr_row - i) < TILES_COUNT])
            diagonal = mix_balls[columns, rows]
            best_length = 0
            bl_tr_score[(curr_col - min_index, curr_row + min_index)] = copy.deepcopy(SCORE_PATTERN)
            for first_element in range(len(diagonal) - TILES_COUNT // 2):
                values, counts = np.unique(diagonal[first_element:first_element + TILES_COUNT // 2 + 1],
                                           return_counts=True)
                if len(counts) == 1 and 0 not in list(values):
                    best_length = 5
                    break
                if len(counts) != 2 or 0 not in list(values):
                    continue
                else:
                    ball_count = counts[1]
                    if ball_count == 1:
                        continue
                    else:
                        if ball_count > best_length:
                            best_length = ball_count
            if best_length != 0:
                bl_tr_score[(curr_col - min_index, curr_row + min_index)][str(best_length)] += 1
        return col_score, row_score, tl_br_score, bl_tr_score

    def get_current_score(self):
        total_score = copy.deepcopy(SCORE_PATTERN)
        col_score = {}
        row_score = {}
        tl_br_score = {}
        bl_tr_score = {}
        mix_balls = copy.deepcopy(self.mix_balls)
        for col in range(TILES_COUNT):
            best_length = 0
            col_score[str(col)] = copy.deepcopy(SCORE_PATTERN)
            for row in range(TILES_COUNT // 2 + 1):
                values, counts = np.unique(mix_balls[col, row:row + TILES_COUNT // 2 + 1], return_counts=True)
                if len(counts) == 1 and 0 not in list(values):
                    best_length = 5
                    break
                if len(counts) != 2 or 0 not in list(values):
                    continue
                else:
                    ball_count = counts[1]
                    if ball_count == 1:
                        continue
                    else:
                        if ball_count > best_length:
                            best_length = ball_count
            if best_length != 0:
                total_score[str(best_length)] += 1
                col_score[str(col)][str(best_length)] += 1

        # print(potential_score)
        for row in range(TILES_COUNT):
            best_length = 0
            row_score[str(row)] = copy.deepcopy(SCORE_PATTERN)
            for col in range(TILES_COUNT - 4):
                values, counts = np.unique(mix_balls[col:col + TILES_COUNT // 2 + 1, row], return_counts=True)
                if len(counts) == 1 and 0 not in list(values):
                    best_length = 5
                    break
                if len(counts) != 2 or 0 not in list(values):
                    continue
                else:
                    ball_count = counts[1]
                    if ball_count == 1:
                        continue
                    else:
                        if ball_count > best_length:
                            best_length = ball_count
            if best_length != 0:
                total_score[str(best_length)] += 1
                row_score[str(row)][str(best_length)] += 1

        # Diagonal. Top left -> bottom right
        for diff in range(-TILES_COUNT // 2 + 1, TILES_COUNT // 2 + 1):
            columns, rows = zip(*[[diff + i, i] for i in range(TILES_COUNT) if TILES_COUNT > diff + i >= 0])
            diagonal = mix_balls[columns, rows]
            best_length = 0
            tl_br_score[(columns[0], rows[0])] = copy.deepcopy(SCORE_PATTERN)
            for first_element in range(len(diagonal) - TILES_COUNT // 2):
                values, counts = np.unique(diagonal[first_element:first_element + TILES_COUNT // 2 + 1],
                                           return_counts=True)
                if len(counts) == 1 and 0 not in list(values):
                    best_length = 5
                    break
                if len(counts) != 2 or 0 not in list(values):
                    continue
                else:
                    ball_count = counts[1]
                    if ball_count == 1:
                        continue
                    else:
                        if ball_count > best_length:
                            best_length = ball_count
            if best_length != 0:
                total_score[str(best_length)] += 1
                tl_br_score[(columns[0], rows[0])][str(best_length)] += 1

        # Diagonal. Bottom left -> top right
        for diff in range(-TILES_COUNT // 2 + 1, TILES_COUNT // 2 + 1):
            columns, rows = zip(
                *[[diff + i, TILES_COUNT - i - 1] for i in range(TILES_COUNT) if TILES_COUNT > diff + i >= 0])
            diagonal = mix_balls[columns, rows]
            best_length = 0
            bl_tr_score[(columns[0], rows[0])] = copy.deepcopy(SCORE_PATTERN)
            for first_element in range(len(diagonal) - 4):
                values, counts = np.unique(diagonal[first_element:first_element + TILES_COUNT // 2 + 1],
                                           return_counts=True)
                if len(counts) == 1 and 0 not in list(values):
                    best_length = 5
                    break
                if len(counts) != 2 or 0 not in list(values):
                    continue
                else:
                    ball_count = counts[1]
                    if ball_count == 1:
                        continue
                    else:
                        if ball_count > best_length:
                            best_length = ball_count
            if best_length != 0:
                total_score[str(best_length)] += 1
                bl_tr_score[(columns[0], rows[0])][str(best_length)] += 1

        return total_score, col_score, row_score, tl_br_score, bl_tr_score

    def materialize_pending_balls(self):
        for col in range(TILES_COUNT):
            for row in range(TILES_COUNT):
                color = self.pending_balls[col][row]
                if color > 0:
                    self.balls[col][row] = color
                    self.pending_balls[col][row] = 0

    def get_reachable_tile_map(self, position):
        reachable_tiles = get_empty_square_matrix(TILES_COUNT)
        expanded = True
        reachable_tiles[position[0]][position[1]] = 1
        while expanded:
            expanded = False
            for col in range(TILES_COUNT):
                for row in range(TILES_COUNT):
                    if reachable_tiles[col][row] == 0 and self.balls[col][row] == 0 and \
                            ((col > 0 and reachable_tiles[col - 1][row] == 1) or
                             (col < TILES_COUNT - 1 and reachable_tiles[col + 1][row] == 1) or
                             (row > 0 and reachable_tiles[col][row - 1] == 1) or
                             (row < TILES_COUNT - 1 and reachable_tiles[col][row + 1] == 1)
                            ):
                        reachable_tiles[col][row] = 1
                        expanded = True

        return reachable_tiles

    def step(self, action):
        s_col, s_row, d_col, d_row = action
        self.balls[d_col][d_row] = self.balls[s_col][s_row]
        self.balls[s_col][s_row] = 0
        balls_removed = self.remove_aligned_balls()
        if balls_removed == 0:
            self.materialize_pending_balls()
            balls_removed += self.remove_aligned_balls()
            if balls_removed == 0:
                # If nothing is removed - adding new pending balls
                freeTilesLeft = self.add_random_balls(0, NEW_PENDING_BALLS_PER_TURN)
                if freeTilesLeft == 0:
                    self.materialize_pending_balls()
                    balls_removed += self.remove_aligned_balls()
                    if balls_removed == 0:
                        # When there're no available moves -> Game Over!
                        self.gameover = True
                        return -5, self.gameover
        return 2 ** balls_removed, self.gameover

    def reset(self):
        self.score = 0
        self.gameover = False
        self.balls = get_empty_square_matrix(TILES_COUNT)
        self.pending_balls = get_empty_square_matrix(TILES_COUNT)
        self.add_random_balls(STARTING_AMOUNT_OF_BALLS, NEW_PENDING_BALLS_PER_TURN)
        self.render()
        return torch.zeros(5, dtype=torch.float32)
