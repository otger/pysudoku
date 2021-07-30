import pygame
from pygame_text_wrapper import TextRectException, render_textrect

import sudoku

CELL_WIDTH = 70
CELL_HEIGHT = CELL_WIDTH
SUBCELL_WIDTH = int(CELL_WIDTH/4)
SUBCELL_HEIGHT = int(CELL_HEIGHT/4)
SUBCELL_MARGIN = int((CELL_WIDTH - 3*SUBCELL_WIDTH)/6)

CELL_MARGIN = 2
BLOCK_SEP_LINE_WIDTH = 3
CELL_SEP_LINE_WIDTH = 1

BOARD_MARGIN = 5

AREA_SEPARATOR_HEIGHT = 3
AREA_SEPARATOR_LR_MARGIN = 5

EXPLANATION_AREA_HEIGHT = 150
EXPLANATION_AREA_MARGIN = BOARD_MARGIN
EXPLANATION_TEXT_SIZE = 25
EXPLANATION_TEXT_COLOR = (90, 90, 90)
EXPLANATION_AREA_PADDING = 10

ERROR_AREA_HEIGHT = 100
ERROR_AREA_MARGIN = BOARD_MARGIN
ERROR_TEXT_SIZE = 90
ERROR_AREA_PADDING = 5
ERROR_TEXT_COLOR = (179, 23, 23)

BACKGROUND_COLOR = (222, 222, 222)
CELL_EMPTY_BACKGROUND = (247, 247, 247)
CELL_FILLED_BACKGROUND = (230, 202, 230)
BLOCK_SEP_LINE_COLOR = (61, 57, 61)
CELL_SEP_LINE_COLOR = (161, 157, 161)
TEXT_SOLUTION_COLOR = (145, 13, 145)
TEXT_OPTION_COLOR = (194, 97, 194)

AREA_SEP_COLOR = (180, 170, 180)
EXPLANATION_AREA_BACKGROUND_COLOR = (195, 195, 195)

BOARD_WIDTH = 9 * (CELL_MARGIN + CELL_WIDTH + CELL_MARGIN) + 6 * CELL_SEP_LINE_WIDTH + 2 * BLOCK_SEP_LINE_WIDTH
BOARD_HEIGHT = BOARD_WIDTH
BOARD_OFFSET = (BOARD_MARGIN, BOARD_MARGIN)

SCREEN_SIZE = (BOARD_WIDTH + 2 * BOARD_MARGIN,
               BOARD_HEIGHT + 2 * BOARD_MARGIN
               + AREA_SEPARATOR_HEIGHT + EXPLANATION_AREA_HEIGHT + 2 * EXPLANATION_AREA_MARGIN
               + AREA_SEPARATOR_HEIGHT + ERROR_AREA_HEIGHT + 2 * ERROR_AREA_MARGIN)

BLOCK_SIZE = (3 * (2 * CELL_MARGIN + CELL_WIDTH) + 2 * CELL_SEP_LINE_WIDTH,
              3 * (2 * CELL_MARGIN + CELL_WIDTH) + 2 * CELL_SEP_LINE_WIDTH)

EXPLANATION_AREA_WIDTH = SCREEN_SIZE[0] - 2 * EXPLANATION_AREA_MARGIN
EXPLANATION_AREA_OFFSET = (EXPLANATION_AREA_MARGIN,
                           BOARD_MARGIN + BOARD_HEIGHT + BOARD_MARGIN + AREA_SEPARATOR_HEIGHT + EXPLANATION_AREA_MARGIN)

ERROR_AREA_OFFSET = (ERROR_AREA_MARGIN,
                     BOARD_MARGIN + BOARD_HEIGHT + BOARD_MARGIN + AREA_SEPARATOR_HEIGHT + EXPLANATION_AREA_MARGIN +
                     EXPLANATION_AREA_HEIGHT + EXPLANATION_AREA_MARGIN + AREA_SEPARATOR_HEIGHT + ERROR_AREA_MARGIN)
ERROR_AREA_WIDTH = EXPLANATION_AREA_WIDTH


class SudokuVisualize:

    def __init__(self, steps):
        pygame.display.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self.GUI = pygame.display.set_mode(SCREEN_SIZE)
        self._run = True
        self.steps = steps
        self._curr_step = 0
        # print(sorted(pygame.font.get_fonts()))

    @property
    def current_step(self) -> sudoku.Step:
        return self.steps[self._curr_step]

    def next_step(self):
        if len(self.steps) > self._curr_step + 1:
            self._curr_step += 1

    def prev_step(self):
        if self._curr_step > 0:
            self._curr_step -= 1

    def draw(self):
        self.GUI.fill(BACKGROUND_COLOR)
        sols = self.current_step.solutions
        opts = self.current_step.options
        font_sols = pygame.font.Font("Roboto/Roboto-Medium.ttf", int(CELL_WIDTH / 1.1))
        font_opts = pygame.font.Font("Roboto/Roboto-Regular.ttf", int(CELL_WIDTH / 3.3))
        font_msg = pygame.font.Font("Roboto/Roboto-Regular.ttf", EXPLANATION_TEXT_SIZE)
        font_err_msg = pygame.font.Font("Roboto/Roboto-Regular.ttf", ERROR_TEXT_SIZE)
        # draw grid lines:
        for i in range(1, 3):
            pygame.draw.line(self.GUI, BLOCK_SEP_LINE_COLOR,
                             (BOARD_OFFSET[0], i * (BLOCK_SIZE[1] + BLOCK_SEP_LINE_WIDTH) + BOARD_OFFSET[1]),
                             (BOARD_OFFSET[0] + BOARD_WIDTH,
                              i * (BLOCK_SIZE[1] + BLOCK_SEP_LINE_WIDTH) + BOARD_OFFSET[1]),
                             BLOCK_SEP_LINE_WIDTH)
            pygame.draw.line(self.GUI, BLOCK_SEP_LINE_COLOR,
                             (i * (BLOCK_SIZE[0] + BLOCK_SEP_LINE_WIDTH) + BOARD_OFFSET[0], BOARD_OFFSET[1]),
                             (i * (BLOCK_SIZE[0] + BLOCK_SEP_LINE_WIDTH) + BOARD_OFFSET[0],
                              BOARD_OFFSET[1] + BOARD_HEIGHT),
                             BLOCK_SEP_LINE_WIDTH)
        for bi in range(3):
            for bj in range(3):
                block_offset = (BOARD_MARGIN + CELL_MARGIN + bi * (
                        3 * (2 * CELL_MARGIN + CELL_WIDTH) + 2 * CELL_SEP_LINE_WIDTH + BLOCK_SEP_LINE_WIDTH),
                                BOARD_MARGIN + CELL_MARGIN + bj * (3 * (
                                        2 * CELL_MARGIN + CELL_WIDTH) + 2 * CELL_SEP_LINE_WIDTH + BLOCK_SEP_LINE_WIDTH))
                for i in range(1, 3):
                    pygame.draw.line(self.GUI, CELL_SEP_LINE_COLOR,
                                     (block_offset[0], block_offset[1] + i * (CELL_WIDTH + 2 * CELL_MARGIN)),
                                     (block_offset[0] + BLOCK_SIZE[0],
                                      block_offset[1] + i * (CELL_WIDTH + 2 * CELL_MARGIN)),
                                     CELL_SEP_LINE_WIDTH)
                    pygame.draw.line(self.GUI, CELL_SEP_LINE_COLOR,
                                     (block_offset[0] + i * (CELL_WIDTH + 2 * CELL_MARGIN), block_offset[1]),
                                     (block_offset[0] + i * (CELL_WIDTH + 2 * CELL_MARGIN),
                                      block_offset[1] + BLOCK_SIZE[1]),
                                     CELL_SEP_LINE_WIDTH)

                for br in range(3):
                    for bc in range(3):
                        y0 = block_offset[0] + \
                             br * (2 * CELL_MARGIN + CELL_WIDTH) + CELL_MARGIN
                        x0 = block_offset[1] + \
                             bc * (2 * CELL_MARGIN + CELL_WIDTH) + CELL_MARGIN
                        row = bi * 3 + br
                        col = bj * 3 + bc
                        if sols.v[row, col] != '-':
                            pygame.draw.rect(self.GUI, CELL_FILLED_BACKGROUND,
                                             pygame.Rect(x0, y0, CELL_WIDTH, CELL_HEIGHT))
                            text = font_sols.render(sols.v[row, col], True, TEXT_SOLUTION_COLOR)
                            text_rect = text.get_rect(center=(x0 + CELL_WIDTH / 2, y0 + CELL_WIDTH / 2))
                            self.GUI.blit(text, text_rect)
                        elif opts.v[row, col]:
                            vals = [int(el) for el in opts.v[row, col]]
                            for i in range(3):
                                for j in range(3):
                                    v = 3 * i + j + 1
                                    if v in vals:
                                        x0_subcell = x0 + j * (SUBCELL_WIDTH + 2 * SUBCELL_MARGIN) + SUBCELL_MARGIN
                                        y0_subcell = y0 + i * (SUBCELL_HEIGHT + 2 * SUBCELL_MARGIN) + SUBCELL_MARGIN
                                        text = font_opts.render(str(v), True, TEXT_OPTION_COLOR)
                                        text_rect = text.get_rect(center=(x0_subcell + int(SUBCELL_WIDTH/2),
                                                                          y0_subcell + int(SUBCELL_HEIGHT/2)))
                                        self.GUI.blit(text, text_rect)

                # Draw area separators

                pygame.draw.line(self.GUI, AREA_SEP_COLOR,
                                 (AREA_SEPARATOR_LR_MARGIN, BOARD_MARGIN + BOARD_HEIGHT + BOARD_MARGIN),
                                 (
                                     SCREEN_SIZE[0] - AREA_SEPARATOR_LR_MARGIN,
                                     BOARD_MARGIN + BOARD_HEIGHT + BOARD_MARGIN),
                                 AREA_SEPARATOR_HEIGHT)
                # draw message_background
                pygame.draw.rect(self.GUI, EXPLANATION_AREA_BACKGROUND_COLOR,
                                 pygame.Rect(EXPLANATION_AREA_OFFSET,
                                             (EXPLANATION_AREA_WIDTH, EXPLANATION_AREA_HEIGHT)))
                if self.current_step.msg:
                    msg_rect = pygame.Rect((EXPLANATION_AREA_OFFSET[0] + EXPLANATION_AREA_PADDING,
                                            EXPLANATION_AREA_OFFSET[1] + EXPLANATION_AREA_PADDING,
                                            EXPLANATION_AREA_WIDTH - 2 * EXPLANATION_AREA_PADDING,
                                            EXPLANATION_AREA_HEIGHT - 2 * EXPLANATION_AREA_PADDING))
                    rendered_text = render_textrect(self.current_step.msg, font_msg, msg_rect, EXPLANATION_TEXT_COLOR,
                                                    EXPLANATION_AREA_BACKGROUND_COLOR)
                    if rendered_text:
                        self.GUI.blit(rendered_text, (EXPLANATION_AREA_OFFSET[0] + EXPLANATION_AREA_PADDING,
                                                      EXPLANATION_AREA_OFFSET[1] + EXPLANATION_AREA_PADDING))

                pygame.draw.line(self.GUI, AREA_SEP_COLOR,
                                 (AREA_SEPARATOR_LR_MARGIN,
                                  BOARD_MARGIN + BOARD_HEIGHT + BOARD_MARGIN + AREA_SEPARATOR_HEIGHT + EXPLANATION_AREA_MARGIN + EXPLANATION_AREA_HEIGHT + EXPLANATION_AREA_MARGIN),
                                 (SCREEN_SIZE[0] - AREA_SEPARATOR_LR_MARGIN,
                                  BOARD_MARGIN + BOARD_HEIGHT + BOARD_MARGIN + AREA_SEPARATOR_HEIGHT + EXPLANATION_AREA_MARGIN + EXPLANATION_AREA_HEIGHT + EXPLANATION_AREA_MARGIN),
                                 AREA_SEPARATOR_HEIGHT)
                # draw error area
                pygame.draw.rect(self.GUI, EXPLANATION_AREA_BACKGROUND_COLOR,
                                 pygame.Rect(ERROR_AREA_OFFSET,
                                             (ERROR_AREA_WIDTH, ERROR_AREA_HEIGHT)))
                if self.current_step.is_broken():
                    msg_rect = pygame.Rect((ERROR_AREA_OFFSET[0] + ERROR_AREA_PADDING,
                                            ERROR_AREA_OFFSET[1] + ERROR_AREA_PADDING,
                                            ERROR_AREA_WIDTH - 2 * ERROR_AREA_PADDING,
                                            ERROR_AREA_HEIGHT - 2 * ERROR_AREA_PADDING))
                    rendered_text = render_textrect("Solution is broken", font_msg, msg_rect, ERROR_TEXT_COLOR,
                                                    EXPLANATION_AREA_BACKGROUND_COLOR)
                    if rendered_text:
                        self.GUI.blit(rendered_text, (ERROR_AREA_OFFSET[0] + ERROR_AREA_PADDING,
                                                      ERROR_AREA_OFFSET[1] + ERROR_AREA_PADDING))

        pygame.display.update()

    def run(self):
        while self._run:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    self._run = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.prev_step()
                    if event.key == pygame.K_RIGHT:
                        self.next_step()
                    if event.key == pygame.K_UP:
                        self.next_step()
                    if event.key == pygame.K_DOWN:
                        self.prev_step()

                self.draw()
                self._clock.tick(5)

    def quit(self):
        pygame.quit()


if __name__ == "__main__":
    from sudoku import Sudoku

    # init_str = '4-8---9--9---4-7----6----48-8---1-7---5--------18-24-6-3-----5-81-3--------98---7'
    init_str = '--1----9----915-------784-26-----7--5----2--14--8------4--2-8-7--349------7------'
    s = Sudoku(init_values=init_str, step_resolution=5)
    s.solve_clean()
    sv = SudokuVisualize(steps=s.steps)
    sv.run()
    sv.quit()
