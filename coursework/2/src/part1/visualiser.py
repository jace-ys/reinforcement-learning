import cv2
import numpy as np

from environment import Environment


class QValueVisualiser:
    def __init__(self, environment, magnification=500):
        self.environment = environment
        self.magnification = magnification
        self.half_cell_length = 0.05 * self.magnification

        # Create the initial Q-values image
        self.q_values_image = np.zeros(
            (int(self.magnification), int(self.magnification), 3), dtype=np.uint8
        )

    def draw_q_values(self, q_values):
        # Create an empty image
        self.q_values_image.fill(0)

        # Loop over the grid cells and actions, and draw each Q-value
        for row in range(q_values.shape[0]):
            for col in range(q_values.shape[1]):
                # Find the Q-value range for this state
                max_q_value = np.max(q_values[row, col])
                min_q_value = np.min(q_values[row, col])
                q_value_range = max_q_value - min_q_value

                x = (col / 10.0) + 0.05
                y = (row / 10.0) + 0.05

                # Draw the Q-value for each state-action
                for action in range(q_values.shape[2]):
                    # Normalise with respect to the Q-value range
                    q_value_norm = float(
                        (q_values[row, col, action] - min_q_value) / q_value_range
                    )

                    self.__draw_q_value(x, y, action, q_value_norm)

        # Draw the grid cells
        self.__draw_grid_cells()

        # Show the image
        cv2.imshow("Q-Values", self.q_values_image)
        cv2.waitKey(1)

    def __draw_q_value(self, x, y, action, q_value_norm):
        # Compute the image coordinates of the centre of the triangle for this action
        centre_x = x * self.magnification
        centre_y = y * self.magnification

        # Compute the colour for this Q-value
        colour_r = int((1 - q_value_norm) * 255)
        colour_g = int(q_value_norm * 255)
        colour_b = 0
        colour = (colour_b, colour_g, colour_r)

        # Depending on the particular action, the triangle representing the action will be drawn in a different position on the image
        if action == 0:  # Move right
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y - self.half_cell_length
            points = np.array(
                [[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(
                self.q_values_image,
                [points],
                True,
                (0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        elif action == 1:  # Move up
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = centre_x - self.half_cell_length
            point_2_y = point_1_y
            points = np.array(
                [[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(
                self.q_values_image,
                [points],
                True,
                (0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        elif action == 2:  # Move left
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y + self.half_cell_length
            points = np.array(
                [[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(
                self.q_values_image,
                [points],
                True,
                (0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        elif action == 3:  # Move down
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = centre_x + self.half_cell_length
            point_2_y = point_1_y
            points = np.array(
                [[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(
                self.q_values_image,
                [points],
                True,
                (0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    def __draw_grid_cells(self):
        # Draw the state cell borders
        for col in range(11):
            p1 = (int((col / 10.0) * self.magnification), 0)
            p2 = (int((col / 10.0) * self.magnification), int(self.magnification))

            cv2.line(
                self.q_values_image,
                p1,
                p2,
                (255, 255, 255),
                thickness=4,
                lineType=cv2.LINE_AA,
            )

        for row in range(11):
            p1 = (0, int((row / 10.0) * self.magnification))
            p2 = (int(self.magnification), int((row / 10.0) * self.magnification))

            cv2.line(
                self.q_values_image,
                p1,
                p2,
                (255, 255, 255),
                thickness=4,
                lineType=cv2.LINE_AA,
            )
