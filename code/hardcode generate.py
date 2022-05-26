import random
import copy


def top(row, column, matrix):
    new_row = row - 3
    if matrix[new_row][column] != 1:
        matrix[new_row][column] = 1
        return False
    else:
        return True


def down(row, column, matrix):
    new_row = row + 3
    if matrix[new_row][column] != 1:
        matrix[new_row][column] = 1
        return False
    else:
        return True


def left(row, column, matrix):
    new_column = column - 1
    if matrix[row][new_column] != 1:
        matrix[row][new_column] = 1
        return False
    else:
        return True


def right(row, column, matrix):
    new_column = column + 1
    if matrix[row][new_column] != 1:
        matrix[row][new_column] = 1
        return False
    else:
        return True


def forward(row, column, matrix):
    new_row = row - 1
    if matrix[new_row][column] != 1:
        matrix[new_row][column] = 1
        return False
    else:
        return True


def backward(row, column, matrix):
    new_row = row + 1
    if matrix[new_row][column] != 1:
        matrix[new_row][column] = 1
        return False
    else:
        return True


def assemble(row, column, matrix):
    marker = True
    if row == 0 and column == 0:
        while marker:
            temp_num = random.randint(1, 3)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
    elif row == 0 and column == 1:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = left(row, column, matrix)
    elif row == 0 and column == 2:
        while marker:
            temp_num = random.randint(1, 3)
            if temp_num == 1:
                marker = left(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
    elif row == 1 and column == 0:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = forward(row, column, matrix)
    elif row == 1 and column == 1:
        while marker:
            temp_num = random.randint(1, 5)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = forward(row, column, matrix)
            elif temp_num == 5:
                marker = left(row, column, matrix)

    elif row == 1 and column == 2:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = left(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = forward(row, column, matrix)

    elif row == 2 and column == 0:
        while marker:
            temp_num = random.randint(1, 3)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = forward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)

    elif row == 2 and column == 1:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = forward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = left(row, column, matrix)

    elif row == 2 and column == 2:
        while marker:
            temp_num = random.randint(1, 3)
            if temp_num == 1:
                marker = left(row, column, matrix)
            elif temp_num == 2:
                marker = forward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)

    elif row == 3 and column == 0:

        temp_num = random.randint(1, 5)
        if temp_num == 1:
            marker = right(row, column, matrix)
        elif temp_num == 2:
            marker = backward(row, column, matrix)
        elif temp_num == 3:
            marker = down(row, column, matrix)
        elif temp_num == 4:
            marker = left(row, column, matrix)
        elif temp_num == 5:
            marker = top(row, column, matrix)
    elif row == 3 and column == 2:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = left(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = top(row, column, matrix)



    elif row == 4 and column == 0:
        while marker:
            temp_num = random.randint(1, 5)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = forward(row, column, matrix)
            elif temp_num == 5:
                marker = top(row, column, matrix)


    elif row == 4 and column == 1:
        while marker:
            temp_num = random.randint(1, 6)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = forward(row, column, matrix)
            elif temp_num == 5:
                marker = left(row, column, matrix)
            elif temp_num == 6:
                marker = top(row, column, matrix)

    elif row == 4 and column == 2:
        while marker:
            temp_num = random.randint(1, 5)
            if temp_num == 1:
                marker = left(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = forward(row, column, matrix)
            elif temp_num == 5:
                marker = top(row, column, matrix)




    elif row == 5 and column == 0:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = forward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = top(row, column, matrix)


    elif row == 5 and column == 1:
        while marker:
            temp_num = random.randint(1, 5)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = forward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = left(row, column, matrix)
            elif temp_num == 5:
                marker = top(row, column, matrix)



    elif row == 5 and column == 2:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = left(row, column, matrix)
            elif temp_num == 2:
                marker = forward(row, column, matrix)
            elif temp_num == 3:
                marker = down(row, column, matrix)
            elif temp_num == 4:
                marker = top(row, column, matrix)

    if row == 6 and column == 0:
        while marker:
            temp_num = random.randint(1, 3)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = top(row, column, matrix)

    elif row == 6 and column == 1:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = top(row, column, matrix)
            elif temp_num == 4:
                marker = left(row, column, matrix)

    elif row == 6 and column == 2:
        while marker:
            temp_num = random.randint(1, 3)
            if temp_num == 1:
                marker = left(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = top(row, column, matrix)

    elif row == 7 and column == 0:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = top(row, column, matrix)
            elif temp_num == 4:
                marker = forward(row, column, matrix)

    elif row == 7 and column == 1:
        while marker:
            temp_num = random.randint(1, 5)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = top(row, column, matrix)
            elif temp_num == 4:
                marker = forward(row, column, matrix)
            elif temp_num == 5:
                marker = left(row, column, matrix)

    elif row == 7 and column == 2:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = left(row, column, matrix)
            elif temp_num == 2:
                marker = backward(row, column, matrix)
            elif temp_num == 3:
                marker = top(row, column, matrix)
            elif temp_num == 4:
                marker = forward(row, column, matrix)

    elif row == 8 and column == 0:
        while marker:
            temp_num = random.randint(1, 3)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = forward(row, column, matrix)
            elif temp_num == 3:
                marker = top(row, column, matrix)

    elif row == 8 and column == 1:
        while marker:
            temp_num = random.randint(1, 4)
            if temp_num == 1:
                marker = right(row, column, matrix)
            elif temp_num == 2:
                marker = forward(row, column, matrix)
            elif temp_num == 3:
                marker = top(row, column, matrix)
            elif temp_num == 4:
                marker = left(row, column, matrix)

    elif row == 8 and column == 2:
        while marker:
            temp_num = random.randint(1, 3)
            if temp_num == 1:
                marker = left(row, column, matrix)
            elif temp_num == 2:
                marker = forward(row, column, matrix)
            elif temp_num == 3:
                marker = top(row, column, matrix)


def bool_checker(new_row, new_column, matrix):
    if new_row == 0:
        if new_column == 0:
            if (matrix[0][1] == 1) and (matrix[1][0] == 1) and (matrix[3][0] == 1):
                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[0][0] == 1) and (matrix[1][1] == 1) and (matrix[0][2] == 1) and (matrix[3][1] == 1):
                return False
            else:
                return True

        elif new_column == 2:
            if (matrix[0][1] == 1) and (matrix[1][2] == 1) and (matrix[3][2] == 1):

                return False
            else:
                return True
    elif new_row == 1:
        if new_column == 0:
            if (matrix[0][0] == 1) and (matrix[1][1] == 1) and (matrix[2][0] == 1) and (matrix[4][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[1][0] == 1) and (matrix[1][2] == 1) and (matrix[0][1] == 1) and (matrix[2][1] == 1) and (
                    matrix[4][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[0][2] == 1) and (matrix[1][1] == 1) and (matrix[2][2] == 1) and (matrix[4][2] == 1):

                return False
            else:
                return True
    elif new_row == 2:
        if new_column == 0:
            if (matrix[1][0] == 1) and (matrix[2][1] == 1) and (matrix[5][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[2][0] == 1) and (matrix[2][2] == 1) and (matrix[1][1] == 1) and (matrix[5][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[2][1] == 1) and (matrix[1][2] == 1) and (matrix[5][2] == 1):

                return False
            else:
                return True
    if new_row == 3:
        if new_column == 0:
            if (matrix[0][0] == 1) and (matrix[3][1] == 1) and (matrix[4][0] == 1) and (matrix[6][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[3][0] == 1) and (matrix[3][2] == 1) and (matrix[0][1] == 1) and (matrix[6][1] == 1) and (
                    matrix[4][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[0][2] == 1) and (matrix[6][2] == 1) and (matrix[3][1] == 1) and (matrix[4][2] == 1):

                return False
            else:
                return True
    elif new_row == 4:
        if new_column == 0:
            if (matrix[4][1] == 1) and (matrix[3][0] == 1) and (matrix[5][0] == 1) and (matrix[1][0] == 1) and (
                    matrix[7][0] == 1):

                return False
            else:
                return True
        if new_column == 1:
            if (matrix[4][0] == 1) and (matrix[4][2] == 1) and (matrix[3][1] == 1) and (matrix[5][1] == 1) and (
                    matrix[1][1] == 1) and (matrix[7][1] == 1):

                return False
            else:
                return True
        if new_column == 2:
            if (matrix[1][2] == 1) and (matrix[4][1] == 1) and (matrix[3][2] == 1) and (matrix[5][2] == 1) and (
                    matrix[7][2] == 1):

                return False
            else:
                return True
    elif new_row == 5:
        if new_column == 0:
            if (matrix[2][0] == 1) and (matrix[5][1] == 1) and (matrix[4][0] == 1) and (matrix[8][0] == 1):

                return False
            else:
                return True
        if new_column == 1:
            if (matrix[5][0] == 1) and (matrix[5][2] == 1) and (matrix[4][1] == 1) and (matrix[2][1] == 1) and (
                    matrix[8][1] == 1):

                return False
            else:
                return True
        if new_column == 2:
            if (matrix[2][2] == 1) and (matrix[5][1] == 1) and (matrix[4][2] == 1) and (matrix[8][2] == 1):

                return False
            else:
                return True
    if new_row == 6:
        if new_column == 0:
            if (matrix[3][0] == 1) and (matrix[6][1] == 1) and (matrix[7][0] == 1):

                return False
            else:
                return True
        if new_column == 1:
            if (matrix[6][0] == 1) and (matrix[7][1] == 1) and (matrix[6][2] == 1) and (matrix[3][1] == 1):

                return False
            else:
                return True
        if new_column == 2:
            if (matrix[6][1] == 1) and (matrix[3][2] == 1) and (matrix[7][2] == 1):

                return False
            else:
                return True
    elif new_row == 7:
        if new_column == 0:
            if (matrix[6][0] == 1) and (matrix[7][1] == 1) and (matrix[8][0] == 1) and (matrix[4][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[7][0] == 1) and (matrix[7][2] == 1) and (matrix[6][1] == 1) and (matrix[8][1] == 1) and (
                    matrix[4][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[6][2] == 1) and (matrix[7][1] == 1) and (matrix[8][2] == 1) and (matrix[4][2] == 1):

                return False
            else:
                return True
    elif new_row == 8:
        if new_column == 0:
            if (matrix[7][0] == 1) and (matrix[8][1] == 1) and (matrix[5][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[8][0] == 1) and (matrix[8][2] == 1) and (matrix[7][1] == 1) and (matrix[5][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[8][1] == 1) and (matrix[7][2] == 1) and (matrix[5][2] == 1):

                return False
            else:
                return True


def bool_checker_answer(new_row, new_column, matrix):
    if new_row == 0:
        if new_column == 0:
            if (matrix[0][1] == 1) or (matrix[1][0] == 1) or (matrix[3][0] == 1):
                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[0][0] == 1) or (matrix[1][1] == 1) or (matrix[0][2] == 1) or (matrix[3][1] == 1):
                return False
            else:
                return True

        elif new_column == 2:
            if (matrix[0][1] == 1) or (matrix[1][2] == 1) or (matrix[3][2] == 1):

                return False
            else:
                return True
    elif new_row == 1:
        if new_column == 0:
            if (matrix[0][0] == 1) or (matrix[1][1] == 1) or (matrix[2][0] == 1) or (matrix[4][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[1][0] == 1) or (matrix[1][2] == 1) or (matrix[0][1] == 1) or (matrix[2][1] == 1) or (
                    matrix[4][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[0][2] == 1) or (matrix[1][1] == 1) or (matrix[2][2] == 1) or (matrix[4][2] == 1):

                return False
            else:
                return True
    elif new_row == 2:
        if new_column == 0:
            if (matrix[1][0] == 1) or (matrix[2][1] == 1) or (matrix[5][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[2][0] == 1) or (matrix[2][2] == 1) or (matrix[1][1] == 1) or (matrix[5][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[2][1] == 1) or (matrix[1][2] == 1) or (matrix[5][2] == 1):

                return False
            else:
                return True
    if new_row == 3:
        if new_column == 0:
            if (matrix[0][0] == 1) or (matrix[3][1] == 1) or (matrix[4][0] == 1) or (matrix[6][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[3][0] == 1) or (matrix[3][2] == 1) or (matrix[0][1] == 1) or (matrix[6][1] == 1) or (
                    matrix[4][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[0][2] == 1) or (matrix[6][2] == 1) or (matrix[3][1] == 1) or (matrix[4][2] == 1):

                return False
            else:
                return True
    elif new_row == 4:
        if new_column == 0:
            if (matrix[4][1] == 1) or (matrix[3][0] == 1) or (matrix[5][0] == 1) or (matrix[1][0] == 1) or (
                    matrix[7][0] == 1):

                return False
            else:
                return True
        if new_column == 1:
            if (matrix[4][0] == 1) or (matrix[4][2] == 1) or (matrix[3][1] == 1) or (matrix[5][1] == 1) or (
                    matrix[1][1] == 1) or (matrix[7][1] == 1):

                return False
            else:
                return True
        if new_column == 2:
            if (matrix[1][2] == 1) or (matrix[4][1] == 1) or (matrix[3][2] == 1) or (matrix[5][2] == 1) or (
                    matrix[7][2] == 1):

                return False
            else:
                return True
    elif new_row == 5:
        if new_column == 0:
            if (matrix[2][0] == 1) or (matrix[5][1] == 1) or (matrix[4][0] == 1) or (matrix[8][0] == 1):

                return False
            else:
                return True
        if new_column == 1:
            if (matrix[5][0] == 1) or (matrix[5][2] == 1) or (matrix[4][1] == 1) or (matrix[2][1] == 1) or (
                    matrix[8][1] == 1):

                return False
            else:
                return True
        if new_column == 2:
            if (matrix[2][2] == 1) or (matrix[5][1] == 1) or (matrix[4][2] == 1) or (matrix[8][2] == 1):

                return False
            else:
                return True
    if new_row == 6:
        if new_column == 0:
            if (matrix[3][0] == 1) or (matrix[6][1] == 1) or (matrix[7][0] == 1):
                return False
            else:
                return True
        if new_column == 1:
            if (matrix[6][0] == 1) or (matrix[7][1] == 1) or (matrix[6][2] == 1) or (matrix[3][1] == 1):
                return False
            else:
                return True
        if new_column == 2:
            if (matrix[6][1] == 1) or (matrix[3][2] == 1) or (matrix[7][2] == 1):

                return False
            else:
                return True
    elif new_row == 7:
        if new_column == 0:
            if (matrix[6][0] == 1) or (matrix[7][1] == 1) or (matrix[8][0] == 1) or (matrix[4][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[7][0] == 1) or (matrix[7][2] == 1) or (matrix[6][1] == 1) or (matrix[8][1] == 1) or (
                    matrix[4][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[6][2] == 1) or (matrix[7][1] == 1) or (matrix[8][2] == 1) or (matrix[4][2] == 1):

                return False
            else:
                return True
    elif new_row == 8:
        if new_column == 0:
            if (matrix[7][0] == 1) or (matrix[8][1] == 1) or (matrix[5][0] == 1):

                return False
            else:
                return True
        elif new_column == 1:
            if (matrix[8][0] == 1) or (matrix[8][2] == 1) or (matrix[7][1] == 1) or (matrix[5][1] == 1):

                return False
            else:
                return True
        elif new_column == 2:
            if (matrix[8][1] == 1) or (matrix[7][2] == 1) or (matrix[5][2] == 1):

                return False
            else:
                return True


def negative_sampling_transformation(some_choice, negative_list):
    negative_counter = 0
    while negative_counter < 3:
        new_choice = copy.deepcopy(some_choice)
        marker = True
        while marker:
            token = random.randint(1, 2)
            # randomly add or not add a cube
            if token == 1:
                wrong_location = random.randint(0, 26)
                wrong_row = wrong_location // 3
                wrong_column = wrong_location % 3
                if new_choice[wrong_row][wrong_column] == 0:
                    new_choice[wrong_row][wrong_column] = 1
                else:
                    new_choice[wrong_row][wrong_column] = 0


            elif token == 2:
                marker_6 = True
                wrong_location = random.randint(0, 26)
                wrong_row = wrong_location // 3
                wrong_column = wrong_location % 3
                if new_choice[wrong_row][wrong_column] == 0:
                    while marker_6:
                        new_location = random.randint(0, 26)
                        new_row = new_location // 3
                        new_column = new_location % 3
                        if new_choice[new_row][new_column] == 1:
                            marker_6 = False
                            new_choice[wrong_row][wrong_column] = 1
                            new_choice[new_row][new_column] = 0
                else:
                    while marker_6:
                        new_location = random.randint(0, 26)
                        new_row = new_location // 3
                        new_column = new_location % 3
                        if new_choice[new_row][new_column] == 0:
                            marker_6 = False
                            new_choice[wrong_row][wrong_column] = 0
                            new_choice[new_row][new_column] = 1
            if new_choice not in negative_list:
                negative_list.append(new_choice)
                marker = False
                negative_counter += 1


def negative_print_out(negative_choice):
    output = ""
    counter_6 = 1
    for element in negative_choice:
        for item in element:
            if counter_6 % 9 != 0:
                output = output + str(item) + ","
            else:
                output = output + str(item) + ";"
            counter_6 += 1
    output = output + "10"
    print(output)


def just_print_out(choice_1, new_list):
    output = ""
    counter_6 = 1
    for element in choice_1:
        for item in element:
            if counter_6 % 9 != 0:
                output = output + str(item) + ","
            else:
                output = output + str(item) + ";"
            counter_6 += 1
    output = output + "10"
    if output not in new_list:
        new_list.append(output)
        return True
    else:
        return False


# final_list is a list containing something
final_list = []
final = 0
while final <= 30500:
    cube = [[1, 1, 1], [1, 1, 1], [1, 1, 1],
            [1, 1, 1], [1, 1, 1], [1, 1, 1],
            [1, 1, 1], [1, 1, 1], [1, 1, 1]]

    choice_1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    choice_3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    cubes_1 = random.randint(1, 26)
    # cubes_3 is the answer cube
    cubes_3 = 27 - cubes_1
    # randomly select the starting cube
    first_location = random.randint(0, 26)
    first_row = first_location // 3
    first_column = first_location % 3
    choice_1[first_row][first_column] = 1

    # check status of cube_1
    if cubes_1 == 1:
        continue
    elif cubes_1 == 2:
        assemble(first_row, first_column, choice_1)
    else:
        position_list = [(first_row, first_column)]
        assemble(first_row, first_column, choice_1)
        counter_1 = 2
        for row in range(0, 9):
            for column in range(0, 3):
                if choice_1[row][column] == 1:
                    if (row, column) not in position_list:
                        position_list.append((row, column))

        while counter_1 < cubes_1:
            position = random.randint(0, len(position_list) - 1)
            temp_row = position_list[position][0]
            temp_column = position_list[position][1]
            tuple_1 = (temp_row, temp_column)
            # if the select thing is fully covered
            # if checker(tuple_1, position_list, choice_1) == False:
            if bool_checker(temp_row, temp_column, choice_1) == False:
                continue
            else:
                assemble(temp_row, temp_column, choice_1)
                counter_1 += 1
                for row in range(0, 9):
                    for column in range(0, 3):
                        if (row, column) not in position_list:
                            if choice_1[row][column] == 1:
                                position_list.append((row, column))
    if cubes_3 == 1:
        for row in range(0, 9):
            for column in range(0, 3):
                if choice_1[row][column] == 0:
                    choice_3[row][column] = 1
                    break
        if (just_print_out(choice_1, final_list) == True):
            if (just_print_out(choice_3, final_list) == True):
                negative_list = []
                negative_sampling_transformation(choice_3, negative_list)
                print(final_list[final])
                print(final_list[final + 1])
                for elements in negative_list:
                    negative_print_out(elements)
                print()
                final = final + 2

    else:
        # We first generate the answer cube, but we need to check whether it is correct or not
        for row in range(0, 9):
            for column in range(0, 3):
                if choice_1[row][column] == 0:
                    choice_3[row][column] = 1

        # Now we are checking whether the answer cube is connected or not
        marker_5 = True
        for row in range(0, 9):
            for column in range(0, 3):
                if bool_checker_answer(row, column, choice_3) == False:
                    continue
                else:
                    marker_5 = False
                    break
        if marker_5 == True:
            if (just_print_out(choice_1, final_list) == True):
                if (just_print_out(choice_3, final_list) == True):
                    negative_list = []
                    negative_sampling_transformation(choice_3, negative_list)
                    print(final_list[final])
                    print(final_list[final + 1])
                    for elements in negative_list:
                        negative_print_out(elements)
                    print()
                    final = final + 2
