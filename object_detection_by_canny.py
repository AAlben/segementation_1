import os
import cv2
import numpy as np


def take_point(X, start_i):
    step = int(X.shape[0] / 4)
    threshold = 0.01
    point = None

    interval = X[start_i * step:(start_i + 1) * step]
    while threshold < 0.05 and point is None:
        where = np.where(interval <= threshold)[0]
        if where.any():
            point = (start_i * step + where[0].item(), start_i * step + where[-1].item())
            break
        threshold += 0.01

    return point


def calculate_Y_interval(edges, K):
    Y = np.count_nonzero(edges, axis=1) / edges.shape[1]
    step_y = int(Y.shape[0] / 4 / 2)

    start_y_1 = take_point(Y[step_y:], 0)
    start_y_2 = take_point(Y, 0)
    if start_y_1:
        start_y = start_y_1[1] + step_y
    elif start_y_2:
        start_y = start_y_2[1]
    else:
        start_y = 0

    end_y = take_point(Y, K - 1)
    if end_y:
        end_y = end_y[0]
    else:
        end_y = Y.shape[0]

    return (start_y, end_y)


def calculate_X_interval(edges, K):
    states_1, states_2 = [], []
    X = np.count_nonzero(edges, axis=0) / edges.shape[0]
    step = int(X.shape[0] / K)

    for i in range(K):
        interval = X[i * step:(i + 1) * step]
        max_ = np.max(interval)
        mean_ = np.mean(interval)
        min_ = np.min(interval)
        std_ = np.std(interval)

        if max_ >= 0.15 and mean_ >= 0.1:
            states_1.append('has cow')
            states_2.append(1)
        elif mean_ <= 0.1 and min_ <= 0.03:
            states_1.append('has gradient')
            states_2.append(0)
        else:
            states_1.append('else')
            states_2.append(-1)

    x_result = statistics_states(X, states_2, K)
    return x_result


def statistics_states(X, states, K):
    start, end = None, None
    pass_states = [[1, 1, 1, 1],
                   [0, 0, 0, 0],
                   [0, 1, 0, 1],
                   [1, 0, 1, 0]]
    if states in pass_states:
        return None

    states = np.array(states)
    has_zero = np.where(states == 0)[0].shape[0] > 0
    has_else = np.where(states == -1)[0].shape[0] > 0
    if has_else:
        if not has_zero:
            states[states == -1] = 0
        else:
            states[states == -1] = 1
    if np.where(states == 0)[0].shape[0] >= 3:
        return None
    states = states.tolist()

    if states in [[0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]]:
        start, end = take_point(X, 0), take_point(X, 3)
        if start is None and end:
            start, end = 0, end[0]
        elif start and end is None:
            start, end = start[1], X.shape[0]
        elif start is None and end is None:
            return None
        else:
            start, end = start[1], end[0]
    elif states in [[1, 0, 1, 1], [0, 0, 1, 1]]:
        start, end = take_point(X, 1), take_point(X, 3)
        start = start[1] if start else int(X.shape[0] / K)
        end = end[0] if end else X.shape[0]
    elif states == [1, 1, 0, 1]:
        start, end = take_point(X, 0), take_point(X, 2)
        start = start[0] if start else 0
        end = end[0] if end else 2 * int(X.shape[0] / K)
    elif states == [1, 0, 0, 1]:
        start, end = take_point(X, 1), take_point(X, 2)
        if not start or not end:
            return None
        start, end = start[0], end[1]
        if start > X.shape[0] - end:
            start, end = 0, start
        else:
            start, end = end, X.shape[0]
    elif states == [1, 1, 0, 0]:
        end = take_point(X, 2)
        if not end:
            return None
        start, end = 0, end[0]
    else:
        return None

    return (start, end)


def process(img, edges):
    '''--
        1）不完整靠前情况：mean低；min也很低；std较大 | 能区别出来前边有牛，后边没有
        2）完整牛 - max没有用；主要看mean、min、std看变化波动
        3）mean < 0.1; min <= 0.05; std >= 0.04
        4）还会出现前后两头牛，要哪一个的问题
    '''

    K = 4

    (start_y, end_y) = calculate_Y_interval(edges, K)
    x_interval = calculate_X_interval(edges, K)

    if not x_interval:
        return None
    else:
        return (start_y, end_y, x_interval[0], x_interval[1])


def execute(img):
    edges = cv2.Canny(img, 100, 200)
    points = process(img, edges)
    return points
