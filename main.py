from PIL import Image
import numpy as np

def line1(x0, y0, x1, y1, image, color=(255, 255, 255)):
    t = np.arange(0, 1, 0.1)
    x, y = np.round((x0 * (1 - t) + x1 * t, y0 * (1 - t) + y1 * t)).astype(int)
    image[x, y] = color
    return image


def line2(x0, y0, x1, y1, image, color=(255, 255, 255)):
    for x in range(min(x0, x1), max(x0, x1) + 1):
        y = int(y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 - x0 != 0 else y0)
        image[x][y] = color
    return image


def line3(x0, y0, x1, y1, image, color=(255, 255, 255)):
    steep = abs(x0 - x1) < abs(y0 - y1)
    if steep:
        x0, y0, x1, y1 = y0, x0, y1, x1
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0
    t = np.linspace(0, 1, x1 - x0 + 1)
    y = np.round(y0 * (1 - t) + y1 * t).astype(int)
    if steep:
        image[y, np.arange(x0, x1 + 1)] = color
    else:
        image[np.arange(x0, x1 + 1), y] = color
    return image


def line4(x0, y0, x1, y1, image, color=(255, 255, 255)):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0, x1, y1 = y0, x0, y1, x1
        steep = True
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0:
        return image
    d_error = abs(dy / dx)
    error = 0
    y = y0
    for x in range(x0, x1 + 1):
        if steep:
            image[y][x] = color
        else:
            image[x][y] = color
        error += d_error
        if error > 0.5:
            y += 1 if y1 > y0 else -1
            error -= 1
    return image


def task0(color=(0, 0, 0)):
    listRGB = np.zeros((500, 500, 3), np.uint8)
    listRGB[:] = color
    Image.fromarray(listRGB).show()


def task1():
    H, W = 500, 500
    listRGB = (np.indices((H, W))[0] + np.indices((H, W))[1])
    Image.fromarray(listRGB.astype(np.uint8)).show()


def task2():
    H, W = 200, 200
    listRGB = np.zeros((H, W, 3), np.uint8)
    for i in range(0, 13):
        listRGB = line4(100, 100, int(100 + 95 * np.cos(2 * np.pi * i / 13)),
                        int(100 + 95 * np.sin(2 * np.pi * i / 13)), listRGB)
    Image.fromarray(listRGB).show()


def read_vertexes(filename: str):
    with open(filename, "r") as input_file:
        return [(float(x), float(y), float(z)) for line in input_file if line.startswith("v ")
                for x, y, z in [line.split()[1:4]]]


def read_polygons(filename: str):
    with open(filename, "r") as input_file:
        return [(int(x.split("/")[0]) - 1, int(y.split("/")[0]) - 1, int(z.split("/")[0]) - 1)
                for line in input_file if line.startswith("f ") for x, y, z in [line.split()[1:4]]]


def read_textures_vertexes(filename: str):
    with open(filename, "r") as input_file:
        return [(float(x), float(y)) for line in input_file if line.startswith("vt ")
                for x, y in [line.split()[1:3]]]


def read_textures_polygons(filename: str):
    with open(filename, "r") as input_file:
        return [(int(x.split("/")[1]) - 1, int(y.split("/")[1]) - 1, int(z.split("/")[1]) - 1)
                for line in input_file if line.startswith("f ") for x, y, z in [line.split()[1:4]]]


def read_normal_vertexes(filename: str):
    with open(filename, "r") as input_file:
        return [(float(x), float(y), float(z)) for line in input_file if line.startswith("vn ")
                for x, y, z in [line.split()[1:4]]]


def read_normal_polygons(filename: str):
    with open(filename, "r") as input_file:
        return [(int(x.split("/")[2]) - 1, int(y.split("/")[2]) - 1, int(z.split("/")[2]) - 1)
                for line in input_file if line.startswith("f ") for x, y, z in [line.split()[1:4]]]


def get_barycentric_coord(x0, y0, x1, y1, x2, y2, x, y):
    try:
        lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
        lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
        lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
        return [lambda0, lambda1, lambda2]
    except:
        return [-1, -1, -1]


def integer_points_in_triangle(xyz_values):
    points = []
    x0, y0, z0, x1, y1, z1, x2, y2, z2 = [xyz_values[i][j] for i in range(3) for j in range(3)]
    for x in range(min(x0, x1, x2), max(x0, x1, x2) + 1):
        for y in range(min(y0, y1, y2), max(y0, y1, y2) + 1):
            l = get_barycentric_coord(x0, y0, x1, y1, x2, y2, x, y)
            if min(l) >= 0:
                points.append((x, y, sum([[z0, z1, z2][i] * l[i] for i in range(3)]), l[0], l[1], l[2]))
    return points


def angle_to_point(triangle_vertices, point):
    vectors = [point - vertex for vertex in triangle_vertices]
    lengths = [np.linalg.norm(vector) for vector in vectors]
    cos_angle = np.dot(vectors[0], vectors[1]) / (lengths[0] * lengths[1])
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def cos_3d(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    mag_v1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    mag_v2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    cosine = dot_product / (mag_v1 * mag_v2)
    return cosine


class Object:
    def __init__(self, filename: str, H: int, W: int):
        self.vertexes = read_vertexes(filename)
        self.polygons = read_polygons(filename)
        self.textures_vertexes = read_textures_vertexes(filename)
        self.textures_polygons = read_textures_polygons(filename)
        self.normal_vertexes = read_normal_vertexes(filename)
        self.normal_polygons = read_normal_polygons(filename)
        self.H = H
        self.W = W

    def show_vertexes(self):
        listRGB = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        for x, y, _ in self.vertexes:
            listRGB[int(-ax * y + u0)][int(ay * x + v0)] = (255, 255, 255)
        new_img = Image.fromarray(listRGB)
        new_img.show()

    def show_edges(self):
        listRGB = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        for p in self.polygons:
            for i in range(3):
                j = (i + 1) % 3
                line4(int(-ax * self.vertexes[p[i]][1] + u0), int(ay * self.vertexes[p[i]][0] + v0),
                      int(-ax * self.vertexes[p[j]][1] + u0), int(ay * self.vertexes[p[j]][0] + v0), listRGB)
        new_img = Image.fromarray(listRGB)
        new_img.show()

    def show_polygons(self, t, ax, ay, u0, v0, textures, alpha, betta, gamma):
        buffer = np.zeros((self.H, self.W, 2))
        listRGB = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        R = np.matmul(
            np.matmul(np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]]),
                      np.array([[np.cos(betta), 0, np.sin(betta)], [0, 1, 0], [-np.sin(betta), 0, np.cos(betta)]])),
            np.array([[np.cos(gamma), np.sin(gamma), 0], [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]))

        for index, polygon in enumerate(self.polygons):

            K = np.array([[ay, 0, v0], [0, -ax, u0], [0, 0, 1]])

            XYZ = [np.array([self.vertexes[polygon[i]][0], self.vertexes[polygon[i]][1], self.vertexes[polygon[i]][2]])
                   for i in range(3)]
            XYZ = [np.matmul(R, XYZ[i]) for i in range(3)]

            XYZ = [np.array([-XYZ[i][1] + t[1], -XYZ[i][0] + t[0], (XYZ[i][2]) * t[2]]) for i in range(3)]

            xyz_values = np.array([[int(np.matmul(XYZ[i], K)[j]) for j in range(3)] for i in range(3)])

            l = [1, 0, 0]
            n0 = self.normal_vertexes[self.normal_polygons[index][0]]
            l0 = cos_3d(n0, l)
            n1 = self.normal_vertexes[self.normal_polygons[index][1]]
            l1 = cos_3d(n1, l)
            n2 = self.normal_vertexes[self.normal_polygons[index][2]]
            l2 = cos_3d(n2, l)

            points = integer_points_in_triangle(xyz_values)
            for point in points:
                cos = point[3] * l0 + point[4] * l1 + point[5] * l2

                texture = textures.load()
                size0, size1 = textures.size
                Txy = [int((sum([self.textures_vertexes[self.textures_polygons[index][i]][0] * point[i + 3]
                                 for i in range(3)])) * size0),
                       -int((sum([self.textures_vertexes[self.textures_polygons[index][i]][1] * point[i + 3]
                                  for i in range(3)])) * size1 - size1)]
                color = tuple([int(-val * cos if -val * cos < 255 else 255)
                               for val in texture[Txy[0], Txy[1]][:3]])

                if buffer[point[0]][point[1]][1] == 0 or buffer[point[0]][point[1]][0] < point[2]:
                    buffer[point[0]][point[1]] = [point[2], 1]
                    if cos < 0:
                        listRGB[point[0]][point[1]] = color
                    elif buffer[point[0]][point[1]][0] < point[2]:
                        listRGB[point[0]][point[1]] = (0, 0, 0)

        new_img = Image.fromarray(listRGB)
        new_img.show()


alpha, betta, gamma = 0, np.pi / 2, 0
t, ax, ay, u0, v0, file, textures = [-0.0625, 0.1125, 8000], 8000, 8000, 900, 500, 'model_1.obj', Image.open("zayac.jpg")
obj = Object(file, 1000, 1000)
obj.show_polygons(t, ax, ay, u0, v0, textures, alpha, betta, gamma)
