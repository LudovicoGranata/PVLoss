from torch import nn
import torch


class PVLoss(nn.Module):
    """Physical Violation Loss"""

    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.unit_vectors = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).to(self.device)
        self.axis_combinations = [[i, j] for i in range(
            len(self.unit_vectors)) for j in range(len(self.unit_vectors)) if j >= i]

    def forward(self, bb, rotations, translations):
        """
        bb: [B, 6] -> 3D bounding boxes of the objects as (x,y,z,w,h,d) where x,y,z are the coordinates of the top-left-bottom corner, w is the width, h is the height and d is the depth
        rotations: [B, 3, 3] -> 3x3 matrix expressing rotations
        translations: [B, 1, 3] -> 1x3 matrix expressing translations

        """

        # bb_corners dimensions :  (b, 8, 3)
        bb_corners = torch.Tensor(
            [[
                [bb[i][0], bb[i][1], bb[i][2]],
                [bb[i][0], bb[i][1], bb[i][2] + bb[i][5]],
                [bb[i][0], bb[i][1] + bb[i][4], bb[i][2]],
                [bb[i][0], bb[i][1] + bb[i][4], bb[i][2] + bb[i][5]],
                [bb[i][0] + bb[i][3], bb[i][1], bb[i][2]],
                [bb[i][0] + bb[i][3], bb[i][1], bb[i][2] + bb[i][5]],
                [bb[i][0] + bb[i][3], bb[i][1] + bb[i][4], bb[i][2]],
                [bb[i][0] + bb[i][3], bb[i][1] + bb[i][4], bb[i][2] + bb[i][5]],
            ] for i in range(len(bb))]
        ).to(self.device)
        
        bb_corners_transformed = torch.matmul(bb_corners, rotations)  # (b,8,3)
        bb_corners_transformed = torch.add(
            bb_corners_transformed, translations)  # (b, 8, 3)

        unit_vector_transformed = torch.matmul(
            self.unit_vectors, rotations)  # (b,3,3)

        n_obj = len(bb)
        combinations = [[i, j]
                        for i in range(n_obj) for j in range(n_obj) if j > i]
        total_intersection = 0
        for first, second in combinations:
            intersection = []
            intersect = True
            corners_first = bb_corners_transformed[first]  # (8,3)
            corners_second = bb_corners_transformed[second]  # (8,3)

            unit_vector_first = unit_vector_transformed[first]  # (3,3)
            unit_vector_second = unit_vector_transformed[second]  # (3,3)

            # test unit vector axis
            for i in range(len(unit_vector_first)):
                axis = unit_vector_first[i]  # size 3
                intersection_i, intersect = self.intersected_projection(
                    corners_first, corners_second, axis)
                intersection.append(intersection_i)
                if intersect == False:
                    break
            if intersect == False:
                continue

            for i in range(len(unit_vector_second)):
                axis = unit_vector_first[i]
                intersection_i, intersect = self.intersected_projection(
                    corners_first, corners_second, axis)
                intersection.append(intersection_i)
                if intersect == False:
                    break
            if intersect == False:
                continue

            # test dot product of unit vector as axis

            for first_axis_i, second_axis_i in self.axis_combinations:
                axis = torch.cross(
                    unit_vector_transformed[first, first_axis_i], unit_vector_transformed[second, second_axis_i])
                # handle the case of parallel axis
                if torch.equal(axis, torch.Tensor([0, 0, 0]).to(self.device)):
                    continue
                intersection_i, intersect = self.intersected_projection(
                    corners_first, corners_second, axis)
                intersection.append(intersection_i)
                if intersect == False:
                    break
            if intersect == False:
                continue

            total_intersection += min(intersection)

        return total_intersection

    def intersected_projection(self, corners_first, corners_second, axis):

        # distance of the projected points
        dist = [torch.dot(corners_first[i], axis) for i in range(8)]
        first_min = min(dist)
        first_max = max(dist)

        dist = [torch.dot(corners_second[i], axis) for i in range(8)]
        second_min = min(dist)
        second_max = max(dist)

        # the total possible length of the projection with no intersection
        total_length = max(first_max, second_max) - min(first_min, second_min)
        # the sum of the two projections
        sum_length = first_max - first_min + second_max - second_min

        return (sum_length - total_length, True) if total_length < sum_length else (0, False)

    # for testing

    def plot_box(self, bb1, bb2, rotations, translations):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        bb_corners1 = [
            [bb1[0], bb1[1], bb1[2]],
            [bb1[0] + bb1[3], bb1[1], bb1[2]],
            [bb1[0], bb1[1], bb1[2] + bb1[5]],
            [bb1[0], bb1[1] + bb1[4], bb1[2]],
        ]

        bb_corners2 = [
            [bb2[0], bb2[1], bb2[2]],
            [bb2[0] + bb2[3], bb2[1], bb2[2]],
            [bb2[0], bb2[1], bb2[2] + bb2[5]],
            [bb2[0], bb2[1] + bb2[4], bb2[2]],
        ]

        bb_corners1 = np.matmul(bb_corners1, rotations[0])
        bb_corners1 = np.add(bb_corners1, translations[0])

        bb_corners2 = np.matmul(bb_corners2, rotations[1])
        bb_corners2 = np.add(bb_corners2, translations[1])

        cube_definition_array1 = [
            np.array(list(item))
            for item in bb_corners1
        ]

        points1 = []
        points1 += cube_definition_array1
        vectors1 = [
            cube_definition_array1[1] - cube_definition_array1[0],
            cube_definition_array1[2] - cube_definition_array1[0],
            cube_definition_array1[3] - cube_definition_array1[0]
        ]

        points1 += [cube_definition_array1[0] + vectors1[0] + vectors1[1]]
        points1 += [cube_definition_array1[0] + vectors1[0] + vectors1[2]]
        points1 += [cube_definition_array1[0] + vectors1[1] + vectors1[2]]
        points1 += [cube_definition_array1[0] +
                    vectors1[0] + vectors1[1] + vectors1[2]]

        points1 = np.array(points1)

        edges1 = [
            [points1[0], points1[3], points1[5], points1[1]],
            [points1[1], points1[5], points1[7], points1[4]],
            [points1[4], points1[2], points1[6], points1[7]],
            [points1[2], points1[6], points1[3], points1[0]],
            [points1[0], points1[2], points1[4], points1[1]],
            [points1[3], points1[6], points1[7], points1[5]]
        ]


        cube_definition_array2 = [
            np.array(list(item))
            for item in bb_corners2
        ]

        points2 = []
        points2 += cube_definition_array2
        vectors2 = [
            cube_definition_array2[1] - cube_definition_array2[0],
            cube_definition_array2[2] - cube_definition_array2[0],
            cube_definition_array2[3] - cube_definition_array2[0]
        ]

        points2 += [cube_definition_array2[0] + vectors2[0] + vectors2[1]]
        points2 += [cube_definition_array2[0] + vectors2[0] + vectors2[2]]
        points2 += [cube_definition_array2[0] + vectors2[1] + vectors2[2]]
        points2 += [cube_definition_array2[0] +
                    vectors2[0] + vectors2[1] + vectors2[2]]

        points2 = np.array(points2)

        edges2 = [
            [points2[0], points2[3], points2[5], points2[1]],
            [points2[1], points2[5], points2[7], points2[4]],
            [points2[4], points2[2], points2[6], points2[7]],
            [points2[2], points2[6], points2[3], points2[0]],
            [points2[0], points2[2], points2[4], points2[1]],
            [points2[3], points2[6], points2[7], points2[5]]
        ]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        faces1 = Poly3DCollection(edges1, linewidths=1, edgecolors='k')
        faces1.set_facecolor((0, 0, 1, 0.1))

        faces2 = Poly3DCollection(edges2, linewidths=1, edgecolors='k')
        faces2.set_facecolor((0, 0, 1, 0.1))

        ax.add_collection3d(faces1)
        ax.add_collection3d(faces2)

        ax.set_xlim(200)
        ax.set_ylim(200)
        ax.set_zlim(200)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation
    from timeit import default_timer as timer

    pv_loss = PVLoss()

    # tests
    # I need a torch list with bb= (x,y,z,w,h,d)
    # I need a torch list with rotations
    # I need a torch list with translation
    n_test = 10
    for _ in range(n_test):
        rotation1 = torch.Tensor(Rotation.random().as_matrix())
        rotation2 = torch.Tensor(Rotation.random().as_matrix())

        trans_range = 20
        translation1 = -trans_range * torch.rand(1, 3) + trans_range
        translation2 = -trans_range * torch.rand(1, 3) + trans_range

        rotations = torch.stack((rotation1, rotation2))
        translations = torch.stack((translation1, translation2))

        first_bb = torch.Tensor([0, 0, 0, 100, 100, 100])
        second_bb = torch.Tensor([0, 0, 0, 100, 100, 100])
        bb = torch.stack((first_bb, second_bb))

        start = timer()
        loss = pv_loss(bb, rotations, translations)
        end = timer()
        print("The loss is: " + str(loss) +
              "   computation time: " + str(end-start) + " seconds")

        pv_loss.plot_box(first_bb, second_bb, rotations, translations)