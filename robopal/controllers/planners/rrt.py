import math
import logging
import random
import copy
from typing import Union, List
from dataclasses import dataclass, field

import mujoco
import numpy as np

try:
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except Exception as e:
    logging.warn("Could not find plot libs.")


@dataclass
class Node:
    """ 节点 """
    x: float  # 节点坐标
    y: float
    z: float
    cost: float = 0.0
    path_x: list[float] = field(default_factory=list)  # 路径，作为画图的数据
    path_y: list[float] = field(default_factory=list)
    path_z: list[float] = field(default_factory=list)
    parent: None = None  # 父节点


class AreaBounds:
    """ 规划区域大小 """

    def __init__(self, area):
        self.xmin = float(area[0])
        self.xmax = float(area[1])
        self.ymin = float(area[2])
        self.ymax = float(area[3])
        self.zmin = float(area[4])
        self.zmax = float(area[5])


class RRT:
    def __init__(self,
                 start,
                 goal,
                 expand_dis=0.02,  # 树枝长度
                 goal_sample_rate=10,
                 max_iter=5000,
                 play_area=None,
                 sim=None
                 ):
        """ RRT*

        Parameters
        start:起点 [x,y]
        goal:目标点 [x,y]
        play_area: 约束随机树的范围 [xmin,xmax,ymin,ymax,zmin,zmax]
        expand_dis: 扩展的步长
        goal_sample_rate: 采样目标点的概率，百分制.default: 5,即表示5%的概率直接采样目标点
        """
        self.start = Node(start[0], start[1], start[2])
        self.end = Node(goal[0], goal[1], goal[2])

        if play_area is not None:
            self.play_area = AreaBounds(play_area)  # 树枝生长区域，左下(-2,0)==>右上(12,14)
        else:
            self.play_area = None  # 数值无限生长

        self.expand_dis = expand_dis  # 树枝一次的生长长度
        self.goal_sample_rate = goal_sample_rate  # 多少概率选择终点
        self.max_iter = max_iter  # 最大迭代次数
        self.node_list = []  # 保存节点

        self.sim = sim

    def planning(self, animation=True):
        """ RRT* path planning

        animation: 是否展示动画
        """
        # 将点作为根节点x_{init}，加入到随机树的节点集合中。
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # 从可行区域内随机选取一个节点x_{rand}
            rnd_node = self.sample_free()

            # 已生成的树中利用欧氏距离判断距离x_{rand}最近的点x_{near}。
            # 从已知节点中选择和目标节点最近的节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)  # 最接近的节点的索引
            nearest_node = self.node_list[nearest_ind]  # 获取该最近已知节点的坐标

            # 从x_{near} 与 x_{rand} 的连线方向上扩展固定步长 u，得到新节点 x_{new}
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 如果在可行区域内，且x_{near}与x_{new}之间无障碍物
            # 判断新点是否在规定的树的生长区域内，新点和最近点之间是否存在障碍物,都满足才保存该点作为树节点
            if self.is_inside_play_area(new_node, self.play_area) and not self.is_collide(self.sim, new_node):
                # 得到范围中点的索引
                nearInds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, nearInds)

                self.node_list.append(new_node)

                self.rewire(new_node, nearInds)
            # 如果此时得到的节点x_new到目标点的距离小于扩展步长，则直接将目标点作为x_rand。
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y,
                                      self.node_list[-1].z) <= self.expand_dis:
                # 以新点为起点，向终点画树枝
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                # 如果最新点和终点之间没有障碍物True，返回最终路径
                if not self.is_collide(self.sim, final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 10 == 0:
                self.draw_graph(rnd_node)
        return None

    # 距离最近的已知节点坐标，随机坐标，从已知节点向随机节点的延展的长度
    def steer(self, from_node, to_node, extend_lengh=float("inf")):
        # d已知点和随机点之间的距离，theta两个点之间的夹角
        d, theta_xy, theta_z = self.calc_distance_and_angle(from_node, to_node)

        # 如果$x_{near}$与$x_{rand}$间的距离小于步长，则直接将$x_{rand}$作为新节点$x_{new}$
        if extend_lengh >= d:  # 如果树枝的生长长度超出随机点，就用随机点位置作为新的节点
            new_x = to_node.x
            new_y = to_node.y
            new_z = to_node.z
        else:
            new_x = from_node.x + math.cos(theta_xy) * extend_lengh  # 最近点 x + cos * extend_len
            new_y = from_node.y + math.sin(theta_xy) * extend_lengh  # 最近点 y + sin * extend_len
            new_z = from_node.z + math.sin(theta_z) * extend_lengh  # 最近点 z + sin * extend_len

        new_node = Node(new_x, new_y, new_z)
        new_node.path_x = [from_node.x]  # 最近点
        new_node.path_y = [from_node.y]
        new_node.path_z = [from_node.z]

        new_node.path_x.append(new_x)
        new_node.path_y.append(new_y)
        new_node.path_z.append(new_z)

        new_node.cost += self.expand_dis
        new_node.parent = from_node  # 根节点变成最近点，用来指明方向
        # 将根节点进行更新，将将更节点修改为最近点，用来指明方向
        return new_node

    def generate_final_course(self, lastIndex):
        path = [[self.end.x, self.end.y, self.end.z]]

        while lastIndex is not None and isinstance(lastIndex, int) and 0 <= lastIndex < len(self.node_list):
            node = self.node_list[lastIndex]
            path.append([node.x, node.y, node.z])
            lastIndex = node.parent

        path.append([self.start.x, self.start.y, self.start.z])
        return path

    def sample_free(self):
        """ 以 100-goal_sample_rate %的概率随机生长，(goal_sample_rate)%的概率朝向目标点生长
        """
        if random.randint(0, 100) > self.goal_sample_rate:  # 大于p%就不选终点方向作为下个节点
            rnd = Node(
                random.uniform(self.play_area.xmin, self.play_area.xmax),  # 在树枝生长区域中随机取一个点
                random.uniform(self.play_area.ymin, self.play_area.ymax),
                random.uniform(self.play_area.zmin, self.play_area.zmax))
        else:
            rnd = Node(self.end.x, self.end.y, self.end.z)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # use the key of esc to stop simulation
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit() if event.key == 'escape' else None])
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        # 画随机点
        if rnd is not None:
            ax.scatter(rnd.x, rnd.y, rnd.z, marker='^', color='k')

        # Drew generated tree in 3D
        for node in self.node_list:
            if node.parent:
                ax.plot(node.path_x, node.path_y, node.path_z, color="green")

        # if a feasible area is defined,drew it in 3D
        if self.play_area is not None:
            # 绘制正方形框
            vertices = [
                (self.play_area.xmin, self.play_area.ymin, self.play_area.zmin),
                (self.play_area.xmax, self.play_area.ymin, self.play_area.zmin),
                (self.play_area.xmax, self.play_area.ymax, self.play_area.zmin),
                (self.play_area.xmin, self.play_area.ymax, self.play_area.zmin),
                (self.play_area.xmin, self.play_area.ymin, self.play_area.zmax),
                (self.play_area.xmax, self.play_area.ymin, self.play_area.zmax),
                (self.play_area.xmax, self.play_area.ymax, self.play_area.zmax),
                (self.play_area.xmin, self.play_area.ymax, self.play_area.zmax),
            ]
            # 定义顶点连接
            edges = [
                [vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]],
                [vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]],
                [vertices[0], vertices[4]],
                [vertices[1], vertices[5]],
                [vertices[2], vertices[6]],
                [vertices[3], vertices[7]],
            ]

            # 绘制线
            for edge in edges:
                x, y, z = zip(*edge)
                ax.plot(x, y, z, color='black')

            # 绘制透明的正方体表面
            ax.add_collection3d(
                Poly3DCollection([edges[0]], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1))
            ax.add_collection3d(
                Poly3DCollection([edges[1]], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1))

        ax.scatter(self.start.x, self.start.y, self.start.z, color='b', marker='x')
        ax.scatter(self.end.x, self.end.y, self.end.z, color='b', marker='x')

        ax.set_xlim([self.play_area.xmin, self.play_area.xmax])
        ax.set_ylim([self.play_area.ymin, self.play_area.ymax])
        ax.set_zlim([self.play_area.zmin, self.play_area.zmax])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.grid(True)
        plt.pause(0.01)

    def choose_parent(self, newNode, nearInds):
        if len(nearInds) == 0:
            return newNode
        dList = []
        for i in nearInds:

            dx = newNode.x - self.node_list[i].x
            dy = newNode.y - self.node_list[i].y
            dz = newNode.z - self.node_list[i].z

            d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            theta_xy = math.atan2(dy, dx)
            theta_z = math.atan2(dz, math.sqrt(dx ** 2 + dy ** 2))

            if self.check_collision(self.node_list[i], theta_xy, theta_z, d):
                dList.append(self.node_list[i].cost + d)
            else:
                dList.append(float('inf'))

        minCost = min(dList)
        minInd = nearInds[dList.index(minCost)]

        if minCost == float('inf'):
            logging.info("min cost is inf")
            return newNode

        newNode.cost = minCost
        newNode.parent = minInd

        return newNode

    def find_near_nodes(self, newNode):
        n_node = len(self.node_list)
        r = 0.3 * math.sqrt((math.log(n_node)) / n_node)
        d_list = [(node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2 + (node.z - newNode.z) ** 2
                  for node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        # 计算所有已知节点和随机节点之间的距离
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 + (node.z - rnd_node.z) ** 2
                 for node in node_list]
        # 获得距离最小的节点的索引
        minind = dlist.index(min(dlist))

        return minind

    # 判断选择的点是否在可行域
    @staticmethod
    def is_inside_play_area(node, play_area):
        if play_area is None:
            return True  # 如果没有定义可行区域，那么任何位置都是合适的

        if node.x < play_area.xmin or node.x > play_area.xmax or \
                node.y < play_area.ymin or node.y > play_area.ymax or \
                node.z < play_area.zmin or node.z > play_area.zmax:
            return False  # 如果节点的 x、y 或 z 坐标在可行区域外，返回 False（不合适）
        else:
            return True  # 如果节点的 x、y 和 z 坐标在可行区域内，返回 True（合适）

    def check_segment_collision(self, x1, y1, z1, x2, y2, z2):
        theta_xy = math.atan2(y2 - y1, x2 - x1)
        theta_z = math.atan2(z2 - z1, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        if distance % self.expand_dis == 0:
            n = distance / self.expand_dis
            for i in range(int(n)):
                ox = x1 + self.expand_dis * math.cos(theta_z) * math.cos(theta_xy) * i
                oy = y1 + self.expand_dis * math.cos(theta_z) * math.sin(theta_xy) * i
                oz = z1 + self.expand_dis * math.sin(theta_z) * i
                temNode = Node(ox, oy, oz)
        else:
            n = distance / self.expand_dis
            if n < 1:
                n = 1
            for i in range(int(n)):
                ox = x1 + self.expand_dis * math.cos(theta_z) * math.cos(theta_xy) * i
                oy = y1 + self.expand_dis * math.cos(theta_z) * math.sin(theta_xy) * i
                oz = z1 + self.expand_dis * math.sin(theta_z) * i
                temNode = Node(ox, oy, oz)
            temNode.path_x.append(x2)
            temNode.path_y.append(y2)
            temNode.path_z.append(z2)
        collision = not self.is_collide(self.sim, temNode)
        return collision

    def check_collision(self, nearNode, theta_xy, theta_z, d):
        tmpNode = copy.deepcopy(nearNode)

        end_x = tmpNode.x + math.cos(theta_xy) * math.cos(theta_z) * d
        end_y = tmpNode.y + math.sin(theta_xy) * math.cos(theta_z) * d
        end_z = tmpNode.z + math.sin(theta_z) * d
        collision = self.check_segment_collision(tmpNode.x, tmpNode.y, tmpNode.z, end_x, end_y, end_z)
        return collision

    def rewire(self, newNode, nearInds):
        for i in nearInds:
            nearNode = self.node_list[i]

            d = math.sqrt((nearNode.x - newNode.x) ** 2
                          + (nearNode.y - newNode.y) ** 2
                          + (nearNode.z - newNode.z) ** 2)
            s_cost = newNode.cost + d
            if nearNode.cost > s_cost:
                theta_xy = math.atan2(newNode.y - nearNode.y, newNode.x - nearNode.x)
                theta_z = math.atan2(newNode.z - nearNode.z,
                                     math.sqrt((newNode.x - nearNode.x) ** 2 + (newNode.y - nearNode.y) ** 2))

                if self.check_collision(nearNode, theta_xy, theta_z, d):
                    self.node_list[i].parent = newNode.parent
                    self.node_list[i].cost = s_cost
        return newNode

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """计算两个节点间的距离和方位角
        Args:
            from_node: 起始节点
            to_node: 目标节点
        Returns:
            d: 两节点之间的直线距离
            theta_xy: 从起始节点到目标节点的水平方位角（弧度）
            theta_z: 从起始节点到目标节点的垂直方位角（弧度）
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # 计算两节点之间的直线距离
        theta_xy = math.atan2(dy, dx)  # 计算水平方位角的弧度值
        theta_z = math.atan2(dz, math.sqrt(dx ** 2 + dy ** 2))  # 计算垂直方位角的弧度值
        return d, theta_xy, theta_z

    def calc_dist_to_goal(self, x, y, z):
        """计算(x, y, z)离目标点的距离
        """
        return np.linalg.norm(np.array([x, y, z]) - np.array([self.end.x, self.end.y, self.end.z]))

    def is_collide(self, sim, node):
        """ 判断p_new点和p_now的连线是否碰撞到障碍物
        """
        TYPE_CHANGED = False

        LEFT_GRIPPER_GEOMS = [
            'link7_collision',
            # 'link6_collision',
            # 'link5_collision',
            # 'link4_collision'
        ]
        COLLISIONS = ['obstacle_box', ]

        if node is None:
            return False

        for x, y, z in zip(node.path_x, node.path_y, node.path_z):

            target_pos = np.array([x, y, z])
            target_rot = np.array([1, 0, 0, 0])
            qpos = sim.controller.ik(target_pos, target_rot)
            sim.set_joint_qpos(qpos)

            sim.forward()
            sim.render()

            if not TYPE_CHANGED:
                LEFT_GRIPPER_GEOMS = sim.get_geom_id(LEFT_GRIPPER_GEOMS)
                COLLISIONS = sim.get_geom_id(COLLISIONS)
                TYPE_CHANGED = True

            is_collision = sim.is_contact(LEFT_GRIPPER_GEOMS, COLLISIONS, verbose=0)
            if is_collision:
                logging.info("find collision")
                return True
        return False


def rrt_star(start, end, sim) -> Union[List[list], None]:
    show_animation = True
    play_area = [0.1, 1.0, -0.65, 0.65, 0.1, 0.75]
    # Set Initial parameters
    rrt = RRT(
        start=start,
        goal=end,
        play_area=play_area,  # 树的生长区域，左下[-2, 0, 0] ==> 右上[13, 13, 13]
        sim=sim,
        expand_dis=0.04,  # 树枝长度
        goal_sample_rate=20,
        max_iter=1000,
    )

    sim.mj_model.geom('obstacle_box').margin[0] = 0.01
    sim.save_state()

    MAX_FAILTURE_TIMES = 3
    path = None
    for i in range(MAX_FAILTURE_TIMES):
        path = rrt.planning(animation=show_animation)
        if path is None:
            logging.info("Cannot find path")
            sim.load_state()
            continue
        else:
            logging.info("found path!!")
            sim.load_state()
            break

    sim.mj_model.geom('obstacle_box').margin[0] = 0.0

    plt.close()
    if path is not None:
        # 绘制最终路径
        if show_animation:
            plt.close()
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')  # 添加三维子图
            ax.set_xlim([play_area[0], play_area[1]])
            ax.set_ylim([play_area[2], play_area[3]])
            ax.set_zlim([play_area[4], play_area[5]])
            ax.plot([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], '-b',
                    linewidth=2.0)

            plt.pause(1)
            plt.close()
        return path
    return None
