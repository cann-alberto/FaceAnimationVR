"""
Project: PosingBySketching
Version: 4.0

==========================
OpenVR Compatible (HTC Vive)
==========================

OpenVR Compatible head mounted display
It uses a python wrapper to connect with the SDK
"""
import threading
import openvr
import bpy
import math
import sys
import copy
from enum import Enum
from mathutils import Quaternion
from mathutils import Matrix
from mathutils import Vector
import winsound

import time

from . import HMD_Base

from ..lib import (
        checkModule,
        )

import datetime
#from scipy.optimize import linear_sum_assignment
import numpy as np


import bmesh
import mathutils
from mathutils import *
from math import *
import time

currObject = ""
currBone = ""
currObject_l = ""
currBone_l = ""

##### MY GLOBAL VARIABLE
startTime = 0
endTime = 0
weights = []
v_index = 0
thread_is_done = False


# Algorithm from "New Algorithms for 2D and 3D Point Matching: Pose Estimation and Correpondance" used also in
#   1) A New Point Matching Algorithm for Non-Rigid Registration
#   2) Enhancing Character Posing by a Sketch-Based Interaction
class softAss_detAnnealing_3(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    # Compute 3D distance
    def compute_dist(self, p1, p2):
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2))

    # Move the first bone at the beggining og the stroke
    def set_init_cond(self):
        bones = bpy.data.objects['Armature'].pose.bones  # <H
        stroke = bpy.data.curves['Stroke'].splines[0]  # <H

        # TODO: Find root bones
        for b in bones:
            if b.parent == None:
                print(bpy.data.objects['Armature'].matrix_world * b.head)

                pmatrix = b.bone.matrix_local
                omatrix = bpy.data.objects['Armature'].matrix_world


                target_loc = bpy.data.objects['StrokeObj'].matrix_world * stroke.bezier_points[0].co
                b.location = omatrix.inverted() * pmatrix.inverted() * target_loc

                bpy.context.scene.update()
                print(bpy.data.objects['Armature'].matrix_world * b.head)

    def run(self):
        bones = bpy.data.objects['Armature'].pose.bones  # <H
        # stroke = bpy.data.curves['Stroke'].splines[0] # <H

        dict = {}
        stroke_points = []
        num_points = 0
        for k in range(0, len(bpy.data.curves['Stroke'].splines)):
            stroke = bpy.data.curves['Stroke'].splines[k]
            for j in range(0, len(stroke.bezier_points)):
                p = stroke.bezier_points[j]
                # World position of the point
                point = bpy.data.objects['StrokeObj'].matrix_world * p.co
                # Add into the dictionoary for correspondance
                dict[num_points] = [k, j]  # [spline_index, point_index]
                # Add in the point list
                stroke_points.append(point)
                num_points += 1

        # [DEBUG]:
        # print(dict)
        # for i in range(0, num_points):
        #    print(dict[i])
        #    print(dict[i][0], dict[i][1])

        # INIT PARAMETERS
        # values of the paper
        # beta_f = 0.2
        # beta_r = 1.075
        # beta = 0.00091
        # alpha = 0.03
        # I0 = 4
        # I1 = 30

        # custom value
        beta_f = 0.2
        beta_r = 1.9
        beta = 0.00091
        alpha = 0.03
        I0 = 4
        I1 = 30

        iteration = 0

        # set starting condition
        self.set_init_cond()

        while beta <= beta_f:
            print('iteration:', iteration)

            Qjk = []

            # Compute Qjk
            for k in range(0, num_points):
                p = bpy.data.curves['Stroke'].splines[dict[k][0]].bezier_points[dict[k][1]]

                # World position of the point
                point = bpy.data.objects['StrokeObj'].matrix_world * p.co

                costs_row = []
                for i in range(0, len(bones)):
                    b = bones[i]
                    # World position of the bone
                    tail = bpy.data.objects['Armature'].matrix_world * b.tail  # <H
                    head = bpy.data.objects['Armature'].matrix_world * b.head  # <H
                    # center = bpy.data.objects['Armature'].matrix_world * b.center # <H

                    dist = self.compute_dist(tail, point)
                    # length_diff = abs(b.length - compute_dist(point,head))
                    # costs_row.append(dist + length_diff)
                    costs_row.append(-(dist - alpha))

                Qjk.append(costs_row)

            m0 = np.asarray(Qjk)

            # Deterministic annealing
            for i in range(0, num_points):
                for j in range(0, len(bones)):
                    m0[i, j] = math.exp(beta * m0[i, j])

                    # TODO: Add outlier row and cols

            # Sinkhorn's method DO until m converges
            m1 = np.ones((num_points, len(bones)))

            for i in range(0, I1):  # TODO: set coverage threshold
                for i in range(0, num_points):
                    for j in range(0, len(bones)):
                        m1[i, j] = m0[i, j] / np.sum(m0[i])

                for i in range(0, num_points):
                    for j in range(0, len(bones)):
                        m0[i, j] = m1[i, j] / np.sum(m1[:, j])

                        # [DEBUG]: Rows - Cols sum up
            # print("rows - cols sum up")
            # for i in range (0,num_points):
            #    print (np.sum(m0[i]))

            # for j in range (0, len(bones)):
            #    print(np.sum(m0[:,j]))

            # Softassign - ENERGY FORMULATION E3D
            E3D = []
            # Compute E3D
            for k in range(0, num_points):
                p = bpy.data.curves['Stroke'].splines[dict[k][0]].bezier_points[dict[k][1]]

                # World position of the point
                point = bpy.data.objects['StrokeObj'].matrix_world * p.co

                costs_row = []
                for i in range(0, len(bones)):
                    b = bones[i]
                    # World position of the bone
                    tail = bpy.data.objects['Armature'].matrix_world * b.tail  # <H
                    head = bpy.data.objects['Armature'].matrix_world * b.head  # <H
                    # center = bpy.data.objects['Armature'].matrix_world * b.center # <H

                    dist = self.compute_dist(tail, point)
                    # length_diff = abs(b.length - compute_dist(point,head))
                    # costs_row.append(dist + length_diff)
                    costs_row.append(dist - alpha * m0[k, i])

                E3D.append(costs_row)

            cost = np.transpose(m0) * np.transpose(E3D)
            row_ind, col_ind = linear_sum_assignment(cost)
            print(col_ind)
            print(cost[row_ind, col_ind].sum())

            # Update pose parameters
            for i in range(0, len(bones)):
                b = bones[i]
                # Set target positions
                #
                bpy.data.objects[b.name].location = copy.deepcopy(bpy.data.objects['StrokeObj'].matrix_world *
                                                                  bpy.data.curves['Stroke'].splines[
                                                                      dict[col_ind[i]][0]].bezier_points[
                                                                      dict[col_ind[i]][1]].co)

                # Articulate armature
                constr = b.constraints['Damped Track']
                constr.target = bpy.data.objects[b.name]

            iteration += 1
            beta = beta * beta_r

class get_weights_by_LQ(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    my_obj = bpy.context.active_object
    stroke = bpy.data.objects["StrokeObj"]
    center_obj = bpy.data.objects["center"]

    # LEAST SQUARE ALGORITHM
    def calc_weights(self, obj, pin_position, v_index, w0):

        # shapekey_array = bpy.data.shape_keys["Key_1"].key_blocks
        # shapes_name = obj.active_shape_key.id_data.name
        # shapekey_array = bpy.data.shape_keys[shapes_name].key_blocks
        shapekey_array = obj.data.shape_keys.key_blocks

        # empty the shape target
        # for v in shapekey_array["Target"].data:
        #     v.co = Vector((0, 0, 0))
        # empty the final
        for v in shapekey_array["Final"].data:
            v.co = Vector((0, 0, 0))

        B = []
        M = []

        n_shapes = len(shapekey_array) - 0

        # create blendshape matrix
        for shape in shapekey_array:
            M = []
            for v in shape.data:
                M.append(v.co)
            B.append(M)
        B = np.array(B)

        n_vrtx = len(B[0])

        movedpin2 = np.copy(B[0])
        movedpin2[v_index] = pin_position
        movedpin2 = np.reshape(movedpin2, (1, n_vrtx * 3))

        f0 = np.copy(np.reshape(B[0], (1, n_vrtx * 3)))

        # difference between basic face and pin
        movedpin2 = np.transpose(movedpin2 - f0)

        # delete the basic shape
        # B = np.delete(B,1,0)

        # trasform tridimensional matrix in twodimensional
        B = np.reshape(B, (n_shapes, n_vrtx * 3))
        # subtract base face from blendshapes
        B = B - f0

        # here the B matrix is already transposed
        Bt = np.copy(B)

        B = np.transpose(B)

        # movedpin2 = np.transpose( movedpin2 - B[:,0] )

        BtB = np.dot(Bt, B)

        # pseudoinverse (not needed)
        # Bp = np.linalg.pinv(B)

        I = np.identity(n_shapes)

        alpha = 0.001

        # movedpin = np.reshape(movedpin, (n_vrtx*3, 1) )

        fac1 = np.linalg.inv(BtB + alpha * I)
        fac2 = np.dot(Bt, movedpin2) + (alpha * w0)

        weights = np.dot(fac1, fac2)

        '''
        i = 0
        for shape in shapekey_array:
            shape.value = weights[i]*100
            i += 1
        '''
        return weights

    def run(self):

        global weights
        stroke_lenght = len(self.stroke.data.splines[-1].bezier_points)
        pin_start = self.stroke.data.splines[-1].bezier_points[0].co
        pin_end = self.stroke.data.splines[-1].bezier_points[stroke_lenght - 1].co
        global v_index

        weights = self.calc_weights(self.my_obj, pin_end, v_index, 0)

        print("initial weights:", weights)

        #mute the final shape for the moment
        shapekey_array = self.my_obj.data.shape_keys.key_blocks
        for shape in shapekey_array:
            if(shape.name == "Final"):
                shape.mute = True

        global thread_is_done
        thread_is_done = True





class State(Enum):
    IDLE = 1
    #DECISIONAL = 2
    CONTROL_PIN = 2
    INTERACTION_LOCAL = 3
    NAVIGATION_ENTER = 4
    NAVIGATION = 5
    NAVIGATION_EXIT = 6
    ZOOM_IN = 7
    ZOOM_OUT = 8
    CAMERA_MOVE_CONT = 9
    CAMERA_ROT_CONT = 10
    SCALING = 11
    CHANGE_AXES = 12
    DRAWING = 13
    TRACKPAD_BUTTON_DOWN = 14

    MANIPULATE_CURVE = 15

    ROTATE_PIN = 16

class StateLeft(Enum):
    IDLE = 1
    DECISIONAL = 2
    INTERACTION_LOCAL = 3
    NAVIGATION_ENTER = 4
    NAVIGATION = 5
    NAVIGATION_EXIT = 6
    ZOOM_IN = 7
    ZOOM_OUT = 8
    CAMERA_MOVE_CONT = 9
    CAMERA_ROT_CONT = 10
    SCALING = 11
    CHANGE_AXES = 12
    DRAWING = 13
    TRACKPAD_BUTTON_DOWN = 14

    TIMELINE_ENTER = 15
    TIMELINE = 16
    TIMELINE_EXIT = 17

    START_FRAME = 18
    END_FRAME = 19

    DRAG_PANEL = 20

    THRESHOLD = 21
    THRESHOLD_EXIT = 22

    SHIFT_TIME = 23
    SCALE_TIME = 24

class OpenVR(HMD_Base):
    ctrl_index_r = 0
    ctrl_index_l = 0
    tracker_index = 0
    hmd_index = 0
    curr_axes_r = 0
    curr_axes_l = 0
    state = State.IDLE
    state_l = StateLeft.IDLE

    diff_rot = Quaternion()
    diff_loc = bpy.data.objects['Controller.R'].location
    initial_loc = Vector((0,0,0))
    initial_rot = Quaternion()

    diff_rot_l = Quaternion()
    diff_loc_l = bpy.data.objects['Controller.L'].location
    initial_loc_l = Vector((0, 0, 0))
    initial_rot_l = Quaternion()

    diff_distance = 0
    initial_scale = 0
    trans_matrix = bpy.data.objects['Camera'].matrix_world * bpy.data.objects['Origin'].matrix_world
    diff_trans_matrix = bpy.data.objects['Camera'].matrix_world * bpy.data.objects['Origin'].matrix_world

    objToControll = ""
    boneToControll = ""
    objToControll_l = ""
    boneToControll_l = ""
    zoom = 1
    rotFlag = True
    axes = ['LOC/ROT_XYZ','LOC_XYZ','LOC_X','LOC_Y','LOC_Z','ROT_XYZ','ROT_X','ROT_Y','ROT_Z']

    gui_obj = ['Camera', 'Origin',
               'Controller.R', 'Controller.L',
               'Text.R', 'Text.L']


    def __init__(self, context, error_callback):
        super(OpenVR, self).__init__('OpenVR', True, context, error_callback)
        checkModule('hmd_sdk_bridge')

    def _getHMDClass(self):
        """
        This is the python interface to the DLL file in hmd_sdk_bridge.
        """
        from bridge.hmd.openvr import HMD
        return HMD

    @property
    def projection_matrix(self):
        if self._current_eye:
            matrix = self._hmd.getProjectionMatrixRight(self._near, self._far)
        else:
            matrix = self._hmd.getProjectionMatrixLeft(self._near, self._far)

        self.projection_matrix = matrix
        return super(OpenVR, self).projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value):
        self._projection_matrix[self._current_eye] = \
            self._convertMatrixTo4x4(value)

    def init(self, context):
        """
        Initialize device

        :return: return True if the device was properly initialized
        :rtype: bool
        """

        vrSys = openvr.init(openvr.VRApplication_Scene)
        self.ctrl_index_r, self.ctrl_index_l, self.tracker_index, self.hmd_index = self.findControllers(vrSys)

        ####AT THE MOEMNT
        # self.ctrl_index_r = 2
        #self.ctrl_index_l = 3

        if bpy.data.objects.get('StrokeObj') is None:
            self.create_curve()
        bpy.data.window_managers['WinMan'].virtual_reality.lock_camera = True

        try:
            HMD = self._getHMDClass()
            self._hmd = HMD()

            # bail out early if we didn't initialize properly
            if self._hmd.get_state_bool() == False:
                raise Exception(self._hmd.get_status())

            # Tell the user our status at this point.
            self.status = "HMD Init OK. Make sure lighthouses running else no display."

            # gather arguments from HMD
            self.setEye(0)
            self.width = self._hmd.width_left
            self.height = self._hmd.height_left

            self.setEye(1)
            self.width = self._hmd.width_right
            self.height = self._hmd.height_right

            # initialize FBO
            if not super(OpenVR, self).init():
                raise Exception("Failed to initialize HMD")

            # send it back to HMD
            if not self._setup():
                raise Exception("Failed to setup OpenVR Compatible HMD")

        except Exception as E:
            self.error("OpenVR.init", E, True)
            self._hmd = None
            return False

        else:
            return True

    def _setup(self):
        return self._hmd.setup(self._color_texture[0], self._color_texture[1])

    # ---------------------------------------- #
    # Functions
    # ---------------------------------------- #

    ## Find the index of the two controllers
    def findControllers(self, vrSys):
        r_index, l_index, tracker_index, hmd_index = -1, -1, -1, -1

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == openvr.TrackedDeviceClass_Invalid:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - ")
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == openvr.TrackedDeviceClass_HMD:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - HMD")
                hmd_index = i
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == openvr.TrackedDeviceClass_TrackingReference:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - TrackingReference")
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == openvr.TrackedDeviceClass_Controller:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - Controller")
                if r_index == -1:
                    r_index = i
                else:
                    l_index = i
            if openvr.IVRSystem.getTrackedDeviceClass(vrSys, i) == 3:
                print(i, openvr.IVRSystem.getTrackedDeviceClass(vrSys, i), " - VIVE Tracker")
                tracker_index = i

        print('r_index = ', r_index, ' l_index = ', l_index)
        return r_index, l_index, tracker_index, hmd_index

    def setController(self):
        poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
        poses = poses_t()
        openvr.VRCompositor().waitGetPoses(poses, len(poses), None, 0)

        matrix = poses[self.ctrl_index_r].mDeviceToAbsoluteTracking
        matrix2 = poses[self.ctrl_index_l].mDeviceToAbsoluteTracking

        try:
            camera = bpy.data.objects["Camera"]
            ctrl = bpy.data.objects["Controller.R"]
            ctrl_l = bpy.data.objects["Controller.L"]


            self.trans_matrix = camera.matrix_world * bpy.data.objects['Origin'].matrix_world
            RTS_matrix = Matrix(((matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]),
                                 (matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]),
                                 (matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]),
                                 (0, 0, 0, 1)))

            RTS_matrix2 = Matrix(((matrix2[0][0], matrix2[0][1], matrix2[0][2], matrix2[0][3]),
                                 (matrix2[1][0], matrix2[1][1], matrix2[1][2], matrix2[1][3]),
                                 (matrix2[2][0], matrix2[2][1], matrix2[2][2], matrix2[2][3]),
                                 (0, 0, 0, 1)))

            # Interaction state active
            if(self.rotFlag):
                ctrl.matrix_world = self.trans_matrix * RTS_matrix
                bpy.data.objects["Text.R"].location = ctrl.location
                bpy.data.objects["Text.R"].rotation_quaternion = ctrl.rotation_quaternion * Quaternion((0.707, -0.707, 0, 0))

                ctrl_l.matrix_world = self.trans_matrix * RTS_matrix2
                bpy.data.objects["Text.L"].location = ctrl_l.location
                bpy.data.objects["Text.L"].rotation_quaternion = ctrl_l.rotation_quaternion * Quaternion((0.707, -0.707, 0, 0))

            # Navigation state active
            else:
                diff_rot_matr = self.diff_rot.to_matrix()
                #inverted_matrix = RTS_matrix * diff_rot_matr.to_4x4() ############# MODIFICATO DA EMANUELE
                inverted_matrix = RTS_matrix2 * diff_rot_matr.to_4x4()
                inverted_matrix = inverted_matrix.inverted()
                stMatrix = self.diff_trans_matrix * inverted_matrix
                quat = stMatrix.to_quaternion()
                camera.rotation_quaternion = quat



        except:
            print("ERROR: ")

    def changeSelection(self,obj,bone,selectState):

        if selectState:
            print("SELECT: ", obj, bone)
            if obj != "":
                if bone != "":
                    bpy.data.objects[obj].select = True
                    bpy.context.scene.objects.active = bpy.data.objects[obj]
                    bpy.ops.object.mode_set(mode='POSE')
                    bpy.data.objects[obj].data.bones[bone].select = True

                else:
                    bpy.data.objects[obj].select = True
                    bpy.context.scene.objects.active = bpy.data.objects[obj]
                    bpy.ops.object.mode_set(mode='OBJECT')

        else:
            print ("DESELECT: ", obj, bone)
            if obj != "":
                if bone != "":
                    bpy.data.objects[obj].select = True
                    bpy.context.scene.objects.active = bpy.data.objects[obj]
                    bpy.ops.object.mode_set(mode='POSE')
                    bpy.data.objects[obj].data.bones[bone].select = False
                    bpy.data.objects[obj].select = False
                else:
                    bpy.data.objects[obj].select = True
                    bpy.context.scene.objects.active = bpy.data.objects[obj]
                    bpy.data.objects[obj].select = False
                    bpy.ops.object.mode_set(mode='OBJECT')

    ## Computes distance from controller
    def computeTargetObjDistance(self, Object, Bone, isRotFlag):

        if isRotFlag:
            tui = bpy.data.objects['Controller.R']
        else:
            tui = bpy.data.objects['Controller.L']

        obj = bpy.data.objects[Object]
        if Bone != "":
            pbone = obj.pose.bones[Bone]
            return (math.sqrt(pow((tui.location[0] - (pbone.center[0] + obj.location[0])), 2) + pow(
                (tui.location[1] - (pbone.center[1] + obj.location[1])), 2) + pow(
                (tui.location[2] - (pbone.center[2] + obj.location[2])), 2)))
        else:
            loc = obj.matrix_world.to_translation()
            return (math.sqrt(pow((tui.location[0] - loc[0]), 2) + pow((tui.location[1] - loc[1]), 2) + pow(
                (tui.location[2] - loc[2]), 2)))

    ## Returns the object closest to the Controller
    def getClosestItem(self, isRight):
        dist = sys.float_info.max
        cObj = ""
        cBone = ""
        distThreshold = 0.5

        for object in bpy.data.objects:
            if object.type == 'ARMATURE':
                if not ':TEST_REF' in object.name:
                    for bone in object.pose.bones:
                        currDist = self.computeTargetObjDistance(object.name, bone.name, isRight)
                        bone.bone_group = None
                        if (currDist < dist and currDist < distThreshold):
                            dist = currDist
                            cObj = object.name
                            cBone = bone.name


            else:
                #if object.type != 'CAMERA' and not object.name in self.gui_obj:
                if not object.name in self.gui_obj:
                    currDist = self.computeTargetObjDistance(object.name, "", isRight)
                    # print(object.name, bone.name, currDist)
                    if (currDist < dist and currDist < distThreshold):
                        dist = currDist
                        cObj = object.name
                        cBone = ""



        # Select the new closest item
        print(cObj, cBone)
        print("--------------------------------")
        if(cBone!=""):
            bpy.data.objects[cObj].pose.bones[cBone].rotation_mode = 'QUATERNION'
            #bpy.data.objects[cObj].pose.bones[cBone].bone_group = bpy.data.objects[cObj].pose.bone_groups["SelectedBones"]

        return cObj, cBone

    ## Resets the original transformation when constraints movement are used
    def applyConstraint(self, isRight):
        if isRight:
            type = self.axes[self.curr_axes_r].split('_')[0]
            axes = self.axes[self.curr_axes_r].split('_')[1]
            obj = self.objToControll
            bone = self.boneToControll
            init_loc = self.initial_loc
            init_rot = self.initial_rot

        else:
            type = self.axes[self.curr_axes_l].split('_')[0]
            axes = self.axes[self.curr_axes_l].split('_')[1]
            obj = self.objToControll_l
            bone = self.boneToControll_l
            init_loc = self.initial_loc_l
            init_rot = self.initial_rot_l

        if type == 'LOC':
            if bone!="":
                bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'XYZ'
                bpy.data.objects[obj].pose.bones[bone].rotation_euler = init_rot
                bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'QUATERNION'

            else:
                bpy.data.objects[obj].rotation_mode = 'XYZ'
                bpy.data.objects[obj].rotation_euler = init_rot
                bpy.data.objects[obj].rotation_mode = 'QUATERNION'

            if axes == 'X':
                if bone != "":
                    bpy.data.objects[obj].pose.bones[bone].location[1] = init_loc[1]
                    bpy.data.objects[obj].pose.bones[bone].location[2] = init_loc[2]
                else:
                    bpy.data.objects[obj].location[1] = init_loc[1]
                    bpy.data.objects[obj].location[2] = init_loc[2]

            if axes == 'Y':
                if bone != "":
                    bpy.data.objects[obj].pose.bones[bone].location[0] = init_loc[0]
                    bpy.data.objects[obj].pose.bones[bone].location[2] = init_loc[2]
                else:
                    bpy.data.objects[obj].location[0] = init_loc[0]
                    bpy.data.objects[obj].location[2] = init_loc[2]

            if axes == 'Z':
                if bone != "":
                    bpy.data.objects[obj].pose.bones[bone].location[0] = init_loc[0]
                    bpy.data.objects[obj].pose.bones[bone].location[1] = init_loc[1]
                else:
                    bpy.data.objects[obj].location[0] = init_loc[0]
                    bpy.data.objects[obj].location[1] = init_loc[1]

        if type == 'ROT':
            if bone!="":
                bpy.data.objects[obj].pose.bones[bone].location = init_loc
            else:
                bpy.data.objects[obj].location = init_loc

            if axes == 'X':
                if bone!="":
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[1] = init_rot[1]
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[2] = init_rot[2]
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'QUATERNION'
                else:
                    bpy.data.objects[obj].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].rotation_euler[1] = init_rot[1]
                    bpy.data.objects[obj].rotation_euler[2] = init_rot[2]
                    bpy.data.objects[obj].rotation_mode = 'QUATERNION'

            if axes == 'Y':
                if bone!="":
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[0] = init_rot[0]
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[2] = init_rot[2]
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'QUATERNION'
                else:
                    bpy.data.objects[obj].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].rotation_euler[0] = init_rot[0]
                    bpy.data.objects[obj].rotation_euler[2] = init_rot[2]
                    bpy.data.objects[obj].rotation_mode = 'QUATERNION'

            if axes == 'Z':
                if bone!="":
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[0] = init_rot[0]
                    bpy.data.objects[obj].pose.bones[bone].rotation_euler[1] = init_rot[1]
                    bpy.data.objects[obj].pose.bones[bone].rotation_mode = 'QUATERNION'
                else:
                    bpy.data.objects[obj].rotation_mode = 'XYZ'
                    bpy.data.objects[obj].rotation_euler[0] = init_rot[0]
                    bpy.data.objects[obj].rotation_euler[1] = init_rot[1]
                    bpy.data.objects[obj].rotation_mode = 'QUATERNION'

    ## Create a curve from a set of points
    def create_curve(self):
        name = "Stroke"
        curvedata = bpy.data.curves.new(name=name, type='CURVE')
        curvedata.dimensions = '3D'
        curvedata.fill_mode = 'FULL'
        curvedata.bevel_depth = 0.01

        ob = bpy.data.objects.new(name + "Obj", curvedata)
        bpy.context.scene.objects.link(ob)
        ob.show_x_ray = True

    def add_spline(self, point):
        curvedata = bpy.data.curves['Stroke']
        polyline = curvedata.splines.new('BEZIER')
        polyline.resolution_u = 1
        polyline.bezier_points[0].co = point

    ## Add new point to the curve
    def update_curve(self, point):
        polyline = bpy.data.curves['Stroke'].splines[-1]
        polyline.bezier_points.add(1)
        polyline.bezier_points[-1].co = point
        polyline.bezier_points[-1].handle_left = point
        polyline.bezier_points[-1].handle_right = point
        print (datetime.datetime.now())

    def remove_spline(self):

        ##update UI
        self.delete_bar(len(self.target_list) - 1)

        polyline = bpy.data.curves['Stroke'].splines[-1]
        bpy.data.curves['Stroke'].splines.remove(polyline)


        # remove last shape target
        TargetIndex = self.my_obj.data.shape_keys.key_blocks.keys().index(self.target_list[-1])
        self.my_obj.active_shape_key_index = TargetIndex
        bpy.ops.object.shape_key_remove()
        # remove keyframes
        target_n = len(self.target_list) - 1
        for action in bpy.data.actions:
            for fcu in action.fcurves:
                if (fcu.data_path == 'key_blocks["' + "Target"+ str(target_n) + '"].value'):
                    print("remove: ", 'key_blocks["' + "Target"+ str(target_n) + '"].value')
                    action.fcurves.remove(fcu)

        pin_obj = bpy.data.objects["Pin_stroke" + str(target_n)]
        bpy.data.objects.remove(pin_obj)
        arrow_obj = bpy.data.objects["arrow_" + str(target_n)]
        bpy.data.objects.remove(arrow_obj)

        if(self.shapes_at_start[-1] != "basis"):
            index = self.target_list.index(self.shapes_at_start[-1])
            self.cut_frame_list[index] = 10**10



        self.beziere_list.remove(self.beziere_list[-1])
        self.main_vertex_list.remove(self.main_vertex_list[-1])
        self.axis_list.remove(self.axis_list[-1])
        self.center_list.remove(self.center_list[-1])
        self.target_list.remove(self.target_list[-1])
        self.shapes_at_start.remove(self.shapes_at_start[-1])
        self.start_frame_list.remove(self.start_frame_list[-1])
        self.end_frame_list.remove(self.end_frame_list[-1])
        self.cut_frame_list.remove(self.cut_frame_list[-1])



    def remove_closest_spline(self, pos):


        d_min = 100000000000000
        i=0
        b_index = 0
        for bezier in self.stroke.data.splines:
            d = self.find_distance( bezier.bezier_points[-1].co, pos )
            if(d < d_min):
                d_min = d
                b_index = i
            i+=1

        ##update UI
        self.delete_bar(b_index)

        bezier = self.stroke.data.splines[b_index]
        self.stroke.data.splines.remove( bezier )

        ## remove PIN
        pin_obj = bpy.data.objects["Pin_stroke" + str(b_index)]
        bpy.data.objects.remove(pin_obj)
        ## rename pins
        for target in self.target_list:
            if(int(target[6]) > b_index):
                pin_obj_2 = bpy.data.objects["Pin_stroke" + target[6]]
                pin_obj_2.name = "Pin_stroke" + str( int(target[6])-1 )


        # # remove keyframes
        flag = 0
        for action in bpy.data.actions:
            i=0
            for fcu in action.fcurves:
                if (fcu.data_path == 'key_blocks["' + "Target"+ str(b_index) + '"].value'):
                    print("remove: ", 'key_blocks["' + "Target"+ str(b_index) + '"].value == ', fcu.data_path)
                    action.fcurves.remove(fcu)
            i+=1


        # remove keyframes
        # i=0
        # for action in bpy.data.actions:
        #     for fcu in action.fcurves:
        #         if (fcu.data_path[:18] == 'key_blocks["Target'):
        #             if (fcu.data_path == 'key_blocks["' + "Target" + str(b_index) + '"].value'):
        #                 action.fcurves.remove(fcu)
        #             elif (i >= b_index):
        #                 fcu.data_path = 'key_blocks["' + "Target" + str(i) + '"].value'
        #             i+=1

        # remove ith shape target
        TargetIndex = self.my_obj.data.shape_keys.key_blocks.keys().index(self.target_list[b_index])
        self.my_obj.active_shape_key_index = TargetIndex
        bpy.ops.object.shape_key_remove()

        print("BEFORE CHANGE NAME:")
        for action in bpy.data.actions:
            for fcu in action.fcurves:
                print(fcu.data_path)

                # rename shape target
        i=0
        shapekey_array = self.my_obj.data.shape_keys.key_blocks
        for shape in shapekey_array:
            name = shape.name
            if(name[:6] == "Target"):
                if(i >= b_index):
                    shape.name = "Target" + str(i)
                i+=1

        print("AFTER CHANGE NAME:")
        for action in bpy.data.actions:
            for fcu in action.fcurves:
                print(fcu.data_path)

        print("RENAMED!!!!:")
        flag = 0
        for action in bpy.data.actions:
            i=0
            for fcu in action.fcurves:
                if (fcu.data_path[:18] == 'key_blocks["Target'):
                    if(i >= b_index):
                        print(fcu.data_path, "BECAME:")
                        fcu.data_path = 'key_blocks["' + "Target" + str(i) + '"].value'
                        print(fcu.data_path)

                    i += 1




        self.beziere_list.remove(self.beziere_list[b_index])
        self.main_vertex_list.remove(self.main_vertex_list[b_index])
        self.axis_list.remove(self.axis_list[b_index])
        self.center_list.remove(self.center_list[b_index])
        self.target_list.remove(self.target_list[b_index])
        self.shapes_at_start.remove(self.shapes_at_start[b_index])
        self.start_frame_list.remove(self.start_frame_list[b_index])
        self.end_frame_list.remove(self.end_frame_list[b_index])
        self.cut_frame_list.remove(self.cut_frame_list[b_index])

        # rename target list
        i=0
        for name in self.target_list:
            if(i == b_index):
                self.target_list[i] = "Target" + str(i)
            if(i > b_index):
                self.target_list[i] = "Target" + str(i-1)
            i+=1



        #self.blend_targets(self.my_obj)






    ########################################################### MY VARIABLES
    my_obj = bpy.data.objects[bpy.context.active_object.name]
    stroke = bpy.data.objects["StrokeObj"]
    center_obj = bpy.data.objects["center"]

    #DATA BASE:
    beziere_list = []
    center_list = []
    axis_list = []
    main_vertex_list = []
    target_list = []
    start_frame_list = []
    end_frame_list = []
    cut_frame_list = []
    shapes_at_start = []
    verts_at_start = []
    symmetry = []

    pin_list = []
    pin_angle = []
    allow_draw = True


    next_state_l = StateLeft.IDLE
    y_pre = 0
    y_pre_10 = 0

    # to move pin
    last_pen_pos = Vector((0,0,0))
    pin_stroke = Vector((0,0,0))
    end_stroke_point = Vector((0,0,0))
    close_b_idx = 0

    symmetry_switch = False
    symmetry_time_UI = 0


    ############################################################ MY FUNCTION
    def my_handler(self, *args):
        # print("Frame Change", bpy.context.scene.frame_current)
        #mute all the shapekey
        shapekey_array = self.my_obj.data.shape_keys.key_blocks
        for shape in shapekey_array:
            if (shape.name != "Final"):
                shape.mute = True
        self.rot_axis()

    def rot_axis(self):

        frame_current = bpy.context.scene.frame_current

        i=0
        ## index list of start shape to be update
        index_list = []
        for shape in self.shapes_at_start:
            frame_start = self.start_frame_list[i]
            if(shape!="basis" and frame_start==frame_current):
                index_list.append(i)
              #  print("found at frame!")
            i+=1

        k = bpy.data.shape_keys.keys()

        my_obj = bpy.context.active_object

        # shapes_name = my_obj.active_shape_key.id_data.name
        # shapekey_array = bpy.data.shape_keys[shapes_name].key_blocks
        shapekey_array = my_obj.data.shape_keys.key_blocks

        # shapekey target
        shapekey1 = shapekey_array[1]
        # shapekey for visualization
        shapekey2 = shapekey_array["Final"]

        '''
        for shape in shapekey_array:
            if(shape.name == "Target"):
                shape_target = shape
        '''
        #shape_target = shapekey_array["Target"]

        #weight = shapekey1.value

        # shapekey3.value = 0
        shapekey2.value = 1

        m_vertices = bpy.context.active_object.data.vertices

        #center_obj = bpy.data.objects["center"]

        # initialize direction rotation
        dir_rot = 1

        # if the center is out of the mesh, change direction rotation
        # if(find_teta(pos_A,pos_B,center) > math.pi/2):
        # dir_rot = -1

        WORLD = bpy.context.object.matrix_world
        WORLD_INV = copy.copy(WORLD)
        WORLD_INV.invert()

        ###
        s=0
        for target in self.target_list:
            #print(s,"th center:",target,self.center_list[s], "simmetry: ",self.symmetry[s])
            s+=1


        i = 0
        while (i < len(m_vertices)):
            # starting vertex
            v1 = m_vertices[i].co
            # v1 = WORLD @ v1
            v1 = WORLD * v1
            # target vertex
            # v2 = shapekey1.data[i].co
            # v2 = shape_target.data[i].co
            # v2 = WORLD @ v2
            # v2 = WORLD * v2


            ######## NEW ALGORITHM
            v_list = []
            j=0
            for bezier in self.beziere_list:
                target = self.target_list[j]
                # if the shape key move this vertex
                #if(frame_current >= self.start_frame_list[j] and frame_current <= self.end_frame_list[j]):
                if(True):
                    if(shapekey_array[target].data[i].co != v1):
                        weight = shapekey_array[target].value
                        if( frame_current >= self.start_frame_list[j] and frame_current <= self.cut_frame_list[j]):

                            if(self.shapes_at_start[j] != "basis"):
                                shape_start_name = self.shapes_at_start[j]
                                v1 = shapekey_array[shape_start_name].data[i].co
                                #v1 = self.verts_at_start[j][i]


                            v2 = shapekey_array[target].data[i].co

                            center_original = self.center_list[j]

                            axis_original = self.axis_list[j]

                            i_axis = axis_original.copy()

                            ## if center is not at infinity -> rotate
                            if(center_original!="INFINITY"):

                                center = self.center_list[j].copy()

                                index = self.main_vertex_list[j]
                                main_vertex = m_vertices[index].co.copy()

                                # symmetry
                                if(self.symmetry[j] == "X"):
                                    if (v1.x * main_vertex.x < 0):
                                        center.x = -center.x
                                        main_vertex.x = -main_vertex.x
                                        i_axis.y = -i_axis.y
                                        i_axis.z = -i_axis.z


                                i_center = self.find_individual_center(center, main_vertex, v1, v2)


                                #i_axis = self.find_axis(v1, v2, i_center)


                                if (self.find_distance(v1, v2) < 0.001):
                                    teta = 0
                                else:
                                    teta = self.find_teta(v1, v2, i_center, i) * dir_rot

                                T = mathutils.Matrix.Translation(i_center)
                                Tinv = mathutils.Matrix.Translation(-i_center)

                                # rotation maximum angle
                                Rmax = mathutils.Matrix.Rotation(teta, 4, i_axis)
                                # vmax = T @ Rmax @ Tinv @ v1
                                vmax = T * Rmax * Tinv * v1

                                # rotation matrix for weighted angle
                                R = mathutils.Matrix.Rotation(teta * weight, 4, i_axis)

                                # find difference for traslation
                                diff = (v2 - vmax) * weight

                                vn = T * R * Tinv * v1
                                # vn = vn + diff * weight
                                vn = vn + diff

                            ## if center is at infinity -> traslate
                            else:
                                diff = v2 - v1
                                vn = v1 + (diff * weight)


                            v_list.append(vn)

                j+=1



            v0 = shapekey_array[0].data[i].co
            v_blended = self.blend_vertex_list(v_list, v0)


            # shapekey2.data[i].co = WORLD_INV @ vn
            shapekey2.data[i].co = WORLD_INV * v_blended

            ## update verts at start for current frame
            # for index in index_list:
            #     self.verts_at_start[index][i] =  WORLD_INV * v_blended

            i += 1

        # i=0
        # for bezier in self.beziere_list:
        #     print(self.shapes_at_start[i], self.target_list[i], self.start_frame_list[i], self.end_frame_list[i])
        #     i+=1

    def find_mipoint(self, verts_list):
        sum = Vector((0,0,0))
        for v in verts_list:
            sum = sum + v
        average = sum / len(verts_list)
        return  average

    def find_box_volume(self, obj):

        A = obj.bound_box[0]
        B = obj.bound_box[1]
        AB = self.find_distance(A,B)

        C = obj.bound_box[1]
        D = obj.bound_box[2]
        CD = self.find_distance(C,D)

        E = obj.bound_box[3]
        F = obj.bound_box[4]
        EF = self.find_distance(E,F)

        return AB * CD * EF

    def find_box_height(self, obj):

        E = obj.bound_box[3]
        F = obj.bound_box[4]
        EF = self.find_distance(E,F)

        return EF

    def find_box_diag(self,obj):

        E = obj.bound_box[6]
        F = obj.bound_box[3]
        EF = self.find_distance(E, F)

        return EF

    def blend_vertex_list(self, vertex_list, v0):

        diff = Vector((0,0,0))
        for v in vertex_list:
            diff = diff + (v-v0)

        return v0 + diff

    def blend_vertex(self, obj, index, frame):

        current_frame =  bpy.context.scene.frame_current

        bpy.context.scene.frame_current = frame

        shapekey_array = obj.data.shape_keys.key_blocks

        v0 = shapekey_array[0].data[index].co

        diff = Vector((0, 0, 0))
        for shape in shapekey_array:
            value = shape.value
            diff = diff + (shape.data[index].co - v0)*value

        # bring back frame
        bpy.context.scene.frame_current = current_frame

       # print("blend_vertex: ",diff)

        return v0 + diff

    def find_between_r(self, s, first, last):
        try:
            start = s.rindex(first) + len(first)
            end = s.rindex(last, start)
            return s[start:end]
        except ValueError:
            return ""

    def blend_shapes(self, obj, frame):

        # bpy.context.scene.frame_current = frame

        shapekey_array = obj.data.shape_keys.key_blocks

        blended_shape = []

        # i=0
        # for v0 in shapekey_array[0].data:
        #     diff = Vector((0, 0, 0))
        #     for shape in shapekey_array:
        #         if(shape.name!= "Final"):
        #             value = shape.value
        #             diff = diff + (shape.data[i].co - v0.co) * value
        #             print(shape.name, diff, value)
        #
        #     blended_shape.append(v0.co + diff)
        #     i+=1

        str1 = 'key_blocks["'
        str2 = '"].value'

        KeyName = obj.data.shape_keys.name

        i=0
        for v0 in shapekey_array[0].data:
            diff = Vector((0, 0, 0))
            for fcurve in bpy.data.actions[ KeyName + "Action" ].fcurves:

                shape_name = self.find_between_r(fcurve.data_path,str1,str2)
                shape = shapekey_array[shape_name]
                value = fcurve.evaluate(frame)
                diff = diff + (shape.data[i].co - v0.co) * value

            blended_shape.append(v0.co + diff)
            shapekey_array["blend"].data[i].co = v0.co + diff
            i+=1

        return blended_shape

    def blend_targets(self, obj):
        shapekey_array = obj.data.shape_keys.key_blocks

        i = 0
        diff = Vector((0, 0, 0))
        for v in shapekey_array[0].data:

            diff = Vector((0, 0, 0))
            n = 0
            for shape in shapekey_array:
                if(shape.name[:6] == "Target"):
                    # vm = vm + shape.data[i].co
                    diff = diff + (shape.data[i].co - v.co)
                    n += 1

            # shapekey_array["blend"].data[i].co = vm/n
            shapekey_array["blend"].data[i].co = v.co + diff
            i += 1

    def find_teta(self, A, B, C, i):
        AB = sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)
        AC = sqrt((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2 + (A[2] - C[2]) ** 2)
        BC = sqrt((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2 + (B[2] - C[2]) ** 2)

        # AB = round(AB,1)
        # AC = round(AC, 1)
        # BC = round(BC, 1)

        # cosin theorem
        arg = AC / (2 * BC) + BC / (2 * AC) - AB ** 2 / (2 * AC * BC)

        if (arg > 1):
            arg = arg - abs(int(arg))
        if (arg < -1):
            arg = arg + abs(int(arg))

        teta = acos(arg)

        return teta

    def find_axis(self, A, B, C):
        # find the normal of the plan
        a = (B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1])
        b = (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2])
        c = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
        axis = Vector((a, b, c))
        return axis

    def find_norm(self, A, B, C):
        # find the normal of the plan
        a = (B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1])
        b = (B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2])
        c = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
        axis = Vector((a, b, c))
        return axis

    def find_center(self, A, B, C):
        # find passing plan through A B C
        # ex+fy+hz+k=0
        N1 = self.find_norm(A, B, C);
        e = N1[0]
        f = N1[1]
        h = N1[2]
        k = -e * A[0] - f * A[1] - h * A[2]

        # find midpoints between A B and A C
        M1 = Vector(((A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2))
        M2 = Vector(((A[0] + C[0]) / 2, (A[1] + C[1]) / 2, (A[2] + C[2]) / 2))

        # find perpendicular plan through M1
        # ax+by+cz+d=0
        # v1 = Vector((B[0]-C[0] , B[1]-C[1] , B[2]-C[2]))
        # v2 = C
        # vectorial product
        # prod = Vector(( v1[1]*v2[2] - v1[2]*v2[1] , v1[2]*v2[0] - v1[0]*v2[2] , v1[0]*v2[1] - v1[1]*v2[0] ))
        # N2 = Vector((prod[0]-A[0] , prod[1]-A[1] , prod[2]-A[2]))

        # find perpendicular plan through M1
        N2 = Vector((A[0] - B[0], A[1] - B[1], A[2] - B[2]))
        a = N2[0]
        b = N2[1]
        c = N2[2]
        d = -a * M1[0] - b * M1[1] - c * M1[2]

        # find perpendicular plan through M2
        N3 = Vector((A[0] - C[0], A[1] - C[1], A[2] - C[2]))
        w = N3[0]
        r = N3[1]
        t = N3[2]
        l = -w * M2[0] - r * M2[1] - t * M2[2]

        # find the center by the linear system
        # the center belongs to all the three plans and has the same distance from A B C
        # put all these condition in one system Coeff*P = D

        a31 = 2 * A[0] - 2 * B[0]
        a32 = 2 * A[1] - 2 * B[1]
        a33 = 2 * A[2] - 2 * B[2]

        Coeff = np.array([[e, f, h], [a, b, c], [w, r, t]])

        CoeffInv = np.linalg.inv(Coeff)

        # G = - A[0]**2 - A[1]**2 - A[2]**2 + B[0]**2 + B[1]**2 + B[2]**2

        D = np.array([[-k], [-d], [-l]])

        P = np.dot(CoeffInv, D)

        center = Vector((P[0], P[1], P[2]))

        return center

    def find_individual_center(self, center, center_vtx, v1, v2):

        i_center = Vector((0, 0, 0))
        i_center[0] = center[0] + (v1[0] - center_vtx[0])
        i_center[1] = center[1] + (v1[1] - center_vtx[1])
        # i_center[2] = center[2] + (v1[2] - center_vtx[2])

        # i_center[2] = (v1[2] + v2[2]) / 2
        i_center[2] = center[2]
        return i_center

    def find_closest_point(self, pin_p, obj):
        # non so il senso, l'ho visto su internet:
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.editmode_toggle()

        mesh = obj.data

        bm = bmesh.new()
        bm.from_object(obj, bpy.context.scene)
        bm.verts.ensure_lookup_table()

        #size = len(mesh.vertices)
        size = len(bm.verts)
        kd = mathutils.kdtree.KDTree(size)

        #for i, v in enumerate(mesh.vertices):
        for i, v in enumerate(bm.verts):
            v_co = obj.matrix_world * v.co
            kd.insert(v_co, i)

        kd.balance()

        co, index, dist = kd.find(pin_p)

        bm.free()

        return co, index

    def find_distance(self, p1, p2):
        d = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
        return d

    def normalize_weights(self, weights, A ,C, v_index):

        wi = np.argmax(weights)

        wi_sim = -1

        AC = self.find_distance(A,C)

        obj = self.my_obj

        #obj = bpy.context.active_object
        #shapes_name = obj.active_shape_key.id_data.name
        #shapekey_array = bpy.data.shape_keys[shapes_name].key_blocks
        shapekey_array = obj.data.shape_keys.key_blocks

        #v1 = obj.data.vertices[v_index].co
        v1 = shapekey_array[0].data[v_index].co

        ## IF (SYMMETRY IS ON):
        if(self.symmetry_switch):
            shape_name = shapekey_array[wi].name
            if(shape_name[-2:] == ".L"):
                shape_name_simm = list(shape_name)
                shape_name_simm[-1] = "R"
                shape_name_simm = "".join(shape_name_simm)

                i=0
                for shape in shapekey_array:
                    if(shape.name == shape_name_simm):
                        wi_sim = i
                    i+=1

            if(shape_name[-2:] == ".R"):
                shape_name_simm = list(shape_name)
                shape_name_simm[-1] = "L"
                shape_name_simm = "".join(shape_name_simm)

                i=0
                for shape in shapekey_array:
                    if(shape.name == shape_name_simm):
                        wi_sim = i
                    i+=1


        #
        # apply just the max weights
        i = 0
        for shape in shapekey_array:
            if (i == wi):
                shape.value = 1
                #print("shapekey: ", shape.name)
                #print("value shape: ", weights[wi])
            else:
                shape.value = 0
            i += 1

        # # non so il senso, l'ho visto su internet:
        # bpy.ops.object.editmode_toggle()
        # bpy.ops.object.editmode_toggle()
        #
        # # create a bmesh copy
        # bm = bmesh.new()
        # bm.from_object(obj,bpy.context.scene)
        # bm.verts.ensure_lookup_table()

        #v2 = bm.verts[v_index].co
        v2 = shapekey_array[wi].data[v_index].co
       # print("v1,v2,A,C-->",v1,v2,A,C)

        ########## QUESTA FUNZINE CAUSA IL CRASH!
        # bm.free()

        v1v2 = self.find_distance(v1,v2)

        #print("AC,v1v2-->", AC, v1v2)

        if(v1v2 != 0):
            w_normalized = AC/v1v2
            # if(w_normalized > 1):
            #     weights[wi] = 1
            # else:
            #     weights[wi] = w_normalized
            weights[wi] = w_normalized
            if(wi_sim != -1):
                weights[wi_sim] = AC/v1v2

        return  weights

    def find_shape_target(self, obj, weights, seed_index, C):

        # non so il senso, l'ho visto su internet:
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.editmode_toggle()

        # shapes_name = obj.active_shape_key.id_data.name
        # shapekey_array = bpy.data.shape_keys[shapes_name].key_blocks
        shapekey_array = obj.data.shape_keys.key_blocks

        # apply all the weights
        i = 0
        for shape in shapekey_array:
            if (shape.name != "Final" and shape.name[:6] != "Target"):
                shape.mute = False
                shape.value = weights[i]
            else:
                shape.value = 0
            i += 1
        '''
        container_mesh = bpy.data.meshes.new("target")
        obj_copy = container_mesh.getFromObject(obj)
        '''
        # create a bmesh copy
        # depsgraph = bpy.context.evaluated_depsgraph_get()
        bm = bmesh.new()
        # bm.from_object(obj, depsgraph)
        # bm.from_mesh(obj.data)
        bm.from_object(obj, bpy.context.scene)
        bm.verts.ensure_lookup_table()

        ###find last Target
        shape_index = 0
        i = 0
        for shape in shapekey_array:
            name = shape.name
            if (name[:6] == "Target"):
                shape_index += 1
            i += 1
        # bpy.ops.object.shape_key_add(from_mix=False)
        # obj.data.shape_keys.key_blocks[i].name = "Target" + str(shape_index)

        # store shapetarget in the shapekey named "targeti"
        i = 0
        for v in bm.verts:
            shapekey_array["Target"+str(shape_index-1)].data[i].co = v.co
            i += 1

        seed = bm.verts[seed_index].co
        diff = C - seed


        bm.free()

        # find the index of ith target and set value 1 to display
        shape_index_general = -1
        i = 0
        for shape in shapekey_array:
            if (shape.name[:6] == "Target"):
                shape_index_general = i
                shape.value = 1
            else:
                shape.value = 0
            i += 1

        # APPLY FFD TRANSFORMATION
        if(max(weights) > 1):
            self.apply_FFD(obj, diff, shape_index_general, seed_index)

    def find_shape_target_rot(self, obj, weights, seed_index, C):


        weight = max(weights)


        shapekey_array = obj.data.shape_keys.key_blocks

        main_vertex = shapekey_array[0].data[seed_index].co

        shape_index = weights.index(weight)
        shape_to_reach = shapekey_array[shape_index]

        ###find last Target
        shape_t_index = 0
        i = 0
        for shape in shapekey_array:
            name = shape.name
            if (name[:6] == "Target"):
                shape_t_index += 1
            i += 1

        ## find intermediate shape with rotation transformation
        if(weight < 1):


            center_original = self.center_list[-1]
            center = center_original.copy()
            axis = self.axis_list[-1]

            T = mathutils.Matrix.Translation(center)
            Tinv = mathutils.Matrix.Translation(-center)

            i=0
            for v1 in shapekey_array[0].data:

                # symmetry
                if (self.symmetry[-1] == "X"):
                    if (v1.co.x * main_vertex.x < 0):
                        center.x = -center.x
                        main_vertex.x = -main_vertex.x

                v2 = shape_to_reach.data[i].co

                i_center = self.find_individual_center(center, main_vertex, v1.co, v2)

                i_axis = self.find_axis(v1.co, v2, i_center)

                teta = self.find_teta(v1.co, v2, center, 0)

                T = mathutils.Matrix.Translation(i_center)
                Tinv = mathutils.Matrix.Translation(-i_center)

                # rotation maximum angle
                Rmax = mathutils.Matrix.Rotation(teta, 4, i_axis)
                vmax = T * Rmax * Tinv * v1.co


                # find difference for traslation
                diff = (v2 - vmax) * weight

                # rotation matrix for weighted angle
                R = mathutils.Matrix.Rotation(teta * weight, 4, i_axis)


                vn = T * R * Tinv * v1.co
                vn = vn + diff

                shapekey_array["Target"+str(shape_t_index-1)].data[i].co = vn

                i+=1


        # find the index of ith target and set value 1 to display
        shape_index_general = -1
        i = 0
        for shape in shapekey_array:
            if (shape.name[:6] == "Target"):
                shape_index_general = i
                shape.value = 1
            else:
                shape.value = 0
            i += 1


        # APPLY FFD TRANSFORMATION
        if(weight > 1):

            #find the difference for FFD
            seed = shapekey_array[shape_index].data[seed_index].co
            diff = C - seed

            i=0
            for v in shape_to_reach.data:
                shapekey_array["Target" + str(shape_t_index - 1)].data[i].co = v.co
                i+=1


            self.apply_FFD(obj, diff, shape_index_general, seed_index)

    def apply_FFD(self, obj, diff, shape_index, v_index):

        bpy.context.object.active_shape_key_index = shape_index

        bpy.ops.object.mode_set(mode='EDIT')

        bpy.context.object.data.use_mirror_x = True

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        for v in bm.verts:
            v.select = False

        bm.verts[v_index].select = True

        transform = (diff[0], diff[1], diff[2])

        volume = self.find_box_volume(self.my_obj)
        height = self.find_box_height(self.my_obj)
        diag = self.find_box_diag(self.my_obj)
        #size = volume / 3309.205132779412
        #size = height / 5.797122095747182
        size = diag / 6.509033696033152

      #  print("diag:", diag, "proportional size: ",size)

        bpy.ops.transform.translate(value=transform,
                                    constraint_axis=(False, False, False),
                                    constraint_orientation='GLOBAL',
                                    mirror=True,
                                    proportional='CONNECTED',
                                    proportional_edit_falloff='SMOOTH',
                                    proportional_size=size)

        bpy.ops.object.mode_set(mode='OBJECT')

    def find_edge_distance(self, ob, v1, v2):

        bpy.ops.object.mode_set(mode='EDIT')

        # ob = bpy.context.object
        me = ob.data
        bm = bmesh.from_edit_mesh(me)

        bm.verts.ensure_lookup_table()

        bpy.ops.mesh.select_all(action='DESELECT')

        start_index = v1  # start points
        end_index = v2  # end points

        bm.verts[start_index].select = True
        bm.verts[end_index].select = True

        bpy.ops.mesh.shortest_path_select()
        verts_list = [bm.verts[start_index]]
        verts = [v for v in bm.verts if v.select]

        bpy.ops.object.mode_set(mode='OBJECT')


        return len(verts)

    ###for time interpolation
    def find_stroke_lenght(self, index_0, index_n, stroke):
        i = index_0
        lenght = 0

        bezier_index = len(stroke.data.splines)

        while (i < index_n - 1):
            p1 = stroke.data.splines[bezier_index-1].bezier_points[i].co
            p2 = stroke.data.splines[bezier_index-1].bezier_points[i + 1].co

            lenght = lenght + self.find_distance(p1, p2)

            i += 1

        return lenght

    def set_keyframes(self, stroke, delta_time, frame_start):

        # stroke = bpy.data.objects["stroke"]

        bezier_index = len(stroke.data.splines)

        stroke_points = len(stroke.data.splines[bezier_index-1].bezier_points)

        print("STOKE_POINTS: ", stroke_points)

        pos_C = stroke.data.splines[bezier_index-1].bezier_points[0].co

        # drawing time of the stroke
        # delta_time = 2

        frame_rate = bpy.context.scene.render.fps

        total_lenght = self.find_stroke_lenght(0, stroke_points, stroke)


        #obj = bpy.context.active_object
        #obj = self.my_obj

        #shapes_name = obj.active_shape_key.id_data.name
        #shapekey_array = bpy.data.shape_keys[shapes_name].key_blocks
        #shape_key = shapekey_array[1]

        shapekey_array = self.my_obj.data.shape_keys.key_blocks


        ###find last Target
        shape_index = 0
        i = 0
        for shape in shapekey_array:
            name = shape.name
            if (name[:6] == "Target"):
                shape_index += 1
            i += 1

        shape_key = shapekey_array["Target"+str(shape_index-1)]


        # set the first keyframe to 0
        shape_key.value = 0
        shape_key.keyframe_insert("value", frame=frame_start)

        # for each bezier point set a key frame
        i = 1
        last_frame = frame_start
        while(i < stroke_points + 1):
            lenght_i = self.find_stroke_lenght(0, i, stroke)

            # frame_i = round( (lenght_i/total_lenght)*delta_time*frame_rate )
            #frame_i = round( ((delta_time * frame_rate) / stroke_points)  * i) + frame_start

            # NO SUBSAMPLING:
            #frame_i = (round(delta_time * frame_rate) / stroke_points) * i + frame_start

            sub = 9

            #if(last_frame != frame_i):
            if(i%sub == 0):

                ## SUBSAMPLING:
                frame_i = (round(delta_time * frame_rate) / round(stroke_points/sub)) * i/sub + frame_start

                last_frame = frame_i

                value_i = lenght_i / total_lenght

                shape_key.value = value_i
                shape_key.keyframe_insert("value", frame=frame_i)

            ## just for subsampling
            if (i%sub!=0 and i == stroke_points):

                frame_i = (round(delta_time * frame_rate) / stroke_points) * i + frame_start

                value_i = lenght_i / total_lenght

                shape_key.value = value_i
                shape_key.keyframe_insert("value", frame=frame_i)



            i += 1

    def set_keyframes_2(self, weights, start, end):

        shapekey_array = self.my_obj.data.shape_keys.key_blocks

        #shape_name = ""
        #wi = np.argmax(weights)
        i=0
        for shape in shapekey_array:
            if(weights[i]>0):
                # shape_name = shape.name
                # set the first keyframe to 0
                shape.value = 0
                shape.keyframe_insert("value", frame=start)

                # set the last keyframe to value
                shape.value = weights[i]
                shape.keyframe_insert("value", frame=end)
            if(weights[i]==-1):
                # set the last keyframe to 0
                shape.value = 0
                shape.keyframe_insert("value", frame=end)
            i+=1

        #value = max(weights)
        #shape_key = shapekey_array[shape_name]

    def scale_keyframes(self, shape_name, value, start_frame, end_frame):

        if (value > 0.5):
            k = (0.01 - 1) / (1 - 0.5) * (value - 0.5) + 1
        if (value < 0.5):
            k = (value - 0) * ((5 - 1) / (0 - 0.5)) + 5
        if (value == 0.5):
            k = 1

        my_obj = self.my_obj

        KeyName = my_obj.data.shape_keys.name


        str1 = 'key_blocks["'
        str2 = '"].value'

        for fcurve in bpy.data.actions[KeyName + "Action"].fcurves:

            name = self.find_between_r(fcurve.data_path, str1, str2)

            window = end_frame - start_frame

            if (name == shape_name):

                ### calc interval between frames:
                count = 1
                for key_point in fcurve.keyframe_points:
                    frame = key_point.co[0]
                    if (frame > start_frame and frame < end_frame):
                        count += 1

                interval = (end_frame - start_frame) / count
                print("interval: ", interval, count)

                i = start_frame

                for key_point in fcurve.keyframe_points:
                    # x = ( key_point.co[0] - start_frame )/window
                    x = (i - start_frame) / window

                    ### EXP FUNCTION
                    y = x ** k

                    key_point.co[0] = y * window + start_frame

                    key_point.interpolation = 'LINEAR'

                    i += interval


    def fix_stroke(self, bezier_index, pos_start , pos_end, center, axis, max_weight):

        bezier = self.stroke.data.splines[bezier_index-1]

        stroke_lenght = len(bezier.bezier_points)
        mid_point = int(round(stroke_lenght / 2))

        #pos_A = bezier.bezier_points[0].co
        pos_A = pos_start
        #pos_B = bezier.bezier_points[mid_point].co
        #pos_C = bezier.bezier_points[stroke_lenght - 1].co
        pos_C = pos_end

        if(max_weight > 1):
            max_weight = 1

        teta = self.find_teta(pos_A, pos_C, center, 0)*max_weight

        T = mathutils.Matrix.Translation(center)
        Tinv = mathutils.Matrix.Translation(-center)

        j=0
        v0 = bezier.bezier_points[0]
        for v in self.stroke.data.splines[bezier_index-1].bezier_points:
            teta_j = (teta/(stroke_lenght-1))*j
            R = mathutils.Matrix.Rotation(teta_j, 4, axis)
            v.co = T * R * Tinv * v0.co
            j+=1

         #### CREATE PIN STROKE:

        pin_stroke = bpy.data.objects["Pin_stroke"]

        new_obj = pin_stroke.copy()
        new_obj.data = pin_stroke.data.copy()
        new_obj.location = bezier.bezier_points[mid_point].co

        resize = self.find_box_diag(self.my_obj) / 250
        new_obj.scale = (resize, resize, resize)

        new_obj.name = "Pin_stroke"+str(bezier_index-1)

        new_obj.show_x_ray = True
        new_obj.show_name = True
        new_obj.hide = False

        scn = bpy.context.scene
        scn.objects.link(new_obj)

        mesh = new_obj.data
        mesh.materials[0] = bpy.data.materials["yellow"]
        mesh.update()

        #### CREATE ARROW:

        arrow = bpy.data.objects["arrow"]

        new_obj_2 = arrow.copy()
        new_obj_2.data = arrow.data.copy()
        new_obj_2.location = bezier.bezier_points[-1].co

        vec = bezier.bezier_points[-1].co - bezier.bezier_points[-2].co
        DirectionVector = mathutils.Vector(vec)
        new_obj_2.rotation_quaternion = DirectionVector.to_track_quat('Z', 'Y')

        resize = self.find_box_diag(self.my_obj) / 250
        new_obj_2.scale = (resize, resize, resize)

        new_obj_2.name = "arrow_" + str(bezier_index - 1)

        new_obj_2.show_x_ray = True
        new_obj_2.show_name = True
        new_obj_2.hide = False

        scn.objects.link(new_obj_2)






    def is_symmetrical(self, obj, shape_index):

        shapekey_array = obj.data.shape_keys.key_blocks

        x_sim = False
        y_sim = False
        z_sim = False

        count_x = 0
        count_y = 0
        count_z = 0

        thresold = 10 ** (-2)

        # find the most moved vertex
        v_index = -1
        diff = 0
        i = 0
        for v in shapekey_array[shape_index].data:
            v0 = shapekey_array[0].data[i].co
            if ((v.co - v0).length > diff):
                diff = (v.co - v0).length
                v_index = i

            i += 1


        v_max = shapekey_array[shape_index].data[v_index].co

        # find the possible symmetric vertex according to xyz
        vmax_reflected_x = v_max.reflect(mathutils.Vector((1, 0, 0)))
        vmax_reflected_y = v_max.reflect(mathutils.Vector((0, 1, 0)))
        vmax_reflected_z = v_max.reflect(mathutils.Vector((0, 0, 1)))

        i = 0
        for v2 in shapekey_array[shape_index].data:

            # v2_reflected_x = v2.co.reflect(mathutils.Vector((1,0,0)))
            # v2_reflected_y = v2.co.reflect(mathutils.Vector((0,1,0)))
            # v2_reflected_z = v2.co.reflect(mathutils.Vector((0,0,1)))

            # compare just with moved vertices
            if (v2.co != shapekey_array[0].data[i].co):

                if ((v2.co - vmax_reflected_x).length < thresold):
                    count_x += 1
                if ((v2.co - vmax_reflected_y).length < thresold):
                    count_y += 1
                if ((v2.co - vmax_reflected_z).length < thresold):
                    count_z += 1

            i += 1


        if (count_x >= 1):
            x_sim = True

        if (count_y >= 1):
            y_sim = True

        if (count_z >= 1):
            z_sim = True

        return x_sim, y_sim, z_sim

    def update_stroke(self, bezier_index, diff, origin_pos_B):

        print("pin angle: ", self.pin_angle[bezier_index])

        bezier = self.stroke.data.splines[bezier_index]

        center = self.center_list[bezier_index]

        #axis = self.axis_list[bezier_index]

        stroke_lenght = len(bezier.bezier_points)

        mid_point = int(round(stroke_lenght / 2))

        pos_A = bezier.bezier_points[0].co
        #pos_B = bezier.bezier_points[mid_point].co
        pos_C = bezier.bezier_points[-1].co

        pos_B = origin_pos_B + diff

        new_center = self.find_center(pos_A, pos_B, pos_C)

        axis = self.find_axis(pos_A, pos_C, new_center)


       # print(pos_A, pos_B, pos_C, new_center)

        teta = self.find_teta(pos_A, pos_C, new_center, 0)

        T = mathutils.Matrix.Translation(new_center)
        Tinv = mathutils.Matrix.Translation(-new_center)

        j=0
        v0 = bezier.bezier_points[0]
        for v in self.stroke.data.splines[bezier_index].bezier_points:
            teta_j = (teta/(stroke_lenght-1))*j
            R = mathutils.Matrix.Rotation(teta_j, 4, axis)
            v.co = T * R * Tinv * v0.co
            j+=1

        self.center_list[bezier_index] = new_center

        self.axis_list[bezier_index] = axis

        pin_stroke = bpy.data.objects["Pin_stroke" + str(bezier_index)]
        pin_mesh = pin_stroke.data
        pin_mesh.materials[0] = bpy.data.materials["yellow"]

        # update pin position
        pin_angle = self.pin_angle[bezier_index]
        R_pin = mathutils.Matrix.Rotation(teta*pin_angle, 4, axis)
        pin_stroke.location = T * R_pin * Tinv * v0.co


        # from arc to segment
        verts = []
        pos_B = bezier.bezier_points[mid_point].co
        verts.append(pos_A)
        verts.append(pos_C)
        M = self.find_mipoint(verts)
        if(self.find_distance(pos_B,M) < 0.005):
           # print("segment!")
            j = 0
            v0 = bezier.bezier_points[0]
            for v in self.stroke.data.splines[bezier_index].bezier_points:
                trasl = (pos_C - pos_A)/(stroke_lenght-1)
                v.co = v0.co + trasl*j
                j+=1

           # update pin position
            pin_stroke.location = v0.co + (pos_C - pos_A)*pin_angle
            self.center_list[bezier_index] = "INFINITY"
            pin_mesh.materials[0] = bpy.data.materials["green"]



        #update for materials
        pin_mesh.update()

        ### UPDATE ARROW

        arrow = bpy.data.objects["arrow_"+str(bezier_index)]

        vec = bezier.bezier_points[-1].co - bezier.bezier_points[-2].co
        DirectionVector = mathutils.Vector(vec)

        arrow.rotation_quaternion = DirectionVector.to_track_quat('Z', 'Y')

    def rotate_pin(self, bezier_index, pen_pos):

        pin = bpy.data.objects["Pin_stroke" + str(bezier_index)]

        A = self.stroke.data.splines[bezier_index].bezier_points[0].co
        B = self.stroke.data.splines[bezier_index].bezier_points[-1].co
        center = self.center_list[bezier_index]

        ratio = 0.5

        if(center!="INFINITY"):

            teta_max = self.find_teta(A, B, center, 0)
            teta = self.find_teta(A, pen_pos, center, 0)
            axis = self.axis_list[bezier_index]

            if(teta > teta_max):
                teta = teta_max

            T = mathutils.Matrix.Translation(center)
            Tinv = mathutils.Matrix.Translation(-center)
            R = mathutils.Matrix.Rotation(teta, 4, axis)

            pin.location = T * R * Tinv * A

            self.pin_angle[bezier_index] = teta/teta_max

            ratio = teta/teta_max

        else:
            tot_lenght = self.find_distance(A,B)
            par_lenght = self.find_distance(pen_pos,A)
            ratio =  par_lenght/tot_lenght
            pin.location = A + (B - A)*ratio
            self.pin_angle[bezier_index] = ratio

        return ratio


    def display_pins(self, indx):
        # non so il senso, l'ho visto su internet:
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.editmode_toggle()

        #my_obj = bpy.context.scene.objects["face.001"]
        bm = bmesh.new()
        bm.from_object(self.my_obj, bpy.context.scene)
        bm.verts.ensure_lookup_table()

        shapekey_array = self.my_obj.data.shape_keys.key_blocks

        pin = bpy.context.scene.objects["Pin"]
        scn = bpy.context.scene

        #vrtx = self.my_obj.data.vertices[indx].co
        vrtx = bm.verts[indx].co

        #threshold = 0.5
        diag = self.find_box_diag(self.my_obj)
        threshold = diag / 65.09033696033151
      #  print("threshold: ", threshold)

        for shape in shapekey_array:
            if(shape.name!="Final" and shape.name[:6]!="Target"):
                if (self.find_distance(shape.data[indx].co, shapekey_array[0].data[indx].co) > threshold):
                    new_obj = pin.copy()
                    new_obj.data = pin.data.copy()
                    new_obj.location = shape.data[indx].co

                    resize = self.find_box_diag(self.my_obj) / 143.8524066485403
                    new_obj.scale = (resize, resize, resize)

                    new_obj.name = shape.name

                    new_obj.show_x_ray = True
                    new_obj.show_name = True
                    new_obj.hide = False

                    scn.objects.link(new_obj)

                    self.pin_list.append(new_obj.name)

        bm.free()

    def remove_pins(self):

        for pin in self.pin_list:
            # bpy.context.scene.objects[pin].select = True
            # bpy.ops.object.delete()
            pin_obj = bpy.context.scene.objects[pin]
            bpy.data.objects.remove(pin_obj)

        self.pin_list = []

    def get_weights_by_pin(self, pin_end, v_pos, v_index):

        weights = []

        d_from_basis = 0

        target_name = ""
        d_min = 10000000000
        for pin_name in self.pin_list:

            pin_pos = bpy.context.scene.objects[pin_name].location
            d = self.find_distance(pin_end, pin_pos)

            if(d < d_min):
                d_min = d
                target_name = pin_name

            if(pin_name == 'Basis'):
                d_from_basis = self.find_distance(v_pos, pin_pos)

      #  print("shapekey by pin: "+ target_name)

        shapekey_array = self.my_obj.data.shape_keys.key_blocks

        diag = self.find_box_diag(self.my_obj)
        threshold = diag / 65.09033696033151

        start_fr = int(bpy.data.objects["Frame_i"].data.body)
        str1 = 'key_blocks["'
        str2 = '"].value'
        KeyName = self.my_obj.data.shape_keys.name
        shape_to_be_undo = []
        try:
            for fcurve in bpy.data.actions[KeyName + "Action"].fcurves:
                if(fcurve.evaluate(start_fr) != 0):
                    shape_name = self.find_between_r(fcurve.data_path, str1, str2)
                   # print("weights not 0: ",shape_name)
                    d = self.find_distance(shapekey_array[shape_name].data[v_index].co , shapekey_array[0].data[v_index].co)
                    if(d > threshold):
                        shape_to_be_undo.append(shape_name)
        except:
            print(KeyName, " has no action yet")



        for shape in shapekey_array:
            if(shape.name == target_name):
                weights.append(1)
            elif(shape.name in shape_to_be_undo):
                weights.append(-1)
            else:
                weights.append(0)


        return weights


    def update_frame(self, trans_x):

        start_bar = bpy.data.objects["Start_frame_bar"]
        end_bar = bpy.data.objects["End_frame_bar"]
        bar = bpy.data.objects["Frame_bar"]

        ## PANEL UI
        start_UI = bpy.data.objects["Frame_start_UI"]
        end_UI = bpy.data.objects["Frame_end_UI"]
        bar_UI = bpy.data.objects["Bar_UI"]

        bar_UI.hide = False

        # frame_start = bpy.data.objects["Start_frame"]
        frame_start = bpy.data.objects["Frame_start_UI"]
        frame_start_int = int(frame_start.data.body)

        # frame_end = bpy.data.objects["End_frame"]
        frame_end = bpy.data.objects["Frame_end_UI"]
        frame_end_int = int(frame_end.data.body)

        ## translate frame

        frame_i = bpy.data.objects["Frame_i"]
        frame_i_UI = bpy.data.objects["Frame_i_UI"]

        frame_i_UI.hide = False

        time_len = self.find_distance(start_bar.location, end_bar.location)

        trans = end_bar.location - start_bar.location
        trans_UI = end_UI.location.x - start_UI.location.x

        bar.location = start_bar.location + trans / 2 + trans_x * (trans / 2)
        bar_UI.location.x = start_UI.location.x + trans_UI / 2 + trans_x * (trans_UI / 2)

        ## update frame

        frame_len = self.find_distance(start_bar.location, bar.location)

        frame_update = round((frame_len / time_len) * (frame_end_int - frame_start_int) + frame_start_int)

        frame_i.data.body = str(frame_update)
        frame_i_UI.data.body = str(frame_update)

        bpy.context.scene.frame_current = frame_update

        return frame_update

    ## UI ##########

    def add_bar(self,start_frame, end_frame, shape_name):

        frame_start_UI_obj = bpy.data.objects["Frame_start_UI"]
        frame_start_UI = int(frame_start_UI_obj.data.body)
        frame_end_UI_obj = bpy.data.objects["Frame_end_UI"]
        frame_end_UI = int(frame_end_UI_obj.data.body)
        last_bar = bpy.data.objects["bar_empty"]
        last_text = bpy.data.objects["shape_name"]

        new_bar = last_bar.copy()
        new_bar.data = last_bar.data.copy()
        ### new_bar.name
        scn = bpy.context.scene
        scn.objects.link(new_bar)

        new_bar.hide = False
        new_bar.name = self.target_list[-1] + "_bar"

        new_text = last_text.copy()
        new_text.data = last_text.data.copy()
        scn.objects.link(new_text)
        new_text.data.body = shape_name
        new_text.name = "shape_name_" + str(len(self.target_list)-1)

        n = len(self.target_list)

        new_bar.location = last_bar.location - mathutils.Vector((0, 0, 0.25*n))
        new_text.location = last_text.location - mathutils.Vector((0, 0, 0.25*n))

        if (start_frame < frame_start_UI):
            start_frame = frame_start_UI
        if (end_frame > frame_end_UI):
            end_frame = frame_end_UI

        if (start_frame >= frame_end_UI or end_frame <= frame_start_UI):
            s = 0
        else:
            time_interval = end_frame - start_frame
            full_time_interval = frame_end_UI - frame_start_UI

            s = (time_interval / full_time_interval) * 80

            new_bar.scale = (s, 1, 1)

            factor = (start_frame - frame_start_UI) / full_time_interval

            translation = (frame_start_UI_obj.location - frame_end_UI_obj.location) * factor

            new_bar.location = new_bar.location - translation

    def update_bars(self):

        i = 0
        for target in self.target_list:

            bar = bpy.data.objects[target + "_bar"]

            frame_start_UI_obj = bpy.data.objects["Frame_start_UI"]
            frame_start_UI = int(frame_start_UI_obj.data.body)
            frame_end_UI_obj = bpy.data.objects["Frame_end_UI"]
            frame_end_UI = int(frame_end_UI_obj.data.body)
            last_bar = bpy.data.objects["bar_empty"]

            start_frame = self.start_frame_list[i]
            end_frame = self.end_frame_list[i]

            if (start_frame < frame_start_UI):
                start_frame = frame_start_UI
            if (end_frame > frame_end_UI):
                end_frame = frame_end_UI
            if (start_frame >= frame_end_UI or end_frame <= frame_start_UI):
                s = 0
                bar.scale = (s, 1, 1)
                bar.hide = True
            else:
                bar.hide = False

                time_interval = end_frame - start_frame
                full_time_interval = frame_end_UI - frame_start_UI

                s = (time_interval / full_time_interval) * 80

                bar.scale = (s, 1, 1)

                bar.location = last_bar.location - mathutils.Vector((0, 0, 0.25 * (i + 1)))

                factor = (start_frame - frame_start_UI) / full_time_interval

                translation = (frame_start_UI_obj.location - frame_end_UI_obj.location) * factor
                bar.location = bar.location - translation

            i += 1

    def delete_bar(self, number):

       # print("Entratoooo")

        i = 0
        for target in self.target_list:

            if (i == number):
                bar = bpy.data.objects[target + "_bar"]
                shape_name = bpy.data.objects["shape_name_" + str(i)]
                bpy.data.objects.remove(bar)
                bpy.data.objects.remove(shape_name)

            if (i > number):
                bar = bpy.data.objects[target + "_bar"]
                bar.name = "Target"+str(i-1)+"_bar"
                shape_name = bpy.data.objects["shape_name_" + str(i)]
                shape_name.name = "shape_name_" + str(i-1)

                bar.location = bar.location + mathutils.Vector((0, 0, 0.25))
                shape_name.location = shape_name.location + mathutils.Vector((0, 0, 0.25))

            i += 1

    def move_fulltime_bar(self,d):

        full_frame_start_obj = bpy.data.objects["Full_Frame_Start"]
        full_frame_end_obj = bpy.data.objects["Full_Frame_End"]
        full_time_bar = bpy.data.objects["Full_Time_Bar"]

        full_frame_start = int(full_frame_start_obj.data.body)
        full_frame_end = int(full_frame_end_obj.data.body)

        d_max = 1 - full_time_bar.scale[0] / 16.650
        if (d > d_max):
            d = d_max

        #factor = d / self.find_distance(full_frame_start_obj.location, full_frame_end_obj.location)
        factor = d

        traslation = (full_frame_start_obj.location - full_frame_end_obj.location) * factor

        full_time_bar.location = full_frame_start_obj.location - traslation

        ## UPDDTAE PANEL

        frame_start_UI_obj = bpy.data.objects["Frame_start_UI"]
        frame_start_UI = int(frame_start_UI_obj.data.body)
        frame_end_UI_obj = bpy.data.objects["Frame_end_UI"]
        frame_end_UI = int(frame_end_UI_obj.data.body)

        start_f_new = round(factor * full_frame_end)
        end_f_new = round(frame_end_UI - (frame_start_UI - start_f_new))

        print("start: ", start_f_new, "end ", end_f_new)

        frame_start_UI_obj.data.body = str(start_f_new)
        frame_end_UI_obj.data.body = str(end_f_new)

        self.update_bars()

        ## UPDATE GREEN BAR:

        bar_UI = bpy.data.objects["Bar_UI"]
        frame_i_obj = bpy.data.objects["Frame_i_UI"]
        frame_i = int(frame_i_obj.data.body)

        if (frame_i >= start_f_new and frame_i <= end_f_new and start_f_new != end_f_new):

            factor = (frame_i - start_f_new) / (end_f_new - start_f_new)

            bar_UI.location.x = frame_start_UI_obj.location.x + (
                        frame_end_UI_obj.location.x - frame_start_UI_obj.location.x) * factor

            bar_UI.hide = False
            frame_i_obj.hide = False

        else:
            bar_UI.hide = True
            frame_i_obj.hide = True

    def scale_time(self, d):

        time_scale_obj = bpy.data.objects["Time_Scale"]
        minus = bpy.data.objects["minus"]
        plus = bpy.data.objects["plus"]

        # scale = d/find_distance(minus.location,plus.location)

        scale = d

        if (scale > 1):
            scale = 1

        traslation = plus.location - minus.location

        time_scale_obj.location = minus.location + traslation * scale

        ### UPDATE_PANEL

        frame_start_UI_obj = bpy.data.objects["Frame_start_UI"]
        frame_start_UI = int(frame_start_UI_obj.data.body)
        frame_end_UI_obj = bpy.data.objects["Frame_end_UI"]
        frame_end_UI = int(frame_end_UI_obj.data.body)

        full_frame_start_obj = bpy.data.objects["Full_Frame_Start"]
        full_frame_end_obj = bpy.data.objects["Full_Frame_End"]
        full_time_bar = bpy.data.objects["Full_Time_Bar"]

        full_frame_start = int(full_frame_start_obj.data.body)
        full_frame_end = int(full_frame_end_obj.data.body)

        mid_point = (frame_end_UI + frame_start_UI) / 2

        x = full_frame_end * scale
        start_f_new = round(mid_point - x / 2)
        end_f_new = round(mid_point + x / 2)

        if (start_f_new < 0):
            mid_point = mid_point - start_f_new
            start_f_new = round(mid_point - x / 2)
            end_f_new = round(mid_point + x / 2)
        if (end_f_new > full_frame_end):
            mid_point = mid_point - (end_f_new - full_frame_end)
            start_f_new = round(mid_point - x / 2)
            end_f_new = round(mid_point + x / 2)

        frame_start_UI_obj.data.body = str(start_f_new)
        frame_end_UI_obj.data.body = str(end_f_new)

        self.update_bars()

        ## UPDATE FULL TIME BAR

        factor = start_f_new / full_frame_end

        traslation = (full_frame_start_obj.location - full_frame_end_obj.location) * factor

        full_time_bar.location = full_frame_start_obj.location - traslation

        factor_2 = (end_f_new - start_f_new) / (full_frame_end - full_frame_start)

        scale = 16.650 * factor_2

        full_time_bar.scale = (scale, 1, 1)

        ## UPDATE GREEN BAR:

        bar_UI = bpy.data.objects["Bar_UI"]
        frame_i_obj = bpy.data.objects["Frame_i_UI"]
        frame_i = int(frame_i_obj.data.body)

        if (frame_i >= start_f_new and frame_i <= end_f_new and start_f_new != end_f_new):

            factor = (frame_i - start_f_new) / (end_f_new - start_f_new)

            print("fi: ", frame_i, "factor: ", factor)

            bar_UI.location.x = frame_start_UI_obj.location.x + (
                        frame_end_UI_obj.location.x - frame_start_UI_obj.location.x) * factor

            bar_UI.hide = False
            frame_i_obj.hide = False

        else:
            bar_UI.hide = True
            frame_i_obj.hide = True

    def truncate(self, number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper

    def update_slider(self, trans_x):

        slider = bpy.data.objects["slider"]
        slider_start = bpy.data.objects["slider_start"]
        slider_end = bpy.data.objects["slider_end"]

        trans = slider_end.location - slider_start.location

        new_loc = slider_start.location + trans / 2 + trans_x * (trans / 2)
        slider.location = new_loc

        value_1 = 0
        value_2 = 100

        full_length = self.find_distance(slider_end.location, slider_start.location)
        short_lenght = self.find_distance(slider_start.location, new_loc)

        value_i = (short_lenght / full_length) * (value_2 - value_1)

        bpy.data.objects["slider_value"].data.body = str(self.truncate(value_i, 3))
        # bpy.data.objects["slider_value"].data.body = str(value_i)

    # EXECUTION BEFORE THREADING
    def my_first_execute(self):

        bezier_index = len(self.stroke.data.splines)

        stroke_lenght = len(self.stroke.data.splines[bezier_index-1].bezier_points)
        pin_start = self.stroke.data.splines[bezier_index-1].bezier_points[0].co
        pin_end = self.stroke.data.splines[bezier_index-1].bezier_points[stroke_lenght - 1].co

        close_v_pos, close_v_index = self.find_closest_point(pin_start, self.my_obj)
        global v_index
        v_index = close_v_index
       # print("index: ", close_v_index,"position: ", close_v_pos)

        shape_start_name = "basis"
        ### hook start stroke with other end stroke
        if(bezier_index > 1):
            i=0
            for bez in self.stroke.data.splines:
                d = self.find_distance(bez.bezier_points[-1].co, pin_start)
                if(d < 0.01 and i < len(self.stroke.data.splines) - 1):
                   # print("stroke hooked "+str(i)+"th")
                    self.stroke.data.splines[-1].bezier_points[0].co = bez.bezier_points[-1].co
                    close_v_index = self.main_vertex_list[i]
                    shape_start_name = self.target_list[i]
                i+=1


        #unmute all the shapekey
        shapekey_array = self.my_obj.data.shape_keys.key_blocks
        for shape in shapekey_array:
            shape.mute = False


        ###find last Target
        shape_index = 0
        i = 0
        for shape in shapekey_array:
            name = shape.name
            if (name[:6] == "Target"):
                shape_index += 1
            i += 1
        bpy.ops.object.shape_key_add(from_mix=False)
        self.my_obj.data.shape_keys.key_blocks[i].name = "Target" + str(shape_index)

        stroke_lenght = len(self.stroke.data.splines[-1].bezier_points)
        pin_start = self.stroke.data.splines[-1].bezier_points[0].co
        pin_end = self.stroke.data.splines[-1].bezier_points[stroke_lenght - 1].co

        global weights
        weights = self.get_weights_by_pin(pin_end, close_v_pos, close_v_index)

        if(max(weights) > 0):

            ## append new target with TragetN format
            self.target_list.append("Target" + str(shape_index))

            #time = round(endTime - startTime) + 1
            time = endTime - startTime

            #initialize with a big mumber
            self.cut_frame_list.append(10**10)

            # start putting keyframes where the green bar is located
            frame_start = int(bpy.data.objects["Frame_i"].data.body)

            frame_rate = bpy.context.scene.render.fps

            #append new end frame
            self.end_frame_list.append(frame_start + round(time*frame_rate))

            # TIME INTERPOLATION WITH HAND EASING
            self.set_keyframes(self.stroke, time, frame_start)

            # extend timeline
            # end_frame = int(bpy.data.objects["End_frame"].data.body)
            # if(frame_start + (time*frame_rate) > end_frame):
            #     bpy.data.objects["End_frame"].data.body = str(frame_start + (time*frame_rate))
            #     bpy.data.objects["Frame_end_UI"].data.body = str(frame_start + (time * frame_rate))


            #append new bezier
            self.beziere_list.append(self.stroke.data.splines[-1])


            ## append new main vertex
            self.main_vertex_list.append(close_v_index)

            bezier_index = len(self.stroke.data.splines)


            # append new start frame
            self.start_frame_list.append(frame_start)


            shape_name = shapekey_array[ weights.index(max(weights)) ].name

            # add bar to panel UI
            self.add_bar(frame_start, self.end_frame_list[-1], shape_name)

            print("FRAMES:",self.start_frame_list[-1],self.end_frame_list[-1])




            #VERIFY X SIMMETRY
            shape_i = np.argmax(weights)
            #simmetry = self.is_symmetrical(self.my_obj, shape_i)
            #if(simmetry[0] == True):
            if(self.symmetry_switch):
                self.symmetry.append("X")
              #  print("SYMMETRICAL!")
            else:
                self.symmetry.append("NONE")
                #self.symmetry.append("X")


            self.remove_pins()

            ###################################self.my_second_execute(shape_start_name)

            bezier_index = len(self.stroke.data.splines)

            # unmute final shape
            shapekey_array = self.my_obj.data.shape_keys.key_blocks
            for shape in shapekey_array:
                if (shape.name == "Final"):
                    shape.mute = False
                else:
                    shape.mute = True

            stroke_lenght = len(self.stroke.data.splines[-1].bezier_points)
            pin_start = self.stroke.data.splines[-1].bezier_points[0].co
            pin_end = self.stroke.data.splines[-1].bezier_points[stroke_lenght - 1].co

            co, close_v_index = self.find_closest_point(pin_start, self.my_obj)
            # print("co, close_v_index:",co, close_v_index)

            close_v_index = self.main_vertex_list[-1]

            global weights
            # print("A:",shapekey_array[0].data[close_v_index].co)
            weights = self.normalize_weights(weights, shapekey_array[0].data[close_v_index].co, pin_end, close_v_index)
            print("normalized weight :", weights)

            # shape_start_name = "basis"
            # if(-1 in weights):
            #     shape_start_name = shapekey_array[weights.index(-1)].name
            #     index = int(shape_start_name[6:])
            #     self.cut_frame_list[index] = self.start_frame_list[-1] - 1

            if (shape_start_name != "basis"):
                index = int(shape_start_name[6:])
                self.cut_frame_list[index] = self.start_frame_list[-1] - 1

            ## append shape start
            self.shapes_at_start.append(shape_start_name)

            ## append start verts
            verts = []
            for v in shapekey_array[0].data:
                verts.append(v.co)
            self.verts_at_start.append(verts)

            shape_name = ""
            wi = np.argmax(weights)
            i = 0
            for shape in shapekey_array:
                if (i == wi):
                    shape_name = shape.name
                i += 1

            # value = max(weights)

            ##start_fr = int(bpy.data.objects["Start_frame"].data.body)
            ##end_fr = int(bpy.data.objects["End_frame"].data.body)

            ## TIME INTRPOLATION SEMPLIFIED
            ##self.set_keyframes_2(weights, start_fr, end_fr)

            ## append frames
            ##self.start_frame_list.append(start_fr)
            ## initialize end frame as infinity
            ##self.end_frame_list.append(10**10)

            # set end frame for shape to be undo
            # i = 0
            # for name in self.target_list:
            #     if (name == shape_start_name):
            #         self.end_frame_list[i] = start_fr
            #     i+=1

            # append blended shapes
            # blended_shape = self.blend_shapes(self.my_obj, start_fr)
            # self.shapes_at_start.append(blended_shape)

            shape_target_name = self.target_list[-1]

            ## append new center
            mid_point = int(round(stroke_lenght / 2))
            pos_B = self.beziere_list[-1].bezier_points[mid_point].co
            # center = self.find_center(pin_start, pos_B, shapekey_array[shape_target_name].data[close_v_index].co)

            center = self.find_center(pin_start, pos_B, pin_end)
            self.center_list.append(center)

            ## appen new axis
            axis = self.find_axis(pin_start, pin_end, center)
            self.axis_list.append(axis)

            self.find_shape_target(self.my_obj, weights, close_v_index, pin_end)
            # self.find_shape_target_rot(self.my_obj, weights, close_v_index, pin_end)

            ## append shape target with proper name
            ##self.target_list.append(shape_name)

            # self.blend_targets(self.my_obj)

            # mute all the shapekey
            # for shape in shapekey_array:
            #     if (shape.name != "Final"):
            #         shape.mute = True

            ## recalc center to fix better
            pin_end_2 = shapekey_array[shape_target_name].data[close_v_index].co
            center = self.find_center(pin_start, pos_B, pin_end_2)
            self.center_list[-1] = center
            axis = self.find_axis(pin_start, pin_end_2, center)
            self.axis_list[-1] = axis

            self.fix_stroke(bezier_index, co, pin_end_2, center, axis, 1)
            self.pin_angle.append(0.5)

            bpy.app.handlers.frame_change_pre.append(self.my_handler)

            handlers = bpy.app.handlers.frame_change_pre
            handlers.clear()
            handlers.append(self.my_handler)

            # UPDATE UI PANEL
            self.update_bars()

        else:
            polyline = bpy.data.curves['Stroke'].splines[-1]
            bpy.data.curves['Stroke'].splines.remove(polyline)



        self.allow_draw = True

        #get_weights_by_LQ().start()




    # EXECURION AFTER THREADING
    # #### NOT USED!!!!
    def my_second_execute(self, shape_start_name):

        bezier_index = len(self.stroke.data.splines)

        # unmute final shape
        shapekey_array = self.my_obj.data.shape_keys.key_blocks
        for shape in shapekey_array:
            if (shape.name == "Final"):
                shape.mute = False
            else:
                shape.mute = True


        stroke_lenght = len(self.stroke.data.splines[-1].bezier_points)
        pin_start = self.stroke.data.splines[-1].bezier_points[0].co
        pin_end = self.stroke.data.splines[-1].bezier_points[stroke_lenght - 1].co

        co, close_v_index = self.find_closest_point(pin_start, self.my_obj)
        #print("co, close_v_index:",co, close_v_index)

        close_v_index = self.main_vertex_list[-1]

        global weights
       # print("A:",shapekey_array[0].data[close_v_index].co)
        weights = self.normalize_weights(weights, shapekey_array[0].data[close_v_index].co, pin_end, close_v_index)
        print("normalized weight :", weights)


        #shape_start_name = "basis"
        # if(-1 in weights):
        #     shape_start_name = shapekey_array[weights.index(-1)].name
        #     index = int(shape_start_name[6:])
        #     self.cut_frame_list[index] = self.start_frame_list[-1] - 1

        if(shape_start_name!= "basis"):
            index = int(shape_start_name[6:])
            self.cut_frame_list[index] = self.start_frame_list[-1] - 1


        ## append shape start
        self.shapes_at_start.append(shape_start_name)

        ## append start verts
        verts = []
        for v in shapekey_array[0].data:
            verts.append(v.co)
        self.verts_at_start.append(verts)


        shape_name = ""
        wi = np.argmax(weights)
        i=0
        for shape in shapekey_array:
            if(i==wi):
                shape_name = shape.name
            i+=1

        #value = max(weights)

        ##start_fr = int(bpy.data.objects["Start_frame"].data.body)
        ##end_fr = int(bpy.data.objects["End_frame"].data.body)

        ## TIME INTRPOLATION SEMPLIFIED
        ##self.set_keyframes_2(weights, start_fr, end_fr)

        ## append frames
        ##self.start_frame_list.append(start_fr)
        ## initialize end frame as infinity
        ##self.end_frame_list.append(10**10)

        # set end frame for shape to be undo
        # i = 0
        # for name in self.target_list:
        #     if (name == shape_start_name):
        #         self.end_frame_list[i] = start_fr
        #     i+=1

        # append blended shapes
        # blended_shape = self.blend_shapes(self.my_obj, start_fr)
        # self.shapes_at_start.append(blended_shape)


        shape_target_name = self.target_list[-1]

        ## append new center
        mid_point = int(round(stroke_lenght / 2))
        pos_B = self.beziere_list[-1].bezier_points[mid_point].co
        # center = self.find_center(pin_start, pos_B, shapekey_array[shape_target_name].data[close_v_index].co)

        center = self.find_center(pin_start, pos_B, pin_end)
        self.center_list.append(center)

        ## appen new axis
        axis = self.find_axis(pin_start, pin_end, center)
        self.axis_list.append(axis)

        self.find_shape_target(self.my_obj, weights, close_v_index, pin_end)
        #self.find_shape_target_rot(self.my_obj, weights, close_v_index, pin_end)

        ## append shape target with proper name
        ##self.target_list.append(shape_name)

        # self.blend_targets(self.my_obj)

        # mute all the shapekey
        # for shape in shapekey_array:
        #     if (shape.name != "Final"):
        #         shape.mute = True

        ## recalc center to fix better
        pin_end_2 = shapekey_array[shape_target_name].data[close_v_index].co
        center = self.find_center(pin_start, pos_B, pin_end_2)
        self.center_list[-1] = center
        axis = self.find_axis(pin_start, pin_end_2, center)
        self.axis_list[-1] = axis

        self.fix_stroke(bezier_index, co, pin_end_2, center, axis, 1)
        self.pin_angle.append(0.5)

        bpy.app.handlers.frame_change_pre.append(self.my_handler)

        handlers = bpy.app.handlers.frame_change_pre
        handlers.clear()
        handlers.append(self.my_handler)

        #UPDATE UI PANEL
        self.update_bars()






    # ---------------------------------------- #
    # Main Loop
    # ---------------------------------------- #
    def loop(self, context):
        """
        Get fresh tracking data
        """
        try:
            data = self._hmd.update()
            self._eye_orientation_raw[0] = data[0]
            self._eye_orientation_raw[1] = data[2]
            self._eye_position_raw[0] = data[1]
            self._eye_position_raw[1] = data[3]


            self.setController()

            # ctrl_state contains the value of the button
            idx, ctrl_state = openvr.IVRSystem().getControllerState(self.ctrl_index_r)
            idx_l, ctrl_state_l = openvr.IVRSystem().getControllerState(self.ctrl_index_l)

            ctrl = bpy.data.objects['Controller.R']
            ctrl_l = bpy.data.objects['Controller.L']
            camera = bpy.data.objects['Camera']

            symmetry_text = bpy.data.objects["symmetry_text"]
            if(time.time() - self.symmetry_time_UI > 5):
                symmetry_text.hide = True


            ########## Right_Controller_States (PEN) ##########

            if self.state == State.IDLE:
                bpy.data.objects["Text.R"].data.body = "Idle\n" + self.objToControll + "-" + self.boneToControll

                # CONTROL PIN
                if (ctrl_state.ulButtonPressed == 4):
                    print("IDLE -> CONTROL_PIN")
                    #self.changeSelection(self.objToControll, self.boneToControll, False)


                    i=0
                    d_min = 10*10
                    for bezier in self.beziere_list:
                        stroke_lenght = len(bezier.bezier_points)
                        mid_point = int(round(stroke_lenght / 2))
                        mid_point_co = bezier.bezier_points[mid_point].co
                        d = self.find_distance(ctrl.location,mid_point_co)
                        if(d < d_min):
                            d_min = d
                            self.pin_stroke = mid_point_co.copy()
                            self.close_b_idx = i
                        i+=1

                    self.last_pen_pos = ctrl.location.copy()

                    self.state = State.CONTROL_PIN

                # ROTATE PIN
                if (ctrl_state.ulButtonPressed == 8589934592):
                    print("IDLE -> ROTATE_PIN")

                    i=0
                    d_min = 10*10
                    for bezier in self.beziere_list:
                        stroke_lenght = len(bezier.bezier_points)
                        mid_point = int(round(stroke_lenght / 2))
                        mid_point_co = bezier.bezier_points[mid_point].co
                        d = self.find_distance(ctrl.location,mid_point_co)
                        if(d < d_min):
                            d_min = d
                            self.pin_stroke = mid_point_co.copy()
                            self.close_b_idx = i
                        i+=1


                    self.state = State.ROTATE_PIN


                # # INTERACTION_LOCAL - VR_BLENDER
                # if ctrl_state.ulButtonPressed == 8589934592 and self.objToControll != "":
                #     print("IDLE -> INTERACTION LOCAL")
                #     self.state = State.INTERACTION_LOCAL
                #     self.curr_axes_r = 0
                #
                #     if self.boneToControll != "":
                #         self.diff_rot = ctrl.rotation_quaternion.inverted() * bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].matrix.to_quaternion()
                #         self.diff_loc = bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].matrix.to_translation() - ctrl.location
                #         self.initial_loc = copy.deepcopy(bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].location)
                #         bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].rotation_mode = 'XYZ'
                #         self.initial_rot = copy.deepcopy(bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].rotation_euler)
                #         bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].rotation_mode = 'QUATERNION'
                #
                #     else:
                #         self.diff_rot = ctrl.rotation_quaternion.inverted() * bpy.data.objects[self.objToControll].rotation_quaternion
                #         self.diff_loc = bpy.data.objects[self.objToControll].location - ctrl.location
                #         self.initial_loc = copy.deepcopy(bpy.data.objects[self.objToControll].location)
                #         bpy.data.objects[self.objToControll].rotation_mode = 'XYZ'
                #         self.initial_rot = copy.deepcopy(bpy.data.objects[self.objToControll].rotation_euler)
                #         bpy.data.objects[self.objToControll].rotation_mode = 'QUATERNION'

                # DRAWING SKETCHES - POSING_SKETCHES
                #if ctrl_state.ulButtonPressed == 8589934592:
                if ctrl_state.ulButtonPressed == 4294967296:
                    if(self.allow_draw):
                        print("IDLE -> DRAWING")

                        self.stroke.data.bevel_depth = self.find_box_diag(self.my_obj) / 650.9033696033151

                        co, indx = self.find_closest_point(bpy.data.objects['Controller.R'].location, self.my_obj)
                        self.display_pins(indx)

                        global startTime
                        startTime = time.time()
                        self.add_spline(bpy.data.objects['Controller.R'].location)
                        self.state = State.DRAWING

                # #TRACKPAD.R
                # if ctrl_state.ulButtonPressed == 4294967296:
                #     self.state = State.TRACKPAD_BUTTON_DOWN

                # NAVIGATION
                if ctrl_state.ulButtonPressed == 2:
                    print("IDLE -> NAVIGATION")
                    self.state = State.NAVIGATION_ENTER

            elif self.state == State.DRAWING:
                self.update_curve(bpy.data.objects['Controller.R'].location)

                # if (ctrl_state.ulButtonPressed != 8589934592):
                if (ctrl_state.ulButtonPressed != 4294967296):

                    self.allow_draw = False
                    global endTime
                    endTime = time.time()
                    self.my_first_execute()
                    self.state = State.IDLE

            elif self.state == State.TRACKPAD_BUTTON_DOWN:

                if ctrl_state.ulButtonPressed != 4294967296:
                    x, y = ctrl_state.rAxis[0].x, ctrl_state.rAxis[0].y
                    # Apply rotation for X setup otherwiise + setup
                    x1, y1 = x * 0.707 - y * -0.707, x * -0.707 + y * 0.707
                    x, y = x1, y1

                    if x > 0 and y > 0:
                        print('UP')
                        #print('LAUNCH ALGORITHM')
                        #softAss_detAnnealing_3().start()

                    if x > 0 and y < 0:
                        print ('RIGHT')

                    if x < 0 and y > 0:
                        for i in range (0, len(bpy.data.curves['Stroke'].splines)):
                            self.remove_spline()
                        print ('LEFT')
                    if x < 0 and y < 0:
                        print ('DOWN')
                        self.remove_spline()

                    self.state = State.IDLE

            elif self.state == State.CONTROL_PIN:

                diff = ctrl.location - self.last_pen_pos

                #print("B: ", self.pin_stroke, " last_pen: ", self.last_pen_pos, " diff: ",diff )


                self.update_stroke(self.close_b_idx, diff, self.pin_stroke)



                # print("Decisional")
                # bpy.data.objects["Text.R"].data.body = "Selection\n " + self.objToControll + "-" + self.boneToControll
                #
                #
                # # Compute the nearest object
                # try:
                #     self.objToControll, self.boneToControll = self.getClosestItem(True)
                # except:
                #     print("Error during selection")
                # global currObject
                # global currBone
                # currObject = self.objToControll
                # currBone = self.boneToControll
                # print("Current obj:", self.objToControll, self.boneToControll)

                if ctrl_state.ulButtonPressed != 4:
                    # print("touch button released")
                    self.changeSelection(self.objToControll, self.boneToControll, True)

                    self.state = State.IDLE

            elif self.state == State.ROTATE_PIN:

                if (ctrl_state.ulButtonPressed == 8589934592):

                    ratio = self.rotate_pin(self.close_b_idx, ctrl.location)

                    self.scale_keyframes(self.target_list[self.close_b_idx],ratio,self.start_frame_list[self.close_b_idx], self.end_frame_list[self.close_b_idx
                    ])

                else:
                    print("ROTATE_PIN -> IDLE")

                    self.state = State.IDLE

            elif self.state == State.INTERACTION_LOCAL:
                bpy.data.objects["Text.R"].data.body = "Interaction\n" + self.objToControll + "-" + self.boneToControll + "\n" + self.axes[self.curr_axes_r]

                ## Controll object scale
                if self.objToControll == self.objToControll_l and self.boneToControll == self.boneToControll_l and ctrl_state_l.ulButtonPressed == 8589934592:
                    if self.boneToControll != "":
                        self.initial_scale = copy.deepcopy(bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale)
                    else:
                        self.initial_scale = copy.deepcopy(bpy.data.objects[self.objToControll].scale)

                    self.diff_distance = self.computeTargetObjDistance("Controller.L", "", True)


                    self.state = State.SCALING
                    self.state_l = StateLeft.SCALING

                else:
                    if self.boneToControll != "":
                        ## The object to move is a bone
                        bone = bpy.data.objects[self.objToControll]
                        pbone = bone.pose.bones[self.boneToControll]
                        scale = copy.deepcopy(pbone.scale)
                        translationMatrix = Matrix(((0.0, 0.0, 0.0, self.diff_loc[0]),
                                                    (0.0, 0.0, 0.0, self.diff_loc[1]),
                                                    (0.0, 0.0, 0.0, self.diff_loc[2]),
                                                    (0.0, 0.0, 0.0, 1.0)))
                        diff_rot_matr = self.diff_rot.to_matrix()
                        pbone.matrix = (ctrl.matrix_world + translationMatrix) * diff_rot_matr.to_4x4()
                        pbone.scale = scale

                        self.applyConstraint(True)


                    else:
                        ## The object to move is a mesh
                        bpy.data.objects[
                            self.objToControll].rotation_quaternion = ctrl.rotation_quaternion * self.diff_rot
                        bpy.data.objects[self.objToControll].location = ctrl.location + self.diff_loc

                        self.applyConstraint(True)

                if (ctrl_state.ulButtonPressed == 8589934596):
                    print("INTERACTION_LOCAL -> CHANGE_AXIS")
                    self.state = State.CHANGE_AXES

                if (ctrl_state.ulButtonPressed != 8589934592 and ctrl_state.ulButtonPressed != 8589934596):
                    # print("grillet released")
                    self.state = State.IDLE

            elif self.state == State.CHANGE_AXES:
                if (ctrl_state.ulButtonPressed == 8589934592):
                    self.curr_axes_r += 1
                    if self.curr_axes_r >= len(self.axes):
                        self.curr_axes_r = 0
                    self.curr_axes_l = 0
                    self.state = State.INTERACTION_LOCAL

                if (ctrl_state.ulButtonPressed == 0):
                    self.state = State.IDLE

            elif self.state == State.NAVIGATION_ENTER:
                bpy.data.objects["Text.R"].data.body = "Navigation\n "
                bpy.data.objects["Text.L"].data.body = "Navigation\n "
                if ctrl_state.ulButtonPressed != 2:
                    #bpy.data.textures['Texture.R'].image = bpy.data.images['Nav-R.png']
                    #bpy.data.textures['Texture.L'].image = bpy.data.images['Hand-L.png']
                    self.state = State.NAVIGATION
                    #self.state_l = StateLeft.NAVIGATION

            elif self.state == State.NAVIGATION_EXIT:
                if ctrl_state.ulButtonPressed != 2:
                    print("NAVIGATION -> IDLE")
                    bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-idle.png"]
                    #bpy.data.textures['Texture.R'].image = bpy.data.images['Perf-R.png']
                    #bpy.data.textures['Texture.L'].image = bpy.data.images['Ctrl-L.png']
                    self.state = State.IDLE
                    #self.state_l = StateLeft.IDLE

            elif self.state == State.NAVIGATION:
                if ctrl_state.ulButtonPressed == 4294967296:
                    x, y = ctrl_state.rAxis[0].x, ctrl_state.rAxis[0].y
                    if (x > -0.3 and x < 0.3 and y < -0.8):
                        print("ZOOM_OUT")
                        camObjDist = bpy.data.objects["Origin"].location - camera.location
                        if self.objToControll != "":
                            camObjDist = bpy.data.objects[self.objToControll].location - camera.location
                        camera.location -= camObjDist

                        scale_factor = camera.scale[0]
                        scale_factor = scale_factor * 2
                        camera.scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.R"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.L"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        self.zoom = scale_factor
                        self.state = State.ZOOM_IN

                    if (x > -0.3 and x < 0.3 and y > 0.8):
                        print("ZOOM_IN")
                        camObjDist = bpy.data.objects["Origin"].location - camera.location
                        if self.objToControll != "":
                            camObjDist = bpy.data.objects[self.objToControll].location - camera.location
                        camObjDist = camObjDist / 2
                        camera.location += camObjDist

                        scale_factor = camera.scale[0]
                        scale_factor = scale_factor / 2
                        camera.scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.R"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.L"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        self.zoom = scale_factor
                        self.state = State.ZOOM_OUT

                if (ctrl_state.ulButtonPressed == 8589934592):
                    print('Camera rot: ', camera.rotation_quaternion)
                    self.diff_rot = ctrl.rotation_quaternion.inverted() * camera.rotation_quaternion
                    print('Diff:       ', self.diff_rot)
                    # self.diff_loc = camera.location - ctrl.location
                    self.diff_trans_matrix = bpy.data.objects['Camera'].matrix_world * bpy.data.objects[
                        'Origin'].matrix_world

                    self.rotFlag = False
                    self.state = State.CAMERA_ROT_CONT

                if (ctrl_state.ulButtonPressed == 4):
                    self.diff_loc = copy.deepcopy(ctrl.location)
                    self.state = State.CAMERA_MOVE_CONT

                if (ctrl_state.ulButtonPressed == 2):
                    self.state = State.NAVIGATION_EXIT

            elif self.state == State.ZOOM_IN:
                if (ctrl_state.ulButtonPressed != 4294967296):
                    self.state = State.NAVIGATION

            elif self.state == State.ZOOM_OUT:
                if (ctrl_state.ulButtonPressed != 4294967296):
                    self.state = State.NAVIGATION

            elif self.state == State.CAMERA_MOVE_CONT:
                camera.location = camera.location + (self.diff_loc - ctrl.location)

                if ctrl_state.ulButtonPressed != 4:
                    self.state = State.NAVIGATION

            elif self.state == State.CAMERA_ROT_CONT:

                if (ctrl_state.ulButtonPressed != 8589934592):
                    self.rotFlag = True
                    self.state = State.NAVIGATION

            elif self.state == State.SCALING:
                currDist = self.computeTargetObjDistance("Controller.L", "", True)
                offset = (currDist - self.diff_distance) / 10
                if self.boneToControll != "":
                    bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale = self.initial_scale
                    bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale[0] += offset
                    bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale[1] += offset
                    bpy.data.objects[self.objToControll].pose.bones[self.boneToControll].scale[2] += offset

                else:
                    bpy.data.objects[self.objToControll].scale = self.initial_scale
                    bpy.data.objects[self.objToControll].scale[0] += offset
                    bpy.data.objects[self.objToControll].scale[1] += offset
                    bpy.data.objects[self.objToControll].scale[2] += offset


                # Exit from Scaling state
                if (ctrl_state.ulButtonPressed != 8589934592):
                    if (ctrl_state_l.ulButtonPressed != 8589934592):
                        self.state = State.IDLE
                        self.state_l = StateLeft.IDLE



            ########## Left_Controller_States ##########


            if self.state_l == StateLeft.IDLE:
                bpy.data.objects["Text.L"].data.body = "Idle\n" + self.objToControll_l + "-" + self.boneToControll_l

                #TRACKPAD.L
                if ctrl_state_l.ulButtonPressed == 4294967296:
                    self.state_l = StateLeft.TRACKPAD_BUTTON_DOWN

                ## TIMELINE NAVIGATION
                if ctrl_state_l.ulButtonPressed == 4294967296:
                    x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y
                    print(x,y)

                # DECISIONAL
                if (ctrl_state_l.ulButtonPressed == 4):
                    # print("IDLE -> DECISIONAL")
                    # print ("DECISIONAL ENTER: ", self.objToControll_l, self.boneToControll_l)
                    # self.changeSelection(self.objToControll_l, self.boneToControll_l, False)
                    # self.state_l = StateLeft.DECISIONAL

                    mw = bpy.data.objects["Panel_UI"].matrix_world

                    shift_ob = bpy.data.objects["Full_Time_Bar"]
                    scale_obj = bpy.data.objects["Time_Scale"]

                    d1 = self.find_distance(ctrl_l.location, mw * shift_ob.location)
                    d2 = self.find_distance(ctrl_l.location, mw * scale_obj.location)

                    if(d1 <= d2):
                        print("IDLE -> SHIFT_TIME")
                        self.state_l = StateLeft.SHIFT_TIME
                    else:
                        print("IDLE -> SCALE_TIME")
                        self.state_l = StateLeft.SCALE_TIME

                # INTERACTION_LOCAL
                if (ctrl_state_l.ulButtonPressed == 8589934592 and self.objToControll_l != ""):
                    print("IDLE -> INTERACTION LOCAL")
                    self.state_l = StateLeft.INTERACTION_LOCAL
                    self.curr_axes_l = 0

                    if self.boneToControll_l != "":
                        self.diff_rot_l = ctrl_l.rotation_quaternion.inverted() * bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].matrix.to_quaternion()
                        self.diff_loc_l = bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].matrix.to_translation() - ctrl_l.location
                        self.initial_loc_l = copy.deepcopy(bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].location)
                        bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].rotation_mode = 'XYZ'
                        self.initial_rot_l = copy.deepcopy(bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].rotation_euler)
                        bpy.data.objects[self.objToControll_l].pose.bones[self.boneToControll_l].rotation_mode = 'QUATERNION'

                    else:
                        self.diff_rot_l = ctrl_l.rotation_quaternion.inverted() * bpy.data.objects[self.objToControll_l].rotation_quaternion
                        self.diff_loc_l = bpy.data.objects[self.objToControll_l].location - ctrl_l.location
                        self.initial_loc_l = copy.deepcopy(bpy.data.objects[self.objToControll_l].location)
                        bpy.data.objects[self.objToControll_l].rotation_mode = 'XYZ'
                        self.initial_rot_l = copy.deepcopy(bpy.data.objects[self.objToControll_l].rotation_euler)
                        bpy.data.objects[self.objToControll_l].rotation_mode = 'QUATERNION'

                # NAVIGATION
                if ctrl_state_l.ulButtonPressed == 2:
                    print("IDLE -> NAVIGATION")
                    self.state_l = StateLeft.NAVIGATION_ENTER
                    bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-navigation.png"]



                # DRAG PANEL
                if ctrl_state_l.ulButtonPressed == 8589934592:
                    print("IDEAL -> DRAG_PANEL")
                    bpy.data.objects["Text.L"].data.body = "Drag Panel"
                    self.state_l = StateLeft.DRAG_PANEL

            elif self.state_l == StateLeft.TRACKPAD_BUTTON_DOWN:

                if ctrl_state_l.ulButtonPressed != 4294967296:
                    x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y
                    # Apply rotation for X setup otherwiise + setup
                    x1, y1 = x * 0.707 - y * -0.707, x * -0.707 + y * 0.707
                    x, y = x1, y1

                    if x < 0 and y > 0:
                        # for i in range (0, len(bpy.data.curves['Stroke'].splines)):
                        #     self.remove_spline()
                        print ('LEFT')
                        self.remove_spline()
                    if x < 0 and y < 0:
                        print ('DOWN')
                        #self.remove_spline()
                        #self.remove_closest_spline(bpy.data.objects['Controller.L'].location)
                        symmetry_text.hide = False
                        self.symmetry_time_UI = time.time()
                        if(self.symmetry_switch):
                            self.symmetry_switch = False
                            symmetry_text.data.body = "Sym OFF"
                        else:
                            self.symmetry_switch = True
                            symmetry_text.data.body = "Sym ON"

                    self.state_l = StateLeft.IDLE

                    if x > 0 and y > 0:
                        print('UP')
                        #softAss_detAnnealing_3().start()
                        #self.my_first_execute()
                        print("TIDLE -> TIMELINE")

                        # DOUBLE TIMELINE (NOT NECESSARY)
                        # bpy.data.objects["Timeline"].hide = False
                        # bpy.data.objects["End_frame"].hide = False
                        # bpy.data.objects["End_frame_bar"].hide = False
                        # bpy.data.objects["Frame_bar"].hide = False
                        # bpy.data.objects["Frame_i"].hide = False
                        # bpy.data.objects["Start_frame"].hide = False
                        # bpy.data.objects["Start_frame_bar"].hide = False

                        self.state_l = StateLeft.TIMELINE
                        bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-timeline.png"]
                        bpy.data.objects["Text.L"].data.body = "Timeline"

                    if x > 0 and y < 0:
                        print ('RIGHT')
                        # global thread_is_done
                        # if(thread_is_done):
                        #     self.my_second_execute()
                            #bpy.ops.screen.animation_play()
                        print("IDLE -> THRESHOLD")
                        self.state_l = StateLeft.THRESHOLD
                        bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-threshold.png"]

                        # SHOW SLIDER
                        bpy.data.objects["slider"].hide = False
                        bpy.data.objects["slider_bar"].hide = False
                        bpy.data.objects["slider_value"].hide = False

            elif self.state_l == StateLeft.INTERACTION_LOCAL:
                bpy.data.objects["Text.L"].data.body = "Interaction\n" + self.objToControll_l + "-" + self.boneToControll_l + "\n" + self.axes[self.curr_axes_l]


                if self.objToControll == self.objToControll_l \
                        and self.boneToControll == self.boneToControll_l \
                        and ctrl_state.ulButtonPressed == 8589934592\
                        and self.state != State.CAMERA_ROT_CONT \
                        and self.state != State.NAVIGATION:
                    self.state_l = StateLeft.SCALING
                    self.state = State.SCALING

                else:
                    if self.boneToControll_l != "":
                        ## The object to move is a bone
                        bone = bpy.data.objects[self.objToControll_l]
                        pbone = bone.pose.bones[self.boneToControll_l]
                        scale = copy.deepcopy(pbone.scale)
                        translationMatrix = Matrix(((0.0, 0.0, 0.0, self.diff_loc_l[0]),
                                                    (0.0, 0.0, 0.0, self.diff_loc_l[1]),
                                                    (0.0, 0.0, 0.0, self.diff_loc_l[2]),
                                                    (0.0, 0.0, 0.0, 1.0)))
                        diff_rot_matr = self.diff_rot_l.to_matrix()
                        pbone.matrix = (ctrl_l.matrix_world + translationMatrix) * diff_rot_matr.to_4x4()
                        pbone.scale = scale
                        self.applyConstraint(False)

                    else:
                        ## The object to move is a mesh
                        bpy.data.objects[self.objToControll_l].rotation_quaternion = ctrl_l.rotation_quaternion * self.diff_rot_l
                        bpy.data.objects[self.objToControll_l].location = ctrl_l.location + self.diff_loc_l
                        self.applyConstraint(False)


                if (ctrl_state_l.ulButtonPressed==8589934596):
                    print("INTERACTION_LOCAL -> CHANGE_AXIS")
                    self.state_l = StateLeft.CHANGE_AXES

                if (ctrl_state_l.ulButtonPressed != 8589934592 and ctrl_state_l.ulButtonPressed != 8589934596):
                    self.state_l = StateLeft.IDLE

            elif self.state_l == StateLeft.NAVIGATION_ENTER:
                bpy.data.objects["Text.R"].data.body = "Navigation\n "
                bpy.data.objects["Text.L"].data.body = "Navigation\n "
                if ctrl_state_l.ulButtonPressed != 2:
                    #bpy.data.textures['Texture.R'].image = bpy.data.images['Nav-R.png']
                    #bpy.data.textures['Texture.L'].image = bpy.data.images['Hand-L.png']
                    #self.state = State.NAVIGATION
                    self.state_l = StateLeft.NAVIGATION

            elif self.state_l == StateLeft.NAVIGATION_EXIT:
                if ctrl_state_l.ulButtonPressed != 2:
                    print("NAVIGATION -> IDLE")
                    bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-idle.png"]
                    #bpy.data.textures['Texture.R'].image = bpy.data.images['Perf-R.png']
                    #bpy.data.textures['Texture.L'].image = bpy.data.images['Ctrl-L.png']
                    #self.state = State.IDLE
                    self.state_l = StateLeft.IDLE

            elif self.state_l == StateLeft.NAVIGATION:
                if (ctrl_state_l.ulButtonPressed == 4294967296):
                    x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y
                    print (x,y)

                if ctrl_state_l.ulButtonPressed == 4294967296:
                    x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y
                    if (x > -0.3 and x < 0.3 and y < -0.8):
                        print("ZOOM_OUT")
                        camObjDist = bpy.data.objects["Origin"].location - camera.location
                        if self.objToControll != "":
                            camObjDist = bpy.data.objects[self.objToControll].location - camera.location
                        camera.location -= camObjDist

                        scale_factor = camera.scale[0]
                        scale_factor = scale_factor * 2
                        camera.scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.R"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.L"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        self.zoom = scale_factor
                        self.state_l = StateLeft.ZOOM_IN

                    if (x > -0.3 and x < 0.3 and y > 0.8):
                        print("ZOOM_IN")
                        camObjDist = bpy.data.objects["Origin"].location - camera.location
                        if self.objToControll != "":
                            camObjDist = bpy.data.objects[self.objToControll].location - camera.location
                        camObjDist = camObjDist / 2
                        camera.location += camObjDist

                        scale_factor = camera.scale[0]
                        scale_factor = scale_factor / 2
                        camera.scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.R"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        bpy.data.objects["Text.L"].scale = Vector((scale_factor, scale_factor, scale_factor))
                        self.zoom = scale_factor
                        self.state_l = StateLeft.ZOOM_OUT

                if (ctrl_state_l.ulButtonPressed == 8589934592):
                    print('Camera rot: ', camera.rotation_quaternion)
                    self.diff_rot = ctrl_l.rotation_quaternion.inverted() * camera.rotation_quaternion
                    print('Diff:       ', self.diff_rot)
                    # self.diff_loc = camera.location - ctrl.location
                    self.diff_trans_matrix = bpy.data.objects['Camera'].matrix_world * bpy.data.objects[
                        'Origin'].matrix_world

                    self.rotFlag = False
                    self.state_l = StateLeft.CAMERA_ROT_CONT

                if (ctrl_state_l.ulButtonPressed == 4):
                    self.diff_loc = copy.deepcopy(ctrl_l.location)
                    self.state_l = StateLeft.CAMERA_MOVE_CONT

                if (ctrl_state_l.ulButtonPressed == 2):
                    self.state_l = StateLeft.NAVIGATION_EXIT

            elif self.state_l == StateLeft.ZOOM_IN:
                if (ctrl_state_l.ulButtonPressed != 4294967296):
                    self.state_l = StateLeft.NAVIGATION

            elif self.state_l == StateLeft.ZOOM_OUT:
                if (ctrl_state_l.ulButtonPressed != 4294967296):
                    self.state_l = StateLeft.NAVIGATION

            elif self.state_l == StateLeft.CAMERA_MOVE_CONT:
                camera.location = camera.location + (self.diff_loc - ctrl_l.location)

                if ctrl_state_l.ulButtonPressed != 4:
                    self.state_l = StateLeft.NAVIGATION

            elif self.state_l == StateLeft.CAMERA_ROT_CONT:

                if (ctrl_state_l.ulButtonPressed != 8589934592):
                    self.rotFlag = True
                    self.state_l = StateLeft.NAVIGATION

            elif self.state_l == StateLeft.CHANGE_AXES:
                if (ctrl_state_l.ulButtonPressed==8589934592):
                    self.curr_axes_l+=1
                    if self.curr_axes_l>=len(self.axes):
                        self.curr_axes_l=0
                    self.curr_axes_r=0
                    print(self.curr_axes_l)
                    print("CHANGE_AXIS -> INTERACTION_LOCAL")
                    self.state_l = StateLeft.INTERACTION_LOCAL

                if (ctrl_state_l.ulButtonPressed==0):
                    # print("grillet released")
                    self.state_l = StateLeft.IDLE

            elif self.state_l == StateLeft.SCALING:

                # Exit from Scaling state
                if (ctrl_state_l.ulButtonPressed != 8589934592):
                    if (ctrl_state.ulButtonPressed != 8589934592):
                        self.state = State.IDLE
                        self.state_l = StateLeft.IDLE

            elif self.state_l == StateLeft.SHIFT_TIME:
                bpy.data.objects["Text.L"].data.body = "Selection\n" + self.objToControll_l + "-" + self.boneToControll_l

                # Compute the nearest object
                # self.objToControll_l, self.boneToControll_l = self.getClosestItem(False)
                # global currObject_l
                # global currBone_l
                # currObject_l = self.objToControll_l
                # currBone_l = self.boneToControll_l
                # print("Current obj:", self.objToControll_l, self.boneToControll_l)

                UI_mw = bpy.data.objects["Panel_UI"].matrix_world

                start = UI_mw * bpy.data.objects["Full_Frame_Start"].location
                end = UI_mw * bpy.data.objects["Full_Frame_End"].location
                a = self.find_distance(end,start)
                b = self.find_distance(ctrl_l.location,start)

                alpha = self.find_teta(start, end, ctrl_l.location, 0)

                c = self.find_distance(ctrl_l.location, end)

                d = b/a

                full_time_bar = bpy.data.objects["Full_Time_Bar"]
                d = d - (full_time_bar.scale[0] / 16.650) / 2

                if(d < 0):
                    d=0

                if(alpha < math.pi/2):
                    if(c > a):
                        d=0

                self.move_fulltime_bar(d)

                if ctrl_state_l.ulButtonPressed != 4:
                    self.changeSelection(self.objToControll_l, self.boneToControll_l, True)
                    self.state_l = StateLeft.IDLE
                    print ("SHIFT_TIME -> IDLE")
                    bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-idle.png"]

            elif self.state_l == StateLeft.SCALE_TIME:

                UI_mw = bpy.data.objects["Panel_UI"].matrix_world

                start = UI_mw * bpy.data.objects["minus"].location
                end = UI_mw * bpy.data.objects["plus"].location
                a = self.find_distance(end,start)
                b = self.find_distance(ctrl_l.location,start)

                alpha = self.find_teta(start, end, ctrl_l.location, 0)

                c = self.find_distance(ctrl_l.location,end)

                d = b/a

                if (alpha < math.pi / 2):
                    if(c > a):
                        d=0

                self.scale_time(d)

                if ctrl_state_l.ulButtonPressed != 4:
                    self.changeSelection(self.objToControll_l, self.boneToControll_l, True)
                    self.state_l = StateLeft.IDLE
                    print ("SCALE_TIME -> IDLE")
                    bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-idle.png"]

            elif self.state_l == StateLeft.TIMELINE_ENTER:

                Panel_UI = bpy.data.objects["Panel_UI"]
                Empty = bpy.data.objects["Panel_empty"]
                #Panel_UI.location = ctrl_l.location
                Panel_UI.matrix_world = Empty.matrix_world



                if ctrl_state_l.ulButtonPressed != 8589934592:
                    print("TIMELINE_ENTER -> TIMELINE")
                    self.state_l = StateLeft.TIMELINE
                    bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-timeline.png"]
                    bpy.data.objects["Text.L"].data.body = "Timeline"

            elif self.state_l == StateLeft.TIMELINE:

                x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y

                if(x!=0 or y!=0):
                    if(y > -0.5):
                        self.update_frame(x)


                if ctrl_state_l.ulButtonPressed == 4294967296:

                    # TIMELINE EXIT
                    if(y < -0.5):
                        bpy.data.objects["Timeline"].hide = True
                        bpy.data.objects["End_frame"].hide = True
                        bpy.data.objects["End_frame_bar"].hide = True
                        bpy.data.objects["Frame_bar"].hide = True
                        bpy.data.objects["Frame_i"].hide = True
                        bpy.data.objects["Start_frame"].hide = True
                        bpy.data.objects["Start_frame_bar"].hide = True

                        print("TIMELINE -> TIMELINE_EXIT")
                        bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-idle.png"]
                        self.state_l = StateLeft.TIMELINE_EXIT

                    # START_FRAME
                    if(x < 0 and y > -0.5):
                        self.next_state_l = StateLeft.START_FRAME
                    # END_FRAME
                    if(x > 0 and y > -0.5):
                        self.next_state_l = StateLeft.END_FRAME




                if ctrl_state_l.ulButtonPressed != 4294967296:

                     if(self.next_state_l == StateLeft.START_FRAME):

                         bpy.data.objects["wheel"].hide = False

                         strfr = bpy.data.objects["Start_frame"]
                         strfr.scale = strfr.scale * 3

                         print("TIMELINE -> START_FRAME")
                         self.state_l = StateLeft.START_FRAME
                         bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-frames.png"]
                         bpy.data.objects["Text.L"].data.body = "Start Frame"

                     if (self.next_state_l == StateLeft.END_FRAME):

                         bpy.data.objects["wheel"].hide = False

                         endfr = bpy.data.objects["End_frame"]
                         endfr.scale = endfr.scale * 3

                         print("TIMELINE -> END_FRAME")
                         self.state_l = StateLeft.END_FRAME
                         bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-frames.png"]
                         bpy.data.objects["Text.L"].data.body = "End Frame"

            elif self.state_l == StateLeft.TIMELINE_EXIT:
                if ctrl_state_l.ulButtonPressed != 4294967296:
                    print("TIMELINE_EXIT -> IDLE")
                    bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-idle.png"]
                    self.state_l = StateLeft.IDLE

            elif self.state_l == StateLeft.START_FRAME:

                x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y

                wheel = bpy.data.objects["wheel"]


                if(self.y_pre != 0):

                    # increment:
                    if (y > self.y_pre and self.y_pre > self.y_pre_10):
                        move = y * 10 - self.y_pre_10 * 10
                        wheel.rotation_euler.rotate_axis("Y", math.radians(-10))
                        if (move > 4):
                            ## increment frame +1
                            frame = int(bpy.data.objects["Start_frame"].data.body)
                            frame += 1
                            bpy.data.objects["Start_frame"].data.body = str(frame)
                            bpy.data.objects["Frame_start_UI"].data.body = str(frame)

                            self.update_bars()

                            self.y_pre_10 = y

                    # decrement:
                    if (y < self.y_pre and self.y_pre < self.y_pre_10):
                        move = self.y_pre_10 * 10 - y * 10
                        wheel.rotation_euler.rotate_axis("Y", math.radians(10))
                        if (move > 4):
                            ## increment frame -1
                            frame = int(bpy.data.objects["Start_frame"].data.body)
                            frame -= 1
                            bpy.data.objects["Start_frame"].data.body = str(frame)
                            bpy.data.objects["Frame_start_UI"].data.body = str(frame)

                            self.update_bars()

                            self.y_pre_10 = y

                else:
                    self.y_pre_10 = y

                self.y_pre = y




                if ctrl_state_l.ulButtonPressed == 4294967296:

                    # TIMELINE
                    if (x < 0):
                        self.next_state_l = StateLeft.TIMELINE



                if ctrl_state_l.ulButtonPressed != 4294967296:

                    if(self.next_state_l == StateLeft.TIMELINE):
                        strfr = bpy.data.objects["Start_frame"]
                        strfr.scale = strfr.scale / 3

                        print("START_FRAME --> TIMELINE")
                        self.state_l = StateLeft.TIMELINE
                        bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-timeline.png"]
                        bpy.data.objects["Text.L"].data.body = "Timeline"

                        bpy.data.objects["wheel"].hide = True

                    if (self.next_state_l == StateLeft.END_FRAME):
                        strfr = bpy.data.objects["Start_frame"]
                        strfr.scale = strfr.scale / 3

                        endfr = bpy.data.objects["End_frame"]
                        endfr.scale = endfr.scale * 3

                        print("START_FRAME --> END_FRAME")
                        self.state_l = StateLeft.END_FRAME

            elif self.state_l == StateLeft.END_FRAME:

                x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y

                wheel = bpy.data.objects["wheel"]

                if (self.y_pre != 0):

                    # increment:
                    if (y > self.y_pre and self.y_pre > self.y_pre_10):
                        move = y * 10 - self.y_pre_10 * 10
                        wheel.rotation_euler.rotate_axis("Y", math.radians(-10))
                        if (move > 4):
                            ## increment frame +1
                            frame = int(bpy.data.objects["End_frame"].data.body)
                            frame += 1
                            bpy.data.objects["End_frame"].data.body = str(frame)
                            bpy.data.objects["Frame_end_UI"].data.body = str(frame)

                            self.update_bars()

                            self.y_pre_10 = y

                    # decrement:
                    if (y < self.y_pre and self.y_pre < self.y_pre_10):
                        move = self.y_pre_10 * 10 - y * 10
                        wheel.rotation_euler.rotate_axis("Y", math.radians(10))
                        if (move > 4):
                            ## increment frame -1
                            frame = int(bpy.data.objects["End_frame"].data.body)
                            frame -= 1
                            bpy.data.objects["End_frame"].data.body = str(frame)
                            bpy.data.objects["Frame_end_UI"].data.body = str(frame)

                            self.update_bars()

                            self.y_pre_10 = y

                else:
                    self.y_pre_10 = y

                self.y_pre = y

                self.y_pre = y

                if ctrl_state_l.ulButtonPressed == 4294967296:

                    # START_FRAME
                    if (x < 0):
                        self.next_state_l = StateLeft.TIMELINE


                if ctrl_state_l.ulButtonPressed != 4294967296:

                    if(self.next_state_l == StateLeft.TIMELINE):

                        # strfr = bpy.data.objects["Start_frame"]
                        # strfr.scale = strfr.scale * 3

                        endfr = bpy.data.objects["End_frame"]
                        endfr.scale = endfr.scale / 3

                        print("END_FRAME --> TIMELINE")
                        self.state_l = StateLeft.TIMELINE
                        bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-timeline.png"]
                        bpy.data.objects["Text.L"].data.body = "Timeline"

                        bpy.data.objects["wheel"].hide = True

            elif self.state_l == StateLeft.DRAG_PANEL:

                if ctrl_state_l.ulButtonPressed == 8589934592:
                    Panel_UI = bpy.data.objects["Panel_UI"]
                    Empty = bpy.data.objects["Panel_empty"]
                    # Panel_UI.location = ctrl_l.location
                    Panel_UI.matrix_world = Empty.matrix_world

                if ctrl_state_l.ulButtonPressed != 8589934592:
                    print("DRAG_PANEL -> IDLE")
                    bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-idle.png"]
                    self.state_l = StateLeft.IDLE

            elif self.state_l == StateLeft.THRESHOLD:

                x, y = ctrl_state_l.rAxis[0].x, ctrl_state_l.rAxis[0].y
                bpy.data.objects["Text.L"].data.body = "Threshold"

                if ctrl_state_l.ulButtonPressed != 4294967296:

                    if(y!= 0 and x!=0):
                        if(y > -0.5):
                            self.update_slider(x)

                if ctrl_state_l.ulButtonPressed == 4294967296:

                    if (y < -0.5):
                        print("THRESHOLD -> THRESHOLD_EXIT")
                        self.state_l = StateLeft.THRESHOLD_EXIT

            elif self.state_l == StateLeft.THRESHOLD_EXIT:

                if ctrl_state_l.ulButtonPressed != 4294967296:
                    print("THRESHOLD_EXIT -> IDLE")
                    self.state_l = StateLeft.IDLE
                    bpy.data.textures["Texture.L"].image = bpy.data.images["Hand-L-idle.png"]

                    # HIDE SLIDER
                    bpy.data.objects["slider"].hide = True
                    bpy.data.objects["slider_bar"].hide = True
                    bpy.data.objects["slider_value"].hide = True

            super(OpenVR, self).loop(context)

        except Exception as E:
            self.error("OpenVR.loop", E, False)
            return False

        #if VERBOSE:
        #    print("Left Eye Orientation Raw: " + str(self._eye_orientation_raw[0]))
        #    print("Right Eye Orientation Raw: " + str(self._eye_orientation_raw[1]))

        return True

    def frameReady(self):
        """
        The frame is ready to be sent to the device
        """
        try:
            self._hmd.frameReady()

        except Exception as E:
            self.error("OpenVR.frameReady", E, False)
            return False

        return True

    def reCenter(self):
        """
        Re-center the HMD device

        :return: return True if success
        :rtype: bool
        """
        return self._hmd.reCenter()

    def quit(self):
        """
        Garbage collection
        """
        self._hmd = None
        return super(OpenVR, self).quit()



