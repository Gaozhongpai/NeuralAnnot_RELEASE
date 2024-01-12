import torch
import numpy as np
import pickle
from typing import Optional
import torch.nn as nn

import smplx
from smplx.lbs import vertices2joints
from smplx.utils import MANOOutput, to_tensor
from smplx.vertex_ids import vertex_ids

class MANO(nn.Module):
    def __init__(self, mano_path, use_pca, is_rhand):
        super(MANO, self).__init__()

        self.layer = smplx.create(mano_path, 'mano', use_pca=use_pca, is_rhand=is_rhand)
        self.joint_regressor = self.layer.J_regressor
        self.faces = self.layer.faces
        
        self.joint_num = 21
        self.joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
        self.skeleton = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )
        self.root_joint_idx = self.joints_name.index('Wrist')

        # add fingertips to joint_regressor
        self.fingertip_vertex_idx = [745, 317, 444, 556, 673] # mesh vertex idx (right hand)
        thumbtip_onehot = torch.tensor([1 if i == 745 else 0 for i in range(self.joint_regressor.shape[1])]).view(1,-1)
        indextip_onehot = torch.tensor([1 if i == 317 else 0 for i in range(self.joint_regressor.shape[1])]).view(1,-1)
        middletip_onehot = torch.tensor([1 if i == 445 else 0 for i in range(self.joint_regressor.shape[1])]).view(1,-1)
        ringtip_onehot = torch.tensor([1 if i == 556 else 0 for i in range(self.joint_regressor.shape[1])]).view(1,-1)
        pinkytip_onehot = torch.tensor([1 if i == 673 else 0 for i in range(self.joint_regressor.shape[1])]).view(1,-1)
        self.joint_regressor = torch.cat((self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
        self.joint_regressor = self.joint_regressor[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]

    def forward(self, betas, hand_pose, global_orient, transl):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 778, 3)
        Output:
            3D joints: size = (B, 21, 3)
        """
        outputs = self.layer(betas=betas, hand_pose=hand_pose, global_orient=global_orient, transl=transl)
        joints = torch.einsum('bik,ji->bjk', [outputs.vertices, self.joint_regressor])
        return outputs, joints



class MANOv2(smplx.MANO):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official MANO implementation to support more joints.
        Args:
            Same as MANOLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(MANOv2, self).__init__(*args, **kwargs)
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        #2, 3, 5, 4, 1
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('extra_joints_idxs', to_tensor(list(vertex_ids['mano'].values()), dtype=torch.long))
        self.register_buffer('joint_map', torch.tensor(mano_to_openpose, dtype=torch.long))

    def forward(self, *args, **kwargs) -> MANOOutput:
        """
        Run forward pass. Same as MANO and also append an extra set of joints if joint_regressor_extra is specified.
        """
        mano_output = super(MANOv2, self).forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output
