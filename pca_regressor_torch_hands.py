# -*- coding: utf-8 -*-
# Script to write registrations as obj files
# Copyright (c) [2015] [Gerard Pons-Moll]

from argparse import ArgumentParser
import os
from os import mkdir
from os.path import join, exists
import numpy as np
import json
import trimesh
from MSCOCO.pca_torch import PCA
from joblib import dump, load
import random 
import pickle

import torch
from glob import glob
import logging
import smplx
from MSCOCO.geometry import frontalize_V2, apply_transformation

# Configure the logging
logging.basicConfig(filename='head_pca.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def write_mesh_as_obj(fname, verts, faces):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def load_skeleton(path):
    
    # load joint info (name, parent_id)
    skeleton = {}
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_name] = joint_id

    return skeleton

def glob_ply_files(folder_path, hand):
    # Search for .ply files in all subdirectories
    if hand == "left":
        ply_files = glob(os.path.join(folder_path, '**/*l.ply'), recursive=True)
    else:
        ply_files = glob(os.path.join(folder_path, '**/*r.ply'), recursive=True)
    return ply_files




if __name__ == '__main__':
    
    dataset = "./dataset/handsOnly/handsOnly_REGISTRATIONS_r_l_lm"
    # Replace 'your_folder_path' with the path of your folder
    
    
    mano_path = 'dataset/models'
    mano_layer = {'right': smplx.create(mano_path, 'mano', use_pca=False, is_rhand=True), 
                  'left': smplx.create(mano_path, 'mano', use_pca=False, is_rhand=False)}
        
    hands = ['right', 'left']
    for hand in hands:
        f_tempate = trimesh.Trimesh(vertices=mano_layer[hand].v_template, faces=mano_layer[hand].faces)
        f_tempate.export("dataset/models/parts/hand_{}_template.obj".format(hand))
        template = mano_layer[hand].v_template.cuda()
        
        # Print the list of found .ply files
        ply_files_all = glob_ply_files(dataset, hand)
        mesh = torch.zeros([len(ply_files_all), template.shape[0], 3]).cuda()
        pca = PCA(n_components=mesh.shape[1]).to("cuda")
        
        for i, file in enumerate(ply_files_all):
            if i % 2000 == 0:
                print("It is at: {}".format(i))
                logging.info("It is at: {}".format(i))
            pmesh = trimesh.load(file)
            vertices = torch.from_numpy(pmesh.vertices).float().cuda()
            rotation_matrix, translation, _ = frontalize_V2(vertices, template)
            mesh[i] = apply_transformation(vertices, rotation_matrix, translation)
        
        pca.fit(mesh.view(mesh.shape[0], -1))
        
        # Save the PCA model parameters
        dump(pca.to("cpu"), 'dataset/models/parts/pca_hand_{}.pkl'.format(hand))
        # Load the PCA model
        pca = load('dataset/models/parts/pca_hand_{}.pkl'.format(hand)).to("cuda")
        print(pca)
        
        components = pca.components_
        mean = pca.mean_
        parameter_pca = pca.transform(mesh.view(mesh.shape[0], -1)) 
        
        num_comps = list(range(128))
        errors = []
        num_comps = list(range(128))
        errors = []
        for num_comp in num_comps:
            recover_align = (torch.matmul(parameter_pca[:, :num_comp+1], components[:num_comp+1]) + mean)
            error = torch.mean(torch.linalg.norm(mesh.reshape(-1, 3) - recover_align.reshape(-1, 3), dim=1))
            errors.append(error)
            print("Hand {}: Number of components: {}, the error is {}".format(hand, num_comp+1, error))
            logging.info("hand {}: Number of components: {}, the error is {}".format(hand, num_comp+1, error))
        errors = torch.stack(errors)
        dump(errors.cpu().numpy(), 'dataset/models/parts/error_hand_{}.pkl'.format(hand))
    print("Successs and Over!")
    logging.info("Successs and Over!")
     
        
     
        
