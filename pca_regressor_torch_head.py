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

def glob_ply_files(folder_path):
    # Search for .ply files in all subdirectories
    ply_files = glob(os.path.join(folder_path, '**/*.ply'), recursive=True)
    return ply_files

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

if __name__ == '__main__':
    
    flame_path = 'dataset/models'
    flame_layer = smplx.create(flame_path, 'flame')
    f_tempate = trimesh.Trimesh(vertices=flame_layer.v_template, faces=flame_layer.faces)
    f_tempate.export("dataset/models/parts/head_template.obj")
    template = flame_layer.v_template.cuda()
    
    dataset = "./dataset/TEMPEH"
    # Replace 'your_folder_path' with the path of your folder
    ply_files_TEMPEH = glob_ply_files(dataset)
    # If there are more than 200 files, randomly select 200 of them
    selected_files = random.sample(ply_files_TEMPEH, 80000)
    dataset = "./dataset/COMA_data"
    # Replace 'your_folder_path' with the path of your folder
    ply_files_COMA = glob_ply_files(dataset)
    ply_files_all = selected_files + ply_files_COMA
    # Print the list of found .ply files
    mesh = torch.zeros([len(ply_files_all), 5023, 3]).cuda()
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
    dump(pca.to("cpu"), 'dataset/models/parts/pca_{}.pkl'.format("head"))
    # Load the PCA model
    pca = load('dataset/models/parts/pca_{}.pkl'.format("head")).to("cuda")
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
        print("Head: Number of components: {}, the error is {}".format(num_comp+1, error))
        logging.info("Head: Number of components: {}, the error is {}".format(num_comp+1, error))
    errors = torch.stack(errors)
    dump(errors.cpu().numpy(), 'dataset/models/parts/error_head.pkl')
    print("Successs and Over!")
    logging.info("Successs and Over!")
     
        
     
        
