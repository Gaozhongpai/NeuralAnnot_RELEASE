import json
import numpy as np
import cv2
import torch
import smplx
from pycocotools.coco import COCO
import os.path as osp
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import trimesh
from joblib import dump, load

def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        bbox = None

    return bbox

def process_bbox(bbox, img_width, img_height):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 1.0
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def render_mesh(img, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
	np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    return img

def flip_mano(hand_pose, trans):
    hand_pose[1::3] *= -1
    hand_pose[2::3] *= -1
    
    trans[0] = -trans[0]
    return hand_pose, trans

def demo():
    target_aid = 476384
    # mano parameter load
    with open('dataset/coco/annotations/MSCOCO_train_MANO_NeuralAnnot.json','r') as f:
        mano_params = json.load(f)

    mano_path = 'models'
    mano_layer = {'right': smplx.create(mano_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(mano_path, 'mano', use_pca=False, is_rhand=False)}
    
    example = load('example.pkl')
    # ann = example['ann']
    img = example['img']
    img_path = example['img_path']
    
    for hand_type in ('right', 'left'):
        # mano parameter
        mano_param = mano_params[str(target_aid)][hand_type]
        pose, shape, trans = mano_param['mano_param']['pose'], mano_param['mano_param']['shape'], mano_param['mano_param']['trans']
        pose, trans = flip_mano(np.array(pose), np.array(trans))
        
        pose = torch.FloatTensor(pose).view(-1,3) # (24,3)
        root_pose = pose[0,None,:]
        hand_pose = pose[1:,:]
        shape = torch.FloatTensor(shape).view(1,-1) # MANO shape parameter
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector
        
        hand_type = "left" if hand_type == "right" else "right"
        # get mesh and joint coordinates
        with torch.no_grad():
            output = mano_layer[hand_type](betas=shape, hand_pose=hand_pose.view(1,-1), global_orient=root_pose, transl=trans)
        mesh_cam = output.vertices[0].numpy()

        # mesh render
        img = cv2.imread(img_path)
        img = cv2.flip(img, 1)
        
        focal = mano_param['cam_param']['focal']
        princpt = mano_param['cam_param']['princpt']
        princpt[0] = img.shape[1] - princpt[0]
        rendered_img = render_mesh(img, mesh_cam, mano_layer[hand_type].faces, {'focal': focal, 'princpt': princpt})
        cv2.imwrite('mano_' + hand_type + '.jpg', rendered_img)
    

if __name__ == "__main__":
    demo()
