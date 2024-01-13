import json
import numpy as np
import cv2
import torch
from pycocotools.coco import COCO
import os.path as osp
import os, sys
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import trimesh
from joblib import dump, load
import smplx
from mano_wrapper import MANO, MANOv2
from geometry import perspective_projection, draw_skeleton, calc_global_translation
from geometry import frontalize_V2, apply_transformation, apply_transformation_center
from pca_torch import PCA

# current_directory = os.path.dirname(os.path.abspath(__file__))
# parent_directory = os.path.dirname(current_directory)
# sys.path.append(parent_directory)

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

def flip_mano(root_pose, hand_pose):
    root_pose[1::3] *= -1
    root_pose[2::3] *= -1
    hand_pose[1::3] *= -1
    hand_pose[2::3] *= -1
    return root_pose, hand_pose

def demo():
    target_aid = 476384
    
    mano_path = 'models/mano'
    num_comps = 64

    mano_layer = {'right': MANOv2(mano_path, use_pca=False, is_rhand=True), 'left': MANOv2(mano_path, use_pca=False, is_rhand=False)}
    # mano_layer = {'right': smplx.create(mano_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(mano_path, 'mano', use_pca=False, is_rhand=False)}
    hand_template = {'left': mano_layer['left'].v_template.cuda(),
                     'right': mano_layer['right'].v_template.cuda()}
    hand_pca = {'left': load('models/parts/pca_hand_left.pkl').to("cuda"),
                'right': load('models/parts/pca_hand_right.pkl').to("cuda")} 
    
    with open('dataset/coco/annotations/MSCOCO_train_MANO_NeuralAnnot.json','r') as f:
        mano_params = json.load(f)

    example = load('example.pkl')
    # ann = example['ann']
    img = example['img']
    img_path = example['img_path']
    
    for hand_type in ('right', 'left'):
        # mano parameter
        mano_param = mano_params[str(target_aid)][hand_type]
        pose, shape, trans = mano_param['mano_param']['pose'], mano_param['mano_param']['shape'], mano_param['mano_param']['trans']
        pose = torch.FloatTensor(pose).view(-1,3) # (24,3)
        root_pose = pose[0,None,:]
        hand_pose = pose[1:,:]
        shape = torch.FloatTensor(shape).view(1,-1) # MANO shape parameter
        trans = torch.FloatTensor(trans).view(1,-1) # translation vector
        focal = torch.FloatTensor(mano_param['cam_param']['focal'])
        princpt = torch.FloatTensor(mano_param['cam_param']['princpt'])
        
        # ===> very important ### to calucate the distributino of trans_uv and scale <=== #
        K = torch.zeros([1, 3, 3])
        K[:,0,0] = focal[None][:,0]
        K[:,1,1] = focal[None][:,1]
        K[:,2,2] = 1.
        K[:,:-1, -1] = princpt[None]
    
        trans_uv = perspective_projection(trans[None], focal[None], princpt[None])
        scale =  focal.mean() / trans[:, -1]
        project3d = calc_global_translation(trans_uv, scale, K)
        # <=== very important ### to calucate the distributino of trans_uv and scale ===> #

        
        # get mesh and joint coordinates
        with torch.no_grad():
            output = mano_layer[hand_type](betas=shape, hand_pose=hand_pose.view(1,-1), global_orient=root_pose, transl=trans)
        
        joint_2d = perspective_projection(output.joints, focal[None], princpt[None])
        mesh_cam = output.vertices[0].cuda()
        
        rotation_matrix, translation, trans_mesh = frontalize_V2(mesh_cam, hand_template[hand_type])
        mesh_cam_trans = apply_transformation(mesh_cam, rotation_matrix, translation)
        parameter_pca = hand_pca[hand_type].transform(mesh_cam_trans.view(1, -1))[:, :num_comps]

        # trans_uv = perspective_projection(-trans_mesh[None, None], focal[None], princpt[None])
        trans_uv = perspective_projection(trans_mesh.cpu()[None, None], focal[None], princpt[None])
        scale = focal.mean() / trans_mesh[-1].cpu()
        trans_mesh_unproject = calc_global_translation(trans_uv, scale, K)
        
        recover_align = (torch.matmul(parameter_pca, hand_pca[hand_type].components_[:num_comps]) + hand_pca[hand_type].mean_).T
        recover_align = recover_align.view(-1, 3)
        
        # mesh render
        img = cv2.imread(img_path)
        
        ###################
        r = 1.5
        padw = 55
        padh = 98
        pad_value = 114  # Padding color value

        h0, w0 = img.shape[:2]  # orig hw
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        
        # Create a new canvas with the size of the original image plus padding
        height, weight = img.shape[:2]
        canvas = np.full((height, weight, 3), pad_value, dtype=np.uint8)
        canvas[padh:height, padw:weight] = img[:height-padh, :weight-padw]
        rendered_img = canvas.copy()        
        
        focal_2 = torch.FloatTensor([5000, 5000])
        princpt_2 = torch.FloatTensor([weight/2, height/2])
        # ===> very important ### to calucate the distributino of trans_uv and scale <=== #
        K_2 = torch.zeros([1, 3, 3])
        K_2[:,0,0] = focal_2[None][:,0]
        K_2[:,1,1] = focal_2[None][:,1]
        K_2[:,2,2] = 1.
        K_2[:,:-1, -1] = princpt_2[None]
        
        #### only needs to change trans_uv and scale ###############
        ############################################################
        trans_uv = trans_uv * r + torch.FloatTensor([padw, padh])   ## scale and shift trans_uv
        scale = scale * r  ## scale the distance
        trans_mesh_unproject = calc_global_translation(trans_uv, scale, K_2)
        
        recover = apply_transformation_center(recover_align, rotation_matrix, trans_mesh_unproject.cuda()[0])

        # rendered_img = render_mesh(rendered_img, mesh_cam.cpu().numpy(), mano_layer[hand_type].faces, {'focal': focal, 'princpt': princpt})
        rendered_img = render_mesh(rendered_img, recover.cpu().numpy(), mano_layer[hand_type].faces, {'focal': focal_2, 'princpt': princpt_2})
        cv2.circle(rendered_img, (int(trans_uv[0, 0, 0]), int(trans_uv[0, 0, 1])), 3, (0,0,255), -1)
        # img_2d = draw_skeleton(img, joint_2d[0].numpy(), True)
        
        cv2.imwrite('mano_' + hand_type + '_aug.jpg', rendered_img)
        
if __name__ == "__main__":
    demo()
