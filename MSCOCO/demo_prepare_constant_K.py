import json
import numpy as np
import cv2
import torch
import smplx
from pycocotools.coco import COCO
import os.path as osp
import os, sys
from pathlib import Path
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import trimesh
from joblib import dump, load
from tqdm import tqdm
import torch.nn.functional as F
import shutil
import logging

# current_directory = os.path.dirname(os.path.abspath(__file__))
# parent_directory = os.path.dirname(current_directory)
# sys.path.append(parent_directory)
from geometry import perspective_projection, draw_skeleton, calc_global_translation
from geometry import frontalize_V2, apply_transformation, apply_transformation_center
from pca_torch import PCA 

# Configure the logging
logging.basicConfig(filename='yolo_mesh_prepare.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

#  (focal length: (5000,5000), princpt:(128,128)),

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

def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x

def flip_mano(hand_pose, trans):
    hand_pose[1::3] *= -1
    hand_pose[2::3] *= -1
    
    trans[0] = -trans[0]
    return hand_pose, trans

def demo():
    # target_aid = 476384
    num_comps = 64
    head_pca = load('models/parts/pca_head.pkl').to("cuda")
    hand_pca = {'left': load('models/parts/pca_hand_left.pkl').to("cuda"),
                'right': load('models/parts/pca_hand_right.pkl').to("cuda")}    
    
    model_path = 'models'
    flame_layer = smplx.create(model_path, 'flame')
    head_template = flame_layer.v_template.cuda()
    
    mano_layer = {'right': smplx.create(model_path, 'mano', use_pca=False, is_rhand=True), 
                  'left': smplx.create(model_path, 'mano', use_pca=False, is_rhand=False)}
    hand_template = {'left': mano_layer['left'].v_template.cuda(),
                     'right': mano_layer['right'].v_template.cuda()}
    
    # flame parameter load
    with open('dataset/coco/annotations/MSCOCO_train_FLAME_NeuralAnnot.json','r') as f:
        flame_params = json.load(f)
        
    with open('dataset/coco/annotations/MSCOCO_train_MANO_NeuralAnnot.json','r') as f:
        mano_params = json.load(f)
    
    coco80 = coco91_to_coco80_class()
    
    json_dir = "dataset/coco/annotations"
    save_dir = "dataset/coco/"
    hand_scales = []
    head_scales = []
    
    for json_file in sorted(Path(json_dir).resolve().glob('*_v1.0.json')):
        print(json_file)
        logging.info(json_file)
        
        if "train" in json_file.name:
            fn = Path(save_dir) / 'labels' / json_file.stem.replace('train_v1.0', 'mesh_train2017')  # folder name
        else:
            fn = Path(save_dir) / 'labels' / json_file.stem.replace('val_v1.0', 'mesh_val2017')  # folder name
        if fn.exists():
            shutil.rmtree(fn)  # delete dir
        fn.mkdir()
        
        fnvis = Path(save_dir) / 'labels' / json_file.stem.replace('_v1.0', '_vis')  # folder name
        if fnvis.exists():
            shutil.rmtree(fnvis)  # delete dir
        fnvis.mkdir()
        
        count = 0    
        n_vis = 500
        is_rendered = False
        
        # Define the constant matrix for element-wise multiplication
        flip_matrix = torch.tensor([[1., -1., -1.],
                                    [-1., 1., 1.],
                                    [-1., 1., 1.]]).cuda()
        db = COCO(json_file)
        for img_id, anns in tqdm(db.imgToAnns.items()):
            img = db.imgs[img_id]
            h, w, f = img['height'], img['width'], img['file_name']
            shape_kpt = np.array([w, h, 1]).astype(np.float32)
            focal_2 = torch.FloatTensor([5000, 5000])
            princpt_2 = torch.FloatTensor([w/2, h/2])
            # ===> very important ### to calucate the distributino of trans_uv and scale <=== #
            K = torch.zeros([1, 3, 3])
            K[:,0,0] = focal_2[None][:,0]
            K[:,1,1] = focal_2[None][:,1]
            K[:,2,2] = 1.
            K[:,:-1, -1] = princpt_2[None]        
        
            img_path = osp.join('dataset/coco/train2017', f)
            # print(img_path)
            if count % n_vis == 0:
                rendered_img = cv2.imread(img_path)
                rendered_img_flip = cv2.flip(rendered_img, 1)
            bboxes = []
            meta = []
            for k, ann in enumerate(anns):
                if ann['iscrowd']:
                    continue
                
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                
                cls = coco80[ann['category_id'] - 1]
                if box[2] <= 0 or box[3] <= 0 or cls !=0:  # if w <= 0 and h <= 0
                    continue
                
                ################ for hands ################
                for hand in ('left', 'right'):
                    
                    lbox = np.array(ann['{}hand_box'.format(hand)], dtype=np.float64)
                    lbox[:2] += lbox[2:] / 2  # xy top-left corner to center
                    lbox[[0, 2]] /= w  # normalize x
                    lbox[[1, 3]] /= h  # normalize y
                    cls_hand = 1 if hand == 'left' else 2
                    
                    if lbox[2] > 0 and lbox[3] > 0:  # if w <= 0 and h <= 0
                        kpts = np.reshape(np.array(ann['{}hand_kpts'.format(hand)], dtype=np.float32), (21,3))
                        kpts = kpts / shape_kpt
                        is_mesh = 0
                        if str(ann['id']) in mano_params:  ### for hand
                            if mano_params[str(ann['id'])][hand]:
                                is_mesh = 1
                        lbox = [cls_hand] + lbox.tolist() + [ann['id']] + [is_mesh]# palm_box.tolist()
                        bboxes.append(lbox)

                        if is_mesh:
                            # mano parameter
                            mano_param = mano_params[str(ann['id'])][hand]
                            pose, shape, trans = mano_param['mano_param']['pose'], mano_param['mano_param']['shape'], mano_param['mano_param']['trans']
                            pose = torch.FloatTensor(pose).view(-1,3) # (24,3)
                            root_pose = pose[0,None,:]
                            hand_pose = pose[1:,:]
                            shape = torch.FloatTensor(shape).view(1,-1) # MANO shape parameter
                            trans = torch.FloatTensor(trans).view(1,-1) # translation vector
                            focal = torch.FloatTensor(mano_param['cam_param']['focal'])
                            princpt = torch.FloatTensor(mano_param['cam_param']['princpt'])
                            
                            # get mesh and joint coordinates
                            with torch.no_grad():
                                output = mano_layer[hand](betas=shape, hand_pose=hand_pose.view(1,-1), global_orient=root_pose, transl=trans)
                            mesh_cam = output.vertices[0].cuda()
                            
                            rotation_matrix, translation, trans_mesh = frontalize_V2(mesh_cam, hand_template[hand])
                            mesh_cam_trans = apply_transformation(mesh_cam, rotation_matrix, translation)
                            parameter_pca = hand_pca[hand].transform(mesh_cam_trans.view(1, -1))[:, :num_comps]
                            
                            trans_uv = perspective_projection(trans_mesh.cpu()[None, None], focal[None], princpt[None])
                            scale =  focal.mean() / trans_mesh[-1].cpu()
                            hand_scales.append(scale.item())
                            trans_mesh_unproject = calc_global_translation(trans_uv, scale, K)
                            
                            if count % n_vis == 0:
                                is_rendered = True
                                recover_align = (torch.matmul(parameter_pca, hand_pca[hand].components_[:num_comps]) + hand_pca[hand].mean_).T
                                recover_align = recover_align.view(-1, 3)
                                recover = apply_transformation_center(recover_align, rotation_matrix, trans_mesh)
                                error = torch.norm(recover - mesh_cam).cpu()
                                recover_2 = apply_transformation(recover_align, rotation_matrix, translation, is_inv=True)
                                error_2 = torch.norm(recover_2 - mesh_cam).cpu()
                                print("Hand error is: {}, error origin is: {}".format(error, error_2))
                                logging.info("Hand error is: {}, error origin is:".format(error, error_2))
                                print("Hand trans_mesh is: {} and trans_mesh_unproject is: {}".format(trans_mesh, trans_mesh_unproject))
                                logging.info("Hand trans_mesh is: {} and trans_mesh_unproject is: {}".format(trans_mesh, trans_mesh_unproject))
                                # mesh render
                                recover = apply_transformation_center(recover_align, rotation_matrix, trans_mesh_unproject.squeeze().cuda())
                                rendered_img = render_mesh(rendered_img, recover.cpu().numpy(), mano_layer[hand].faces, {'focal': focal_2, 'princpt': princpt_2})
                                cv2.circle(rendered_img, (int(trans_uv[0, 0, 0]), int(trans_uv[0, 0, 1])), 3, (0,0,255), -1)
                            
                            
                            ## flip mano parameter
                            mano_param = mano_params[str(ann['id'])][hand]
                            pose, shape, trans = mano_param['mano_param']['pose'], mano_param['mano_param']['shape'], mano_param['mano_param']['trans']
                            pose, trans = flip_mano(np.array(pose), np.array(trans))
                            
                            pose = torch.FloatTensor(pose).view(-1,3) # (24,3)
                            root_pose = pose[0,None,:]
                            hand_pose = pose[1:,:]
                            shape = torch.FloatTensor(shape).view(1,-1) # MANO shape parameter
                            trans = torch.FloatTensor(trans).view(1,-1) # translation vector
                            focal = torch.FloatTensor(mano_param['cam_param']['focal'])
                            princpt = torch.FloatTensor(mano_param['cam_param']['princpt'])
                            princpt[0] = w - princpt[0] # flip the camera
                        
                            hand_flip = "right" if hand == "left" else "left"
                            # get mesh and joint coordinates
                            with torch.no_grad():
                                output = mano_layer[hand_flip](betas=shape, hand_pose=hand_pose.view(1,-1), global_orient=root_pose, transl=trans)
                            mesh_cam = output.vertices[0].cuda()
                            
                            # rotation_matrix_flip, translation_flip, trans_mesh_flip = frontalize_V2(mesh_cam, hand_template[hand_flip])
                            rotation_matrix_flip = rotation_matrix*flip_matrix
                            translation_flip = translation.clone()
                            translation_flip[..., 0] = -translation_flip[..., 0]
                            trans_mesh_flip = trans_mesh.clone()
                            trans_mesh_flip[..., 0] = -trans_mesh_flip[..., 0]
                        
                            mesh_cam_trans = apply_transformation(mesh_cam, rotation_matrix_flip, translation_flip)
                            parameter_pca_flip = hand_pca[hand_flip].transform(mesh_cam_trans.view(1, -1))[:, :num_comps]
                            
                            trans_uv_flip = perspective_projection(trans_mesh_flip.cpu()[None, None], focal[None], princpt[None]) # flip the cameara
                            scale_flip = focal.mean() / trans_mesh_flip[-1].cpu()
                            trans_mesh_unproject = calc_global_translation(trans_uv_flip, scale_flip, K)
                            
                            if count % n_vis == 0:
                                is_rendered = True
                                recover_align = (torch.matmul(parameter_pca_flip, hand_pca[hand_flip].components_[:num_comps]) + hand_pca[hand_flip].mean_).T
                                recover_align = recover_align.view(-1, 3)
                                recover = apply_transformation_center(recover_align, rotation_matrix_flip, trans_mesh_flip)
                                error = torch.norm(recover - mesh_cam).cpu()
                                recover_2 = apply_transformation(recover_align, rotation_matrix_flip, translation_flip, is_inv=True)
                                error_2 = torch.norm(recover_2 - mesh_cam).cpu()
                                print("Hand flip error is: {}, error orgin is: {}".format(error, error_2))
                                logging.info("Hand flip error is: {}, error orgin is:".format(error, error_2))
                                print("Hand flip trans_mesh is: {} and trans_mesh_unproject is: {}".format(trans_mesh_flip, trans_mesh_unproject))
                                logging.info("Hand trans_mesh is: {} and trans_mesh_unproject is: {}".format(trans_mesh_flip, trans_mesh_unproject))
                                # mesh render
                                recover = apply_transformation_center(recover_align, rotation_matrix_flip, trans_mesh_unproject.squeeze().cuda())
                                rendered_img_flip = render_mesh(rendered_img_flip, recover.cpu().numpy(), mano_layer[hand_flip].faces, {'focal': focal_2, 'princpt': princpt_2})
                                cv2.circle(rendered_img_flip, (int(trans_uv_flip[0, 0, 0]), int(trans_uv_flip[0, 0, 1])), 3, (0,0,255), -1)
                                
                            meta.append({"rotation": rotation_matrix.cpu(), 
                                        "trans_uv": trans_uv.squeeze() / torch.tensor([w, h]),
                                        "scale": scale,
                                        "parameter_pca": parameter_pca.cpu(),
                                        "parameter_pca_flip": parameter_pca_flip.cpu(),  
                                        "kpts": kpts, 
                                        "cls": cls_hand,
                                        "id": ann['id'],
                                        "box": lbox,
                                        "is_mesh": is_mesh})
                            
                        else:
                            print(f)
                            logging.info(f)
                            meta.append({"kpts": kpts, 
                                        "cls": cls_hand,
                                        "id": ann['id'],
                                        "box": lbox,
                                        "is_mesh": is_mesh})
                
                ################ for face ################
                fbox = np.array(ann['face_box'], dtype=np.float64)
                fbox[:2] += fbox[2:] / 2  # xy top-left corner to center
                fbox[[0, 2]] /= w  # normalize x
                fbox[[1, 3]] /= h  # normalize y
                if fbox[2] > 0 and fbox[3] > 0:  # if w <= 0 and h <= 0
                    kpts = np.reshape(np.array(ann['face_kpts'], dtype=np.float32), (68,3))
                    kpts = kpts / shape_kpt
                    # print(kpts)
                    is_mesh = 0
                    if str(ann['id']) in flame_params:
                        is_mesh = 1
                    fbox = [3] + fbox.tolist() + [ann['id']] + [is_mesh] # palm_box.tolist()
                    bboxes.append(fbox)
                    
                    if is_mesh:
                    
                        flame_param = flame_params[str(ann['id'])]
                        root_pose, jaw_pose, expr, shape, trans = flame_param['flame_param']['root_pose'], flame_param['flame_param']['jaw_pose'], flame_param['flame_param']['expr'], flame_param['flame_param']['shape'], flame_param['flame_param']['trans']
                        root_pose = torch.FloatTensor(root_pose).view(1,3)
                        jaw_pose = torch.FloatTensor(jaw_pose).view(1,3)
                        expr = torch.FloatTensor(expr).view(1,-1) # facial expression code
                        shape = torch.FloatTensor(shape).view(1,-1) # FLAME shape parameter
                        trans = torch.FloatTensor(trans).view(1,-1) # translation vector
                        focal = torch.FloatTensor(flame_param['cam_param']['focal'])
                        princpt = torch.FloatTensor(flame_param['cam_param']['princpt'])
                        
                        # get mesh and joint coordinates
                        with torch.no_grad():
                            output = flame_layer(betas=shape, jaw_pose=jaw_pose, global_orient=root_pose, transl=trans, expression=expr)
                        mesh_cam = output.vertices[0].cuda()
                        
                        rotation_matrix, translation, trans_mesh = frontalize_V2(mesh_cam, head_template)
                        mesh_cam_trans = apply_transformation(mesh_cam, rotation_matrix, translation)
                        parameter_pca = head_pca.transform(mesh_cam_trans.view(1, -1))[:, :num_comps]
                        
                        trans_uv = perspective_projection(trans_mesh.cpu()[None, None], focal[None], princpt[None])
                        scale =  focal.mean() / trans_mesh[-1].cpu()
                        head_scales.append(scale.item())
                        trans_mesh_unproject = calc_global_translation(trans_uv, scale, K)
                        
                        if count % n_vis == 0:
                            is_rendered = True
                            recover_align = (torch.matmul(parameter_pca, head_pca.components_[:num_comps]) + head_pca.mean_).T
                            recover_align = recover_align.view(-1, 3)
                            recover = apply_transformation_center(recover_align, rotation_matrix, trans_mesh)
                            error = torch.norm(recover - mesh_cam).cpu()
                            recover_2 = apply_transformation(recover_align, rotation_matrix, translation, is_inv=True)
                            error_2 = torch.norm(recover_2 - mesh_cam).cpu()
                            print("Face error is: {}, error origin is: {}".format(error, error_2))
                            logging.info("Face error is: {}, error origin is: {}".format(error, error_2))
                            print("Face trans_mesh is: {} and trans_mesh_unproject is: {}".format(trans_mesh, trans_mesh_unproject))
                            logging.info("Face trans_mesh is: {} and trans_mesh_unproject is: {}".format(trans_mesh, trans_mesh_unproject))
                            # mesh render
                            recover = apply_transformation_center(recover_align, rotation_matrix, trans_mesh_unproject.squeeze().cuda())
                            rendered_img = render_mesh(rendered_img, recover.cpu().numpy(), flame_layer.faces, {'focal': focal_2, 'princpt': princpt_2})
                            cv2.circle(rendered_img, (int(trans_uv[0, 0, 0]), int(trans_uv[0, 0, 1])), 3, (0,0,255), -1)
                        
                        meta.append({"rotation": rotation_matrix.cpu(), 
                                    "trans_uv": trans_uv.squeeze() / torch.tensor([w, h]),
                                    "scale": scale,
                                    "parameter_pca": parameter_pca.cpu(),
                                    "kpts": kpts, 
                                    "cls": 3,
                                    "id": ann['id'],
                                    "box": fbox,
                                    "is_mesh": is_mesh})
                        
                    else:
                        print(f)
                        logging.info(f)
                        meta.append({"kpts": kpts, 
                                    "cls": 3,
                                    "id": ann['id'],
                                    "box": fbox,
                                    "is_mesh": is_mesh})
                    
                ################ for body ################
                kpts = np.reshape(np.array(ann['keypoints'], dtype=np.float32), (17,3))
                kpts = kpts / shape_kpt
                meta.append({"kpts": kpts, 
                            "cls": 0,
                            "id": ann['id'],
                            "box": box})
                bbox = [cls] + box.tolist() + [ann['id']] + [0]
                bboxes.append(bbox)
                
            # Write
            if len(bboxes):
                if "train" in json_file.name:
                    filelist = (Path(save_dir) / json_file.stem.replace("train_v1.0", "mesh_train2017")).with_suffix('.txt')
                else:
                    filelist = (Path(save_dir) / json_file.stem.replace("val_v1.0", "mesh_val2017")).with_suffix('.txt')
                with open(filelist, 'a') as file_list:
                    file_list.write(os.path.join('./images', json_file.stem.split('_')[2] + '2017', f) + '\n')  ########################
                dump(meta, (fn / f).with_suffix('.pkl'))
                with open((fn / f).with_suffix('.txt'), 'a') as file:
                    for i in range(len(bboxes)):
                        line = bboxes[i]  # cls, box or segments
                        # Writing the formatted elements of 'line' to the file
                        file.write((' '.join(['%f'] * len(line)) % tuple(line)).rstrip() + '\n')
                if count % n_vis == 0 and is_rendered:
                    cv2.imwrite(str((fnvis / f).with_suffix('.jpg')), rendered_img)
                    cv2.imwrite(str((fnvis / f).with_suffix('.jpg')).replace(".jpg", "_flip.jpg"), rendered_img_flip)
                    
            count = count + 1
            is_rendered = False
    dump(hand_scales, 'models/parts/hand_scales.pkl')
    dump(head_scales, 'models/parts/head_scales.pkl')
    hand_scales = np.array(hand_scales)
    head_scales = np.array(head_scales)
    print("Hand scale mean: {} and std: {}".format(np.mean(hand_scales), np.std(hand_scales)))
    print("Head scale mean: {} and std: {}".format(np.mean(head_scales), np.std(head_scales)))
    logging.info("Hand scale mean: {} and std: {}".format(np.mean(hand_scales), np.std(hand_scales)))
    logging.info("Head scale mean: {} and std: {}".format(np.mean(head_scales), np.std(head_scales)))

if __name__ == "__main__":
    demo()
