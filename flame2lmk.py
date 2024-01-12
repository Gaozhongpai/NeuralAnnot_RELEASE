import trimesh
import numpy as np
from flame_fitting.fitting.landmarks import landmarks_to_mesh, load_embedding, mesh_points_by_barycentric_coordinates

mesh = trimesh.load("/code/dataset/COMA_data/FaceTalk_170725_00137_TA/bareteeth/bareteeth.000001.ply")
lmk_emb_path = './dataset/models/flame/flame_static_embedding_68.pkl' 
lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
mesh_verts = mesh.vertices
mesh_faces = mesh.faces
v_selected = mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords )
lmk_num  = lmk_face_idx.shape[0]
# an index to select which landmark to use
lmk_selection = np.arange(0,lmk_num).ravel() # use all
lmks = np.array(v_selected[lmk_selection])
lmk_mesh = landmarks_to_mesh(lmks)
lmk_mesh.set_face_colors([255, 0, 0])
lmk_mesh.write_ply("lmk_mesh.ply")
