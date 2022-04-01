fileDir = './data/test.dxf'

# indices of the vertices will be shown once the code runs

# ups: list, indices of vertices with upward perturbation
ups = [38, 39, 33, 26, 32, 25]

# downs: list, indices of vertices with downward perturbation
downs = [9, 18, 23, 22, 4, 5, 19, 8, 12, 1, 11, 2]

# shifts: dict, shifts of vertices
#   {
#      the 1st index of vertex:  [final position of the vertex],
#      the 2st index of vertex:  [final position of the vertex],
#   }

shifts = {
    33: [0, -11.23, 0],
    26: [0, -11.23, 0],
    32: [0, -7.374, 0],
    25: [0, -7.374, 0],
}

# dt: float, [0.0 - 1.0] speed of the folding,
dt = 0.1


# ---------------------------- ↑ params ------------------------------

import igl
import numpy as np
import scipy.optimize
from scipy import optimize
import polyscope as ps
from ezdxf import recover
import torch
from matplotlib import pyplot as plt
from matplotlib import collections  as mc
import triangle as tr


dxf = recover.readfile(fileDir)

points = None

F = []
msp = dxf[0].modelspace()
for entity in msp:
    assert(len(entity.vertices) >= 3)
    ipoints = []
    for v in entity.vertices:
        p = np.array(v.dxfattribs()['location'])
        if points is not None:
            id_same = np.where(((p - points) ** 2).sum(1) < 1e-3)[0]
            if len(id_same):
                ip = id_same[0]
            else:
                points = np.vstack([points, p])
                ip = len(points) - 1
        else:
            points = p.reshape(1, 3)
            ip = 0
        ipoints.append(ip)
    if ipoints[-1] == ipoints[0]:
        ipoints.pop()
    F.append(ipoints)

V = points
V0 = torch.tensor(points.copy())


# V = np.array([
#     [0.5, 0, 0],
#     [0.5, -0.5, 0],
#     [1, -0.5, 0],
#     [1, -1, 0],
#     [-1, -1, 0],
#     [-1, 1, 0],
#     [1, 1, 0],
#     [1, 0.5, 0],
#     [0.5, 0.5, 0]
# ])
# ivs = [0, 1, 2, 3, 4, 5, 6, 7, 8]

def triangulate(ivs, V, direction=1):
    # direction: the direction to rotate the normal, 1 or -1
    ivs = np.array(ivs, dtype=int)
    V0 = V
    V = V[:, :2]
    es = np.hstack([ivs[:-1].reshape(-1, 1), ivs[1:].reshape(-1, 1)])
    es = np.vstack([es, [ivs[-1], ivs[0]]])
    holes = []
    
    for i in range(len(ivs)):
        iv = ivs[i]
        iv_next = ivs[(i+1) % len(ivs)]
        iv_prev = ivs[(i-1) % len(ivs)]
        
        vec_next = V[iv_next] - V[iv]
        vec_next = vec_next / np.linalg.norm(vec_next)
        vec_prev = V[iv] - V[iv_prev]
        vec_prev = vec_prev / np.linalg.norm(vec_prev)
        
        normal = (vec_next + vec_prev) / 2
        normal[0], normal[1] = -direction * normal[1], direction * normal[0]
        hole = V[iv] + normal / np.sqrt((normal ** 2).sum()) * 1e-1
        holes.append(hole)
    
    A = dict(vertices=V, segments=es.tolist(), holes=holes)
    B = tr.triangulate(A, 'p')
    
    if 'triangles' not in B:
        faces = triangulate(ivs, V0, -1)
        return faces
    
    # tr.compare(plt, A, B)
    # plt.show()
    return B['triangles'].tolist()
    
F_new = []
face_groups = []
for face in F:
    if len(face) > 3:
        faces_new = triangulate(face, V)
        
        i_f_begin = len(F_new)
        F_new += faces_new
        face_groups.append(np.arange(i_f_begin, i_f_begin + len(faces_new)))
    else:
        F_new.append(face)
F = F_new

F = np.array(F, dtype=int)
F, _ = igl.bfs_orient(F)
F = torch.tensor(F)

# visualize vertex ids
ES = np.vstack([F[:, :2], F[:, 1:3], np.hstack([F[:, 2].reshape(-1, 1), F[:, 0].reshape(-1, 1)])])
lines = V[:, :2][ES]
lc = mc.LineCollection(lines, linewidths=2)
fig, ax = plt.subplots()
ax.add_collection(lc)

for iv, v in enumerate(V):
    ax.text(v[0], v[1], str(iv))

ax.autoscale()
ax.axis('equal')

plt.show()

n_face_groups = torch.tensor(np.array([len(face_group) for face_group in face_groups], dtype=int))

n_max_face_groups = max([len(f) for f in face_groups])
face_groups = np.array([
    np.pad(face_group, (0, n_max_face_groups - len(face_group)), 'constant', constant_values=-1)
    for face_group in face_groups
], dtype=int)
face_groups = torch.tensor(face_groups)

mask_face_groups = np.ones([face_groups.shape[0], face_groups.shape[1]])
for i_f, f in enumerate(face_groups):
    num_neg_1 = (f == -1).sum()
    if num_neg_1:
        mask_face_groups[i_f][-num_neg_1: ] = 0

mask_face_groups = mask_face_groups.reshape(mask_face_groups.shape[0], mask_face_groups.shape[1], 1)
mask_face_groups = torch.tensor(mask_face_groups)


def group_flatness(V):
    Nf = torch.cross(V[F[:, 0]] - V[F[:, 1]], V[F[:, 0]] - V[F[:, 2]])
    Nf = Nf / torch.sqrt((Nf ** 2).sum(1)).reshape(-1, 1)
    group_normals = (Nf[face_groups] * mask_face_groups)
    # group_normals = np.array([[[0, 0, 1], [0, 0, -2], [0, 0, 1]], [[0, 0, 1], [0, 0, -1], [0, 0, 0]]])
    group_normals_mean = group_normals.sum(1) / n_face_groups.reshape(-1, 1)
    group_normals_mean = group_normals_mean.reshape(group_normals_mean.shape[0], 1, group_normals_mean.shape[1]) * mask_face_groups
    normal_var_sum = ((group_normals - group_normals_mean) ** 2).sum()
    
    return normal_var_sum


es = []
for f in F:
    for i in range(3):
        j = (i+1) % 3
        if (f[i], f[j]) not in es and (f[j], f[i]) not in es:
            es.append((f[i], f[j]))

E = np.array(es, dtype=int)

l2 = torch.Tensor(((V[E][:, 0, :] - V[E][:, 1, :]) ** 2).sum(1))

E = torch.tensor(E)


def drag_energy(V, ids, ps):
    # ids: nP, indices of vertices under manipulation
    # ps: nP x 3, positions of the manipulated vertices
    return ((V[ids].reshape(-1, 3) - ps.reshape(-1, 3)) ** 2).sum(1).mean()


ids_drag = np.array([0], dtype=int)
ps_drag = np.array([[10, 10, 10]], dtype=float)

ids_drag = torch.tensor(ids_drag)
ps_drag = torch.Tensor(ps_drag)

def energy(V, requires_grad=False):
    V = V.reshape(-1, 3)
    V = torch.Tensor(V)
    V.requires_grad = True
    
    edge_energy = ((((V[E[:, 0]] - V[E[:, 1]]) ** 2).sum(1) - l2) ** 2).mean()
    
    face_group_energy = group_flatness(V)
    
    global t, ivs_shift, shifts
    
    drag_e = drag_energy(V, ivs_shift, shifts)
    
    stay_e = ((V - V0) ** 2).sum(1).mean()
    
    e = edge_energy + face_group_energy * 10 + drag_e + stay_e * 1e-2
    # print(edge_energy, face_group_energy, drag_e, stay_e)
    
    if requires_grad:
        e.backward()
        return e.detach().numpy(), V.grad.detach().numpy()
    else:
        return e.detach().numpy()

def jac(V):
    j = energy(V, requires_grad=True)[1]
    return j

ps.init()
M = ps.register_surface_mesh("mesh", V, F.numpy())


ups = np.array(ups, dtype=int)
downs = np.array(downs, dtype=int)
V[ups] += 1e-3
V[downs] -= 1e-3

ivs_shift = torch.tensor(list(shifts.keys()))
shifts = torch.Tensor([shifts[iv_shift] for iv_shift in shifts.keys()])

M2 = ps.register_surface_mesh("mesh2", V, F.numpy())

t = 0
Vs = np.array([0, 1, 0])
Ve = np.array([0, 0, 1])


def callback():
    global V, dV, t

    # A = np.eye(21)
    # lb = np.ones(21) * -1e5
    # lb[0:3] = 5
    # ub = np.ones(21) * 1e5
    # ub[0:3] = 5
    # c0 = scipy.optimize.LinearConstraint(A, lb, ub)
    # c1 = scipy.optimize.NonlinearConstraint(group_flatness, 0, 0)
    result = optimize.minimize(energy, x0=V.reshape(-1), jac=jac, method="SLSQP", options={'maxiter': 50})
    # print(result)
    # print(t)
    
    
    # print(result.success, result.nit, result.message, result.fun)
    
    V = result.x.reshape(-1, 3)

    if t < 1:
        t += dt
    # V[2] = Ve * t + Vs * (1 - t)
    #
    M2.update_vertex_positions(V)

ps.set_user_callback(callback)

ps.show()





################
# import glfw
#
# def key_callback(window, key, scancode, action, mods):
#
#     print(key, scancode, action, mods)
#
#
# def main():
#     # Initialize the library
#     if not glfw.init():
#         return
#     # Create a windowed mode window and its OpenGL context
#     window = glfw.create_window(640, 480, "Hello World", None, None)
#     if not window:
#         glfw.terminate()
#         return
#
#     # Make the window's context current
#     glfw.make_context_current(window)
#     glfw.set_key_callback(window, key_callback)
#
#     # Loop until the user closes the window
#     while not glfw.window_should_close(window):
#         # Render here, e.g. using pyOpenGL
#
#         print(glfw.get_cursor_pos(window), glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT))
#
#         # Swap front and back buffers
#         glfw.swap_buffers(window)
#
#         # Poll for and process events
#         glfw.poll_events()
#
#     glfw.terminate()
#
# if __name__ == "__main__":
#     main()


