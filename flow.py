import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.cpu)


@ti.kernel
def flow_forward_kernel(rgb: ti.template(), bias: ti.template(), w: ti.template(), s: ti.template()):
    for i, j in rgb:
        c = rgb[i,j]
        x = i + bias[i,j][0]
        y = j + bias[i,j][1]
        _x = ti.cast(tm.floor(x), ti.i32)
        _y = ti.cast(tm.floor(y), ti.i32)
        wx = x-_x
        wy = y-_y
        ti.atomic_add(w[_x, _y]   ,       wx * wy    )
        ti.atomic_add(w[_x+1, _y] ,   (1-wx) * wy    )
        ti.atomic_add(w[_x, _y+1] ,       wx * (1-wy))
        ti.atomic_add(w[_x+1, _y+1],  (1-wx) * (1-wy))

        ti.atomic_add(s[_x, _y]   ,       wx * wy     * c)
        ti.atomic_add(s[_x+1, _y] ,   (1-wx) * wy     * c)
        ti.atomic_add(s[_x, _y+1] ,       wx * (1-wy) * c)
        ti.atomic_add(s[_x+1, _y+1],  (1-wx) * (1-wy) * c)


# @ti.kernel
# def flow_forward(img: ti.types.ndarray(), flow: ti.types.ndarray()):
#     H, W = img.shape[:2]
#     assert flow.shape[:2] == (H, W)
#     w = ti.field(float, shape=(H, W))
#     for i, j in ti.ndrange(H, W):
#         x = i + flow[i,j,0]
#         y = i + flow[i,j,1]

#     pass

def flow_forward(img: np.ndarray, flow: np.ndarray):
    n, m = img.shape[:2]
    assert flow.shape[:2] == (n, m)
    print(img.dtype, flow.dtype)
    dtype = ti.f32
    rgb = ti.Vector.field(3,dtype, shape=(n, m))
    rgb.from_numpy(img)
    bias = ti.Vector.field(2, dtype, shape=(n, m))
    bias.from_numpy(flow)
    w = ti.field(dtype, shape=(n, m))
    s = ti.Vector.field(3, dtype, shape=(n, m))

    flow_forward_kernel(rgb, bias, w, s)
    return w.to_numpy(), s.to_numpy()

if __name__ == '__main__':
    img = np.zeros((100, 100, 3), dtype=np.float32)
    flow = np.zeros((100, 100, 2), dtype=np.float32)
    w, s = flow_forward(img, flow)
    print(w)
    print(w.shape, s.shape)