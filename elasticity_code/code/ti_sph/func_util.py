import taichi as ti
import os


@ti.func
def if_array_has_negative(vec: ti.template()):
    is_n = False
    for i in ti.static(range(vec.n)):
        if vec[i] < 0:
            is_n = True
    return is_n


@ti.kernel
def tmp_func(a: ti.f32, b: ti.f32) -> ti.i32:
    return int(ti.ceil(a / b))


@ti.func
def has_negative(vec) -> bool:
    has_negative = False
    for i in ti.static(range(vec.n)):
        if vec[i] < 0:
            has_negative = True
    return has_negative


@ti.func
def has_positive(vec) -> bool:
    has_positive = False
    for i in ti.static(range(vec.n)):
        if vec[i] > 0:
            has_positive = True
    return has_positive


@ti.func
def node_encode(node_pos: ti.template(), lb: ti.template(), cell_size: ti.f32):
    return int((node_pos - lb[None]) // cell_size)


@ti.func
def distance_1(x: ti.template(), y: ti.template()) -> ti.f32:
    vec = x - y
    return ti.sqrt(vec.dot(vec))


@ti.func
def distance_2(x_ij: ti.template()) -> ti.f32:
    return ti.sqrt(x_ij.dot(x_ij))


@ti.kernel
def clean_attr_val(obj: ti.template(), obj_attr: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj_attr[i] = 0


@ti.kernel
def clean_attr_arr(obj: ti.template(), obj_attr: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj_attr[i]*=0

@ti.kernel
def clean_attr_mat(obj: ti.template(), obj_attr: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj_attr[i]*=0
        


@ti.kernel
def set_attr(obj: ti.template(), obj_attr: ti.template(), val: ti.template()):
    for i in range(obj.info.stack_top[None]):
        obj_attr[i] = val


def modulustolame_lambda(E, nu):
    return E * nu / ((1+nu) * (1-2*nu))

def modulustolame_mu(E, nu):
    return E / (2 * (1+nu))

# @ti.kernel
# def rotation(obj: ti.template(), center:ti.template(), axis:ti.template(), radian:ti.template()):
#     for i in range(obj.info.stack_top[None]):
#         obj.basic.pos[i] = val
#         R = ti.Matrix([[1,0,0],[0,ti.cos(radian),-ti.sin(radian)],[0,ti.sin(radian),ti.cos(radian)]])
        
#     return 