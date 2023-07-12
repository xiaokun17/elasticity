import taichi as ti
import ti_sph as tsph
from ti_sph.solver import DFSPH_layer
from ti_sph.solver.ISPH_Elastic import ISPH_Elastic
from ti_sph.solver.DFSPH import DFSPH
import os
import time


ti.init(arch=ti.cuda, device_memory_GB=9)


"""""" """ CONFIG """ """"""
# CONFIG
config_capacity = ["info_space", "info_discretization", "info_sim", "info_gui"]
config = tsph.Config(dim=3, capacity_list=config_capacity)

# space
config_space = ti.static(config.space)
config_space.dim[None] = 3
config_space.lb[None] = [-8, -8, -8]
config_space.rt[None] = [8, 8, 8]

# sim
config_sim = ti.static(config.sim)
config_sim.gravity[None] = ti.Vector([0, -9.8, 0])
config_sim.kinematic_vis[None] = 1e-1

# discretization
config_discre = ti.static(config.discre)
config_discre.part_size[None] = 0.05
config_discre.cs[None] = 220
config_discre.cfl_factor[None] = 0.5
config_discre.dt[None] = (
    tsph.fixed_dt(
        config_discre.cs[None],
        config_discre.part_size[None],
        config_discre.cfl_factor[None],
    )
    
)
config_discre.inv_dt[None] = 1 / config_discre.dt[None]
standart_dt = config_discre.dt[None]

# gui
config_gui = ti.static(config.gui)
config_gui.res[None] = [1920, 1080]
config_gui.frame_rate[None] = 60
config_gui.cam_fov[None] = 55
config_gui.cam_pos[None] = [6.0, 1.0, 0.0]
config_gui.cam_look[None] = [0.0, 0.0, 0.0]
config_gui.canvas_color[None] = [0.2, 0.2, 0.6]
config_gui.ambient_light_color[None] = [0.7, 0.7, 0.7]
config_gui.point_light_pos[None] = [2, 1.5, -1.5]
config_gui.point_light_color[None] = [0.8, 0.8, 0.8]

rho0=100
rho1=1000
rhob=1000

"""""" """ OBJECT """ """"""
elastic_list = []
elastic_neighb_list = []
elastic_neighb_0_list = []
elastic_solver_list = []
elastic_df_solver_list = []
# ELASTIC
elastic_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_ISPH_Elastic",
    "node_implicit_sph",
]
elastic_list.append(
    tsph.Node(
        dim=config_space.dim[None],
        id=0,
        node_num=int(1e4),
        capacity_list=elastic_capacity,
    )
)

elastic_node_num1 = elastic_list[0].push_cube_with_basic_attr(
    lb=ti.Vector([-0.6, -0.1, -0.8]),
    rt=ti.Vector([0.6, 0.3, 0.8]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    rest_density=rho0,
    color=ti.Vector([1, 0, 0]),
)

elastic_list.append(
    tsph.Node(
        dim=config_space.dim[None],
        id=0,
        node_num=int(1e4),
        capacity_list=elastic_capacity,
    )
)

elastic_node_num2 = elastic_list[1].push_cube_with_basic_attr(
    lb=ti.Vector([-0.6, -0.6, -1]),
    rt=ti.Vector([0.6, -0.2, 1]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    rest_density=rho1,
    color=ti.Vector([0, 1, 1]),
)
print('elastic_node_num', elastic_node_num1,elastic_node_num2)
 


# BOUND
bound_capacity = [
    "node_basic",
    "node_color",
    "node_sph",
    "node_implicit_sph",
]
bound = tsph.Node(
    dim=config_space.dim[None],
    id=0,
    node_num=int(2e5),
    capacity_list=bound_capacity,
)

bound_node_num = bound.push_box_with_basic_attr(
    lb=ti.Vector([-0.8, -0.8, -1.2]),
    rt=ti.Vector([0.8, 0.5, 1.2]),
    span=config_discre.part_size[None],
    size=config_discre.part_size[None],
    layers=2,
    rest_density=rhob,
    color=ti.Vector([0.3, 0.3, 0.3]),
)

print('bound_node_num', bound_node_num)


"""""" """ COMPUTE """ """"""
# /// --- NEIGHB --- ///
search_template = tsph.Neighb_search_template(
    dim=config_space.dim[None],
    search_range=1,
)

for elastic in elastic_list:
    elastic_neighb_list.append(
        tsph.Neighb_grid(
            obj=elastic,
            dim=config_space.dim[None],
            lb=config_space.lb,
            rt=config_space.rt,
            cell_size=config_discre.part_size[None] * 2,
        )
    )
    elastic_neighb_0_list.append(
        tsph.Neighb_grid(
            obj=elastic,
            dim=config_space.dim[None],
            lb=config_space.lb,
            rt=config_space.rt,
            cell_size=config_discre.part_size[None] * 2,
        )
    )
bound_neighb_grid = tsph.Neighb_grid(
    obj=bound,
    dim=config_space.dim[None],
    lb=config_space.lb,
    rt=config_space.rt,
    cell_size=config_discre.part_size[None] * 2,
)


# /// --- INIT SOLVER --- ///
# /// ISPH_Elastic ///
for elastic, neighb, neighb_0 in zip(
    elastic_list, elastic_neighb_list, elastic_neighb_0_list
):
    elastic_solver_list.append(
        ISPH_Elastic(
            obj=elastic,
            dt=config_discre.dt[None],
            background_neighb_grid=neighb,
            background_neighb_grid_0=neighb_0,
            search_template=search_template,
        )
    )

def tolame(E, nu):
    lame_mu = E / (2 * (1 + nu))
    lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return lame_mu, lame_lambda

lame_mu, lame_lambda = tolame(1e6, 0.4)

elastic_list[0].attr_set(elastic_list[0].elastic_sph.lame_mu, lame_mu)
elastic_list[0].attr_set(elastic_list[0].elastic_sph.lame_lambda, lame_mu)
elastic_list[1].attr_set(elastic_list[1].elastic_sph.lame_mu, lame_mu)
elastic_list[1].attr_set(elastic_list[1].elastic_sph.lame_lambda, lame_mu)



for elastic, neighb in zip(elastic_list, elastic_neighb_list):
    elastic_df_solver_list.append(
        DFSPH(
            obj=elastic,
            dt=config_discre.dt[None],
            background_neighb_grid=neighb,
            search_template=search_template,
            port_sph_psi="implicit_sph.sph_compression_ratio",
            port_rest_psi="implicit_sph.one",
            port_X="basic.rest_volume",
        )
    )

bound_df_solver = DFSPH(
    obj=bound,
    dt=config_discre.dt[None],
    background_neighb_grid=bound_neighb_grid,
    search_template=search_template,
    port_sph_psi="implicit_sph.sph_compression_ratio",
    port_rest_psi="implicit_sph.one",
    port_X="basic.rest_volume",
)

solvers = (
    [bound_df_solver] + elastic_df_solver_list + elastic_solver_list
)

coupling_solver = [bound_df_solver] + elastic_df_solver_list
solver_type = ["static"]
for i in range(len(elastic_df_solver_list)):
    solver_type.append("elastic")

df_solver_layer = DFSPH_layer(coupling_solver, solver_type)

sim_time = 0.0

# /// --- LOOP --- ///
def loop():
    global sim_time
    sim_time += config_discre.dt[None]
    #  /// neighb search ///
    for elastic, neighb in zip(elastic_list, elastic_neighb_list):
        neighb.register(obj_pos=elastic.basic.pos)
    bound_neighb_grid.register(obj_pos=bound.basic.pos)


    #  /// elastic sim  ///
    for elastic, elastic_solver in zip(elastic_list, elastic_solver_list):
        elastic.clear(elastic.basic.force)
        elastic.clear(elastic.basic.acc)

        #  /// advection  ///
        elastic_solver.compute_vis(
            kinetic_vis_coeff=config_sim.kinematic_vis,
            output_acc=elastic.basic.acc,
        )
        elastic.attr_add(
            obj_attr=elastic.basic.acc,
            val=config_sim.gravity,
        )

        elastic_solver.time_integral_arr(
            obj_frac=elastic.basic.acc,
            obj_output_int=elastic.basic.vel,
        )

        elastic.clear(elastic.basic.force)
        elastic.clear(elastic.basic.acc)

        elastic_solver.internal_loop(output_force=elastic.basic.force)    

        elastic_solver.update_acc(
            input_force=elastic.basic.force,
            output_acc=elastic.basic.acc,
        )

        elastic_solver.time_integral_arr(
            obj_frac=elastic.basic.acc,
            obj_output_int=elastic.basic.vel,
        )

    #  /// df sim  ///
    df_solver_layer.loop()

    # /// update vel to pos ///
    for elastic, elastic_solver in zip(elastic_list, elastic_solver_list):
        elastic_solver.time_integral_arr(
            obj_frac=elastic.basic.vel,
            obj_output_int=elastic.basic.pos,
        )
# /// --- END OF LOOP --- ///


# /// --- GUI --- ///
gui = tsph.Gui(config_gui)
gui.env_set_up()
while gui.window.running:
    gui.monitor_listen()
    if gui.op_system_run == True:
        loop()
    if gui.op_refresh_window:
        gui.scene_setup()

        for elastic in elastic_list:
            gui.scene_add_parts(elastic, size=config_discre.part_size[None])

        if gui.show_bound:
            gui.scene_add_parts(bound, size=config_discre.part_size[None])
            
        gui.scene_render()
        gui.window.show()

