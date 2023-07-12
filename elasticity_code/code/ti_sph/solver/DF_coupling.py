import taichi as ti
from .DFSPH import *


class DFSPH_layer:
    def __init__(self, solvers, types, number_density=False):
        self.available_types = ["static", "elastic", "fluid"]
        self.dynamic_types = ["elastic", "fluid"]
        self.compressible_types = []

        self.solver_list = solvers
        self.type_list = types
        self.if_number_density = number_density
        self.check_pair()

        for solver, type in zip(self.solver_list, self.type_list):
            solver.background_neighb_grid.register(obj_pos=solver.obj_pos)

    def check_pair(self):
        if len(self.solver_list) != len(self.type_list):
            raise Exception(
                "DF_coupling(): length of solver_list and type_list are not equal."
            )
        for type in self.type_list:
            if not type in self.available_types:
                raise Exception("DF_coupling() does not support '" + type + "' object.")

    def is_dynamic(self, solver_type):
        if solver_type in self.dynamic_types:
            return True
        else:
            return False

    def is_compressible(self, solver_type):
        if solver_type in self.compressible_types:
            return True
        else:
            return False

    def loop(self):
        for solver, type in zip(self.solver_list, self.type_list):

            solver.set_vel_adv()

            solver.clear_psi()
            solver.clear_alpha()
            solver.comp_iter_count[None] = 0

            # loop including itself
            for neighbour_solver in self.solver_list:
                if self.if_number_density:
                    solver.compute_number_density_psi_from(neighbour_solver)
                else:
                    solver.compute_psi_from(neighbour_solver)
                    
            #------------------------------------------------------------------------------
            # solver.obj.attr_set_arr(solver.obj.implicit_sph.sph_density, solver.obj_sph_psi)
            #------------------------------------------------------------------------------    


            if self.is_dynamic(type):
                # loop including itself
                for neighbour_solver in self.solver_list:
                    solver.compute_alpha_1_from(neighbour_solver)

            # loop including itself
            for neighbour_solver, neighbour_type in zip(
                self.solver_list, self.type_list
            ):
                if self.is_dynamic(neighbour_type):
                    solver.compute_alpha_2_from(neighbour_solver)

            solver.compute_alpha_self()

        while not self.is_global_incompressible():
            self.comp_iter()
        # solver.compute_delta_psi_self()
        # solver.ReLU_delta_psi()
        # solver.check_if_compressible()
        
        for solver in self.solver_list:
            solver.obj.attr_set_arr(
                obj_attr=solver.obj_vel,
                val_arr=solver.obj_vel_adv,
            )

    def is_global_incompressible(self):
        is_incompressible = True
        for solver in self.solver_list:
            if solver.is_compressible():
                is_incompressible = False
        return is_incompressible

    def comp_iter(self):
        for solver, type in zip(self.solver_list, self.type_list):
            solver.comp_iter_count[None] += 1
            solver.compute_delta_psi_self()

            # loop including itself
            if self.if_number_density:
                for neighbour_solver in self.solver_list:
                    solver.compute_delta_numbder_density_psi_advection_from(neighbour_solver)
            else:
                for neighbour_solver in self.solver_list:
                    solver.compute_delta_psi_advection_from(neighbour_solver)

            solver.ReLU_delta_psi()
            solver.check_if_compressible()

        for solver, type in zip(self.solver_list, self.type_list):
            if self.is_dynamic(type):
                # loop including itself
                for neighbour_solver in self.solver_list:
                    solver.update_vel_adv_from(neighbour_solver)