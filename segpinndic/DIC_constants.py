from segpinndic.DIC_importlib import pickle, np, optax, socket, seed
from segpinndic.utils import io

from segpinndic import (
    DIC_assemblies,
    DIC_domains,
    DIC_decompositions,
    DIC_networks,
    DIC_schedulers,
    DIC_problem,
)
from segpinndic.DIC_readImg import BufferManager

class ConstantsBase:

    # note can set members freely, below only for index assignment
    def __getitem__(self, key):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        return getattr(self, key)
    def __setitem__(self, key, item):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        setattr(self, key, item)

    def __str__(self):
        s = repr(self) + '\n'
        for k in vars(self): s+=f"{k}: {self[k]}\n"
        return s
    
    # calculated variables
    @property
    def summary_out_dir(self):
        return f"{self.reslut_dir}/summaries/{self.run}/"
    @property
    def model_out_dir(self):
        return f"{self.reslut_dir}/models/{self.run}/"
    @property
    def mat_out_dir(self):
        return f"{self.reslut_dir}/mats/{self.run}/"
    @property
    def fig_out_dir(self):
        return f"{self.reslut_dir}/figs/{self.run}/"

    def get_outdirs(self):
        io.get_dir(self.summary_out_dir)
        io.get_dir(self.model_out_dir)
        io.get_dir(self.mat_out_dir)
        io.get_dir(self.fig_out_dir)
        
    def clear_outdirs(self):
        io.clear_dir(self.summary_out_dir)
        io.clear_dir(self.model_out_dir)
        io.clear_dir(self.mat_out_dir)
        io.clear_dir(self.fig_out_dir)

    def save_constants_file(self):
        "Save a constants to file in self.summary_out_dir"
        with open(self.summary_out_dir + f"constants_{self.run}.txt", 'w') as f:
            for k in vars(self): f.write(f"{k}: {self[k]}\n")
        with open(self.summary_out_dir + f"constants_{self.run}.pickle", 'wb') as f:
            pickle.dump(vars(self), f)

    @property
    def constants_file(self):
        return self.summary_out_dir + f"constants_{self.run}.pickle"
    
    
# main constants class
class Constants(ConstantsBase):

    def __init__(self, DICconfig, roi_id):
        "Defines global constants for model"

        # Define results directories
        self.reslut_dir = DICconfig.output_dir
        
        # image shape
        self.roi_id = roi_id
        self.pair_idx = None  # set by DIC_analysis before training
        roi = BufferManager.mask[roi_id]
        roi = np.asarray(roi)
        assert roi.ndim == 2, "ROI must be (H, W) bool array"
        H, W = roi.shape
        
        ys, xs = np.where(roi)
        if len(xs) == 0:
            raise ValueError("ROI is empty")

        xmin_roi = xs.min()
        xmax_roi = xs.max()
        ymin_roi = ys.min()
        ymax_roi = ys.max()
        
        self.roi_bbox = (xmin_roi, xmax_roi, ymin_roi, ymax_roi)

        # Define domain
        self.domain = DIC_domains.RectangularDomainND
        self.domain_init_kwargs = dict(
            xmin=np.array([xmin_roi, ymin_roi]),
            xmax=np.array([xmax_roi, ymax_roi])
        )
        
        # Define problem
        loss_name = DICconfig.loss_fun
        if not hasattr(DIC_problem, loss_name):
            raise ValueError(f"Unknown network: {loss_name}")
        self.problem = getattr(DIC_problem, loss_name)
        self.problem_init_kwargs = dict(
            ref_img = None,
            QKBQKT_def = None,
            mask = None,
            degree = 5,
            znssd_kernel_size = DICconfig.znssd_kernel_size,
        )

        # Define domain decomposition
        nx, ny = getattr(DICconfig, "n_subdomains", [4, 8])
        if nx*ny == 1:
            self.subdomain_xs = [
                np.linspace(xmin_roi, xmax_roi, 5), 
                np.linspace(ymin_roi, ymax_roi, 5)
                ]
            self.subdomain_ws = get_subdomain_ws(self.subdomain_xs, 2.4)
        else:
            nodes_xs = [
                np.linspace(xmin_roi, xmax_roi, nx+1), 
                np.linspace(ymin_roi, ymax_roi, ny+1)
            ]
            self.subdomain_ws = [
                1.6 * np.min(np.diff(nodes)) * np.ones(nx_or_ny_current)                 for nodes, nx_or_ny_current in zip(nodes_xs, [nx, ny])
            ]
            self.subdomain_xs = [
                (nodes[:-1] + nodes[1:]) / 2 
                for nodes in nodes_xs
            ]
            
        self.decomposition = DIC_decompositions.RectangularDecompositionND
        self.decomposition_init_kwargs = dict(
            subdomain_xs=self.subdomain_xs,
            subdomain_ws=self.subdomain_ws,
            unnorm=(np.array([0., 0.]), np.array([1., 1.])),
            )

        # Define global multi-subdomain assembly.  Missing config preserves the
        # historical cosine POU and its output-directory naming.
        self.assembly_method = getattr(DICconfig, "assembly_method", "cosine_pou")
        self.assembly_explicit = hasattr(DICconfig, "assembly_method")
        self.assembly = DIC_assemblies.get_assembly(self.assembly_method)
        self.assembly_init_kwargs = dict(
            eps=getattr(DICconfig, "assembly_eps", 1e-8),
            bicubic_cell_size=getattr(DICconfig, "bicubic_cell_size", 4.0),
            correction_support_ratio=getattr(
                DICconfig, "correction_support_ratio", 0.5),
            residual_gate=getattr(DICconfig, "residual_gate", 1.0),
            global_coordinate_scale=getattr(
                DICconfig, "global_coordinate_scale", 1.0),
            p2_regularization=getattr(
                DICconfig, "p2_regularization", 1.0e-8),
            freeze_backbone_stage2=getattr(
                DICconfig, "freeze_backbone_stage2", False),
            local_projection_order=(
                -1 if str(getattr(
                    DICconfig, "local_projection_order", "none")).lower()
                == "none" else int(DICconfig.local_projection_order)),
            projection_regularization=getattr(
                DICconfig, "projection_regularization", 1.0e-8),
            projection_point_count=getattr(
                DICconfig, "projection_point_count", 256),
        )
        if self.assembly_method in (
                "hermite_1d_c1", "bicubic_hermite_c1",
                "c1_minimal_correction_pou") and ny != 1:
            raise ValueError(
                f"{self.assembly_method} is currently qualified only for "
                f"x-chain decomposition [nx, 1], received [{nx}, {ny}]."
            )

        # Define neural network
        net_name = DICconfig.network
        if not hasattr(DIC_networks, net_name):
            raise ValueError(f"Unknown network: {net_name}")
        self.network = getattr(DIC_networks, net_name)
        self.network_init_kwargs = dict(
            layer_sizes=[2] + DICconfig.hidden_units + [2],
            )
        self.global_network_init_kwargs = dict(
            layer_sizes=[2] + getattr(
                DICconfig, "global_hidden_units", [16, 16, 16, 16]) + [2],
        )
        self.field_mode = getattr(DICconfig, "field_mode", "full_local")
        self.global_field_type = getattr(DICconfig, "global_field_type", "none")
        self.bspline_grid = tuple(getattr(DICconfig, "bspline_grid", [6, 6]))
        self.bspline_degree = getattr(DICconfig, "bspline_degree", 3)
        self.freeze_backbone_stage2 = getattr(
            DICconfig, "freeze_backbone_stage2", False)
        self.local_projection_order = self.assembly_init_kwargs[
            "local_projection_order"]
        self.projection_weighting = getattr(
            DICconfig, "projection_weighting", "uniform")
        self.projection_regularization = self.assembly_init_kwargs[
            "projection_regularization"]
        self.projection_point_count = self.assembly_init_kwargs[
            "projection_point_count"]
        if self.projection_weighting != "uniform":
            raise ValueError("projection_weighting is currently fixed at uniform.")
        if self.global_field_type == "bspline":
            if self.bspline_grid not in ((4, 4), (6, 6), (8, 8)):
                raise ValueError("bspline_grid must be [4,4], [6,6], or [8,8].")
            if self.bspline_degree != 3:
                raise ValueError("bspline_degree is fixed at 3.")
        self.assembly_extra_init_kwargs = dict(
            **self.global_network_init_kwargs,
            bspline_grid=self.bspline_grid,
            bspline_degree=self.bspline_degree,
            xmin=self.domain_init_kwargs["xmin"],
            xmax=self.domain_init_kwargs["xmax"],
        )

        # Define scheduler
        self.n_steps = getattr(DICconfig, "adam_epochs", 15000)
        self.backbone_stage1_steps = getattr(
            DICconfig, "backbone_stage1_steps", 0)
        self.backbone_stage2_steps = getattr(
            DICconfig, "backbone_stage2_steps", 0)
        self.backbone_stage2_gate = getattr(
            DICconfig, "backbone_stage2_gate", 0.25)
        self.backbone_residual_schedule = getattr(
            DICconfig, "backbone_residual_schedule", "none")
        if self.assembly.uses_backbone_schedule:
            if self.backbone_stage1_steps <= 0:
                raise ValueError("backbone-first assemblies require a positive Stage 1.")
            if self.backbone_stage1_steps + self.backbone_stage2_steps >= self.n_steps:
                raise ValueError("backbone stages must leave a positive Stage 3.")
            if self.backbone_residual_schedule not in ("direct", "progressive"):
                raise ValueError("backbone_residual_schedule must be direct or progressive.")
        scheduler_name = DICconfig.train_schedulers
        if not hasattr(DIC_schedulers, scheduler_name):
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        self.scheduler = getattr(DIC_schedulers, scheduler_name)
        if scheduler_name == "AllActiveSchedulerND":
            self.scheduler_kwargs = dict()
            scheduler_rename = "All" 
        elif scheduler_name == "PointSchedulerRectangularND":
            self.scheduler_kwargs = dict(
                point=np.array([(xmin_roi+xmax_roi)/2, (ymin_roi+ymax_roi)/2])
            )
            scheduler_rename = "Point"
        elif scheduler_name == "LineSchedulerRectangularND":
            scheduler_rename = "Line"
            if nx > ny:
                self.scheduler_kwargs = dict(
                    point=np.array([xmin_roi]),
                    iaxis=1
                )
            else:
                self.scheduler_kwargs = dict(
                    point=np.array([ymin_roi]),
                    iaxis=0
                )

        # Define optimisation parameters
        self.ns = ((1,),)# batch_shape for placeholder
        self.n_test = (200,)# batch_shape for test data
        self.seed_lr = getattr(DICconfig, "adam_lr", 1e-3)
        self.dic_lr = getattr(DICconfig, "dic_lr", self.seed_lr)
        self.optimiser = optax.adam
        self.optimiser_kwargs = dict(
            learning_rate=self.dic_lr
            )
        self.seed = seed

        # L-BFGS refinement parameters
        self.lbfgs_epochs = getattr(DICconfig, "lbfgs_epochs", 0)
        self.lbfgs_history_size = getattr(DICconfig, "lbfgs_history_size", 10)
        self.lbfgs_maxls = getattr(DICconfig, "lbfgs_maxls", 15)
        self.lbfgs_lr = getattr(DICconfig, "lbfgs_lr", 1.0)

        # seed supervised pre-training
        self.seed_pos = None        # (N,2) seed point coordinates
        self.seed_uv = None         # (N,2) seed point displacements
        self.seed_train_epochs = 0  # number of seed pre-training steps
        self.seed_smooth_lambda = getattr(DICconfig, "seed_smooth_lambda", 0.0)    # gradient-norm penalty weight
        self.seed_smooth_npoints = getattr(DICconfig, "seed_smooth_npoints", 0)    # collocation points for smoothness

        # Define summary output parameters
        self.summary_freq = getattr(DICconfig, "summary_freq", 1000)        # outputs train stats to command line
        self.test_freq = getattr(DICconfig, "test_freq", 1000)              # outputs test stats to plot / file / command line
        self.test_flag = getattr(DICconfig, "test_flag", False)              # whether to print test information
        self.model_save_freq = getattr(DICconfig, "model_save_freq", 10000)
        self.show_figures = getattr(DICconfig, "show_figures", False)       # whether to show figures
        self.save_figures = getattr(DICconfig, "save_figures", True)        # whether to save figures
        self.clear_output = getattr(DICconfig, "clear_output", False)       # whether to clear ipython output periodically
        self.stage_snapshot_steps = tuple(getattr(
            DICconfig, "stage_snapshot_steps", []))

        # other constants
        self.hostname = socket.gethostname().lower()
        
        # Define run
        if nx*ny == 1:
            self.run = net_name + "_" + loss_name
        else:
            self.run = net_name + "_" + scheduler_rename + "_" + f"{nx}x{ny}"  + "_" + loss_name
        if self.assembly_explicit and nx*ny > 1:
            self.run += "_" + self.assembly_method

    def load_kwargs(self, **kwargs):
        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]# invokes __setitem__ in ConstantsBase

    def residual_gate_at_step(self, completed_steps):
        """Structural residual gate for a v2 backbone-first training schedule."""

        if not self.assembly.uses_backbone_schedule:
            return 1.0
        step = int(completed_steps)
        stage1 = self.backbone_stage1_steps
        stage2 = self.backbone_stage2_steps
        if getattr(self.assembly, "residual_activates_at_stage1", False):
            return 0.0 if step < stage1 else 1.0
        if step <= stage1:
            return 0.0
        if self.backbone_residual_schedule == "direct":
            return 1.0
        if step <= stage1 + stage2:
            return self.backbone_stage2_gate*(step-stage1)/stage2
        stage3 = self.n_steps-stage1-stage2
        return self.backbone_stage2_gate + (1.0-self.backbone_stage2_gate)*(
            step-stage1-stage2)/stage3

def print_c_dicts(c_dicts):
    "Pretty print a list of c_dicts"

    # get full list of keys
    keys = []
    for c_dict in c_dicts[::-1]:
        for k in c_dict.keys():
            if k not in keys: keys.append(k)

    for k in keys:
        print(f"{k}: ",end="")
        for i,c_dict in enumerate(c_dicts):
            if k in c_dict.keys(): item=str(c_dict[k])
            else: item='None'
            if i == len(c_dicts)-1: print(f"{item}",end="")
            else: print(f"{item} | ",end="")
        print("")
        
def get_subdomain_ws(subdomain_xs, width):
    return [width*np.min(np.diff(x))*np.ones_like(x) for x in subdomain_xs]


if __name__ == "__main__":
    from segpinndic.DIC_config import DIC_config_txt
    config = DIC_config_txt("./config/PINN-DIC-2D.txt")
    
    # BufferManager.refImg = np.zeros((100, 200))# dummy image for testing
    # constants = Constants(config)
    # print(constants)
