from segpinndic.DIC_importlib import pickle, np, optax, socket
from segpinndic.utils import io

from segpinndic import DIC_domains, DIC_decompositions, DIC_networks, DIC_schedulers

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
        return f"results/summaries/{self.run}/"
    @property
    def model_out_dir(self):
        return f"results/models/{self.run}/"

    def get_outdirs(self):
        io.get_dir(self.summary_out_dir)
        io.clear_dir(self.summary_out_dir)
        io.get_dir(self.model_out_dir)
        io.clear_dir(self.model_out_dir)

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

    def __init__(self, **kwargs):
        "Defines global constants for model"

        # Define run
        self.run = "test"

        # Define domain
        self.domain = DIC_domains.RectangularDomainND
        self.domain_init_kwargs = dict(
            xmin=np.array([0.]),
            xmax=np.array([1.])
            )

        # Define domain decomposition
        subdomain_xs = [np.linspace(0,1,5)]
        subdomain_ws = get_subdomain_ws(subdomain_xs, 2.99)
        self.decomposition = DIC_decompositions.RectangularDecompositionND
        self.decomposition_init_kwargs = dict(
            subdomain_xs=subdomain_xs,
            subdomain_ws=subdomain_ws,
            unnorm=(0., 1.),
            )

        # Define neural network
        self.network = DIC_networks.FCN
        self.network_init_kwargs = dict(
            layer_sizes=[1, 32, 1],
            )

        # Define scheduler
        self.n_steps = 15000
        self.scheduler = DIC_schedulers.AllActiveSchedulerND
        self.scheduler_kwargs = dict()

        # Define optimisation parameters
        self.ns = ((60,),)# batch_shape for each training constraint
        self.n_test = (200,)# batch_shape for test data
        self.optimiser = optax.adam
        self.optimiser_kwargs = dict(
            learning_rate=1e-3
            )
        self.seed = 0

        # Define summary output parameters
        self.summary_freq    = 1000# outputs train stats to command line
        self.test_freq       = 1000# outputs test stats to plot / file / command line
        self.model_save_freq = 10000
        self.show_figures = False# whether to show figures
        self.save_figures = True# whether to save figures
        self.clear_output = False# whether to clear ipython output periodically

        # other constants
        self.hostname = socket.gethostname().lower()

        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]# invokes __setitem__ in ConstantsBase


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