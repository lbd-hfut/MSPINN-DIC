from segpinndic.DIC_importlib import os, pickle, jax, jnp, np

from segpinndic.DIC_config import seed_config_txt, DIC_config_txt
from segpinndic.DIC_readImg import BufferManager, ImgDataset
from segpinndic.DIC_seedcalc import CalcSeeds, Seed_match_visualization

from segpinndic.utils.logger import logger
from segpinndic.utils.other import DictToObj
from segpinndic.DIC_trainers import FBPINNTrainer
from segpinndic.DIC_constants import Constants
from segpinndic import DIC_networks

def main(
    seed_config_path="./config/Seed_Configuration.txt",
    dic_config_path="./config/PINN-DIC-2D.txt"
    ):
    
    DIC_config = DIC_config_txt(dic_config_path)
    Seed_config = seed_config_txt(seed_config_path)
    
    ImgData = ImgDataset(DIC_config, Seed_config)
    SeedCalculator = CalcSeeds(Seed_config)
    
    N_pairs, N_roi = len(ImgData), len(BufferManager.mask)
    constants_list = []
    for roi_id in range(N_roi):
        c_ = Constants(DIC_config, roi_id)
        constants_list.append(c_)
    
    for i in range(N_pairs):
        logger.info(f"Processing imgage pair {i+1}/{N_pairs}")
        ImgData.get_image(i)
        seed_pos, seed_uv = SeedCalculator.analyze()
        if DIC_config.save_figures:
            Seed_match_visualization(
                BufferManager.refImg*255, 
                BufferManager.defImg*255,
                seed_pos, seed_uv, DIC_config.output_dir, f'seed{i+1:03d}', i+1
            )
        
        
            
            
if __name__ == "__main__":
    main()