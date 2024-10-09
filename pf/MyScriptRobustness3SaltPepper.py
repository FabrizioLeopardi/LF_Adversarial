"""

            Please put this script in the folder 'Scripts' of the PULP-frontnet project to run it.
            Code to run in order to retrieve the results presented in the thesis:

            python3 MyScriptRobustness3SaltPepper.py '160x32' --batch-size 64 --epochs 10 --lr 0.01 --momentum 0.5 --seed 1 --save-model None --load-model ../Models/Frontnet160x32.pt --load-trainset '../Data/160x96OthersTrainsetAug.pickle' --load-testset '../Data/160x96StrangersTestset.pickle' --quantize --regime 'Scripts/regime.json'
            
"""
import pickle
import logging
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils import data
import numpy as np

from Frontnet.DataProcessor import DataProcessor
from Frontnet.ModelTrainer import ModelTrainer
from Frontnet.Dataset import Dataset
from Frontnet.Frontnet import FrontnetModel
from Frontnet import Utils
from Frontnet.Utils import ModelManager
from mpmath import *
import random

num_samples = 0
tot_incr = 11

def salt_pepper_pixel(test_loader, probability):
    int_prob = int(100*probability)
    for i in range(len(test_loader.dataset.data)):
        for j in range(96):
            for k in range(160):
                if (random.randint(1,100)<=int_prob):
                    if (random.randint(0,1)==0):
                        test_loader.dataset.data[i][0][j][k] = 0.0
                    else:
                        test_loader.dataset.data[i][0][j][k] = 255.0
                
    return test_loader


def main():
    args = Utils.ParseArgs()

    model_path = args.load_model
    testset_path = args.load_testset
    data_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 1}
    w, h, c = args.model_params['w'], args.model_params['h'], args.model_params['c']

    out_folder = "Results/{}x{}".format(w, c)
    os.makedirs(out_folder, exist_ok=True)

    Utils.Logger(logfile=os.path.join(out_folder, "FPTesting.log"))

    torch.manual_seed(args.seed)

    # Load the test data
    [x_test, y_test] = DataProcessor.ProcessTestData(testset_path)

    # Create the PyTorch data loaders
    test_set = Dataset(x_test, y_test)
    test_loader = data.DataLoader(test_set, **data_params)
    
    # Choose your model
    model = FrontnetModel(**args.model_params)
    
    if model_path is None:
        model_path = os.path.join(out_folder, "{}.pt".format(model.name))

    logging.info("[FPTesting] Loading model checkpoint {}".format(model_path))
    ModelManager.Read(model_path, model)

    model.eval()
    
    MSE_rho = []
    MSE_theta = []
    
    MSE_X_SWISS = []
    MSE_Y_SWISS = []
    MSE_Z_SWISS = []
    MSE_theta_SWISS = []
    
    MAE_X_SWISS = []
    MAE_Y_SWISS = []
    MAE_Z_SWISS = []
    MAE_theta_SWISS = []
    
    probability = 0.0
    for incr in range(tot_incr):
        test_loader = salt_pepper_pixel(test_loader,probability)
        probability += 0.01
        num_samples = len(test_loader.dataset.data)
        x = []
        y = []
        z = []
        t = []
        
        true_x = []
        true_y = []
        true_z = []
        true_t = []
        
        rho_norm = []
        rho_norm2 = []
        
        trainer = ModelTrainer(model)
        trainer.folderPath = out_folder
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = trainer.ValidateSingleEpoch(
        test_loader)
            
        for i in range(num_samples):
            x.append(y_pred[i][0])
            y.append(y_pred[i][1])
            z.append(y_pred[i][2])
            t.append(y_pred[i][3])
        
            true_x.append(gt_labels[i][0])
            true_y.append(gt_labels[i][1])
            true_z.append(gt_labels[i][2])
            true_t.append(gt_labels[i][3])
            
            rho_norm.append((x[i]*x[i]+y[i]*y[i]+z[i]*z[i])**(1/2))
            rho_norm2.append((true_x[i]*true_x[i]+true_y[i]*true_y[i]+true_z[i]*true_z[i])**(1/2))
            
        sum1 = 0.0
        sum2 = 0.0
        
        for i in range(num_samples):
            sum1 += (rho_norm2[i]-rho_norm[i])*(rho_norm2[i]-rho_norm[i])
            sum2 += (true_t[i]-t[i])*(true_t[i]-t[i])
       
        sum1 /= num_samples
        sum2 /= num_samples
       
        MSE_rho.append(sum1)
        MSE_theta.append(sum2)
        
        MSE, MAE, r2_score, outputs, labels = trainer.Test(test_loader)
        
        MAE_X_SWISS.append(MAE[0].item())
        MAE_Y_SWISS.append(MAE[1].item())
        MAE_Z_SWISS.append(MAE[2].item())
        MAE_theta_SWISS.append(MAE[3].item())
        
        MSE_X_SWISS.append(MSE[0].item())
        MSE_Y_SWISS.append(MSE[1].item())
        MSE_Z_SWISS.append(MSE[2].item())
        MSE_theta_SWISS.append(MSE[3].item())
        
        test_loader = data.DataLoader(test_set, **data_params)
    
    X_quadratic_error_SWISS = lambda a: MSE_X_SWISS[int(a)]
    Y_quadratic_error_SWISS = lambda a: MSE_Y_SWISS[int(a)]
    Z_quadratic_error_SWISS = lambda a: MSE_Z_SWISS[int(a)]
    theta_quadratic_error_SWISS = lambda a: MSE_theta_SWISS[int(a)]
    
    X_absolute_error_SWISS = lambda a: MAE_X_SWISS[int(a)]
    Y_absolute_error_SWISS = lambda a: MAE_Y_SWISS[int(a)]
    Z_absolute_error_SWISS = lambda a: MAE_Z_SWISS[int(a)]
    theta_absolute_error_SWISS = lambda a: MAE_theta_SWISS[int(a)]
    
    
    plot(X_quadratic_error_SWISS,[0,tot_incr-1])
    plot(Y_quadratic_error_SWISS,[0,tot_incr-1])
    plot(Z_quadratic_error_SWISS,[0,tot_incr-1])
    plot(theta_quadratic_error_SWISS,[0,tot_incr-1])
    
    plot(X_absolute_error_SWISS,[0,tot_incr-1])
    plot(Y_absolute_error_SWISS,[0,tot_incr-1])
    plot(Z_absolute_error_SWISS,[0,tot_incr-1])
    plot(theta_absolute_error_SWISS,[0,tot_incr-1])
        
if __name__ == '__main__':
    main()
