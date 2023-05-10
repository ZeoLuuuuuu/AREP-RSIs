


#-------------------------- Common
checkpoints_dir = r'C:\\Users\sunha\Desktop\pADV\Adv_Detect\\'
adv_data_dir = 'D:\\Code_all\\detectors_review-main\\adv_data\\'
adv_data_gray_dir = 'D:\\Code_all\\detectors_review-main\\adv_data\\gray\\'
DATASETS = ['UC']
ATTACK = ['Deepfool']
          # 'PGD','FGSM','CW','HopSkipJump']
          # 'fgsm_0.5','fgsm_0.03125','fgsm_0.0625','fgsm_0.125', 'fgsm_0.25', 'fgsm_0.3125'
          #   'bim_0.03125','bim_0.0625','bim_0.125', 'bim_0.25', \
          #   'cwi',
          #   'pgd2_1', 'pgd2_1.5', 'pgd2_2', 'pgd2_0.5', 'pgd2_0.25', 'pgd2_0.125','pgd2_0.3125'
          # ]


ALL_ATTACKS = ['fgsm_0.03125', 'fgsm_0.0625', 'fgsm_0.125', 'fgsm_0.25', 'fgsm_0.3125',\
            'bim_0.03125', 'bim_0.0625', 'bim_0.125', 'bim_0.25', 'bim_0.3125',\
            'pgd1_5', 'pgd1_10', 'pgd1_15', 'pgd1_20', 'pgd1_25',\
            'pgd2_0.25','pgd2_0.3125', 'pgd2_0.5', 'pgd2_1', 'pgd2_1.5', 'pgd2_2',\
            'pgdi_0.03125', 'pgdi_0.0625', 'pgdi_0.125', 'pgdi_0.25',\
            'cwi', \
            'hca_0.03125', 'hca_0.0625', 'hca_0.125', 'hca_0.3125', 'hca_0.5',\
            'df',\
            'sa', 'hop', 'sta'
           ]
fieldnames = ['type',	'nsamples',	'acc_suc',	'acc',	'tpr',	'fpr',	'tp',	'ap',	'fb',	'an',	'tprs',	'fprs',	'auc']
env_param = 'env /home/aaldahdo/adv_dnn/venv/bin/python -- ' 
detectors_dir = '/home/aaldahdo/detectors/'
results_path = '/home/aaldahdo/detectors/results/'
#-------------------------- detect KD_BU
# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00, 'tiny': 0.26}
# BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00, 'tiny': 0.26}
#[0.1, 0.16681005372000587, 0.2782559402207124, 0.46415888336127786, 0.774263682681127, 1.291549665014884, 2.1544346900318834, 3.593813663804626, 5.994842503189409, 10.0]
kd_bu_results_dir = 'D:\\Code_all\\detectors_review-main\\results\\kd_bu\\'
#-------------------------- detect LID
k_nn = [20, 30, 20, 30]
lid_results_dir = 'D:\\Code_all\\detectors_review-main\\results\\lid\\'

#-------------------------- detect MagNet
magnet_results_dir = '/home/aaldahdo/detectors/results/magnet/'

#-------------------------- detect FS
fs_results_dir = 'D:\\Code_all\\detectors_review-main\\results\\fs\\'
fs_results_gray_dir = '/home/aaldahdo/detectors/results/fs/gray/'

#-------------------------- detect DNR
layer_names = [[['l_16'], ['l_14'], ['l_10']],\
                [['l_31'], ['l_26'], ['l_21']],\
                [['l_16'], ['l_14'], ['l_10']],\
                [['bn'], ['conv5_block17_0_bn'], ['pool4_bn']]]
dnr_results_dir = '/home/aaldahdo/detectors/results/dnr/'
dnr_results_gray_dir = '/home/aaldahdo/detectors/results/dnr/gray/'

#-------------------------- detect NSS
pgd_percent = [[0.02, 0.1, 0.18, 0.3, 0.3,0.1], [0.3, 0.3, 0.1, 0.1, 0.1,0.1], [0.3, 0.3, 0.1, 0.1, 0.1,0.1], [0.3, 0.3, 0.1, 0.1, 0.1,0.1]]
nss_results_dir ='D:\\Code_all\\detectors_review-main\\adv_data\\results\\nss\\'

#-------------------------- detect SFAD
sfad_results_dir = '/home/aaldahdo/detectors/results/sfad/'
sfad_results_gray_dir = '/home/aaldahdo/detectors/results/sfad/gray/'

#-------------------------- detect NIC
nic_results_dir = '/home/aaldahdo/detectors/results/nic/'
nic_results_gray_dir = '/home/aaldahdo/detectors/results/nic/gray/'
nic_layers_dir = '/media/aaldahdo/SAMSUNG/nic/'
nic_layers_gray_dir = '/media/aaldahdo/SAMSUNG/nic/gray/'


