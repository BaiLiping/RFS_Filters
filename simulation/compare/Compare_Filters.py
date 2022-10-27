"""
Compare Filters using GOSPA metric.
"""

import pickle
from matplotlib import pyplot as plt

if __name__ == '__main__':
    """ path_to_save_data = 'D:Tech_Resource/Paper_Resource/Perception_R_or_RC_Fusion_with_BingZhu_Project/Projects/Project_3/compare/' """
    path_to_save_data = '/gs/home/zhubing/Radar_Perception_Project/Project_3/compare/scenario1/'

    # Extract Data GM_PHD
    path =  path_to_save_data + 'gm_phd/gospa_record.pickle'
    f = open(path, 'rb')
    phd_gospa_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'gm_phd/gospa_localization_record.pickle'
    f = open(path, 'rb')
    phd_gospa_localization_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'gm_phd/gospa_missed_record.pickle'
    f = open(path, 'rb')
    phd_gospa_missed_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'gm_phd/gospa_false_record.pickle'
    f = open(path, 'rb')
    phd_gospa_false_record = pickle.load(f)
    f.close()
    
    # Extract Data  GM_CPHD
    path =  path_to_save_data + 'pmb/gospa_record.pickle'
    f = open(path, 'rb')
    pmb_gospa_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'pmb/gospa_missed_record.pickle'
    f = open(path, 'rb')
    pmb_gospa_missed_record = pickle.load(f)
    f.close()
    
    path =  path_to_save_data + 'pmb/gospa_localization_record.pickle'
    f = open(path, 'rb')
    pmb_gospa_localization_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'pmb/gospa_false_record.pickle'
    f = open(path, 'rb')
    pmb_gospa_false_record = pickle.load(f)
    f.close()
    

    # Extract Data  PMBM
    path =  path_to_save_data + 'pmbm/gospa_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'pmbm/gospa_missed_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_missed_record = pickle.load(f)
    f.close()
    
    path =  path_to_save_data + 'pmbm/gospa_localization_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_localization_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'pmbm/gospa_false_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_false_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'jpda/gospa_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'jpda/gospa_missed_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_missed_record = pickle.load(f)
    f.close()
    
    path =  path_to_save_data + 'jpda/gospa_localization_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_localization_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'jpda/gospa_false_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_false_record = pickle.load(f)
    f.close()


    path =  path_to_save_data + 'pda/gospa_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'pda/gospa_missed_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_missed_record = pickle.load(f)
    f.close()
    
    path =  path_to_save_data + 'pda/gospa_localization_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_localization_record = pickle.load(f)
    f.close()

    path =  path_to_save_data + 'pda/gospa_false_record.pickle'
    f = open(path, 'rb')
    pmbm_gospa_false_record = pickle.load(f)
    f.close()




    # Plot the metrix
    x = range(len(phd_gospa_record)) 
    plt.title("RMS GOSPA Error") 
    plt.xlabel("frame number") 
    plt.ylabel("RMS GOSPA Error") 
    plt.ylabel("gospa") 
    plt.plot(x,phd_gospa_record,label='gm_phd')
    plt.plot(x,pmb_gospa_record,label='pmb')
    plt.plot(x,pmbm_gospa_record,label='pmbm')

    plt.legend()
    plt.savefig('gospa.png')
    plt.close()

    plt.title("RMS GOSPA Normalized Localization Error") 
    plt.xlabel("frame number") 
    plt.ylabel("RMS GOSPA Normalized Localization Error") 
    plt.plot(x,phd_gospa_localization_record,label='gm_phd')
    plt.plot(x,pmb_gospa_localization_record,label='pmb') 
    plt.plot(x,pmbm_gospa_localization_record,label='pmbm')
 
    plt.legend()
    plt.savefig('gospa_localization.png')
    plt.close()


    plt.title("RMS GOSPA Missed Target Error") 
    plt.xlabel("frame number") 
    plt.ylabel("RMS GOSPA Missed Target Error") 
    plt.plot(x,phd_gospa_missed_record,label='gm_phd')
    plt.plot(x,pmb_gospa_missed_record,label='pmb') 
    plt.plot(x,pmbm_gospa_missed_record,label='pmbm')

    plt.legend()
    plt.savefig('missed.png')
    plt.close()

    plt.title("RMS GOSPA False Target Error") 
    plt.xlabel("frame number") 
    plt.ylabel("RMS GOSPA False Target Error") 
    plt.plot(x,phd_gospa_false_record,label='gm_phd')
    plt.plot(x,pmb_gospa_false_record,label='pmb') 
    plt.plot(x,pmbm_gospa_false_record,label='pmbm')
  
    plt.legend()
    plt.savefig('gospa_false.png')
    plt.close()
