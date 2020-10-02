import numpy as np
import pandas as pd
import h5py
import pickle as pk

type_dict = {
    # dyno
    'dentate':'UMI', 
    'immune':'UMI', 
    'neonatal':'UMI', 
    'mouse_brain':'UMI', 
    'planaria_full':'UMI', 
    'planaria_muscle':'UMI',
    'aging':'non-UMI', 
    'cell_cycle':'non-UMI',
    'fibroblast':'non-UMI', 
    'germline':'non-UMI',    
    'human':'non-UMI', 
    'mesoderm':'non-UMI',
    
    # dyngen
    'bifurcating_1000_2000_2':'non-UMI',
    
    # our model
    'linear':'UMI',
    'bifurcation':'UMI',
    'multifurcating':'UMI',
    'tree':'UMI',
}

def get_data(path, file_name):
    if file_name in ['dentate', 'immune', 'aging']
    
    data = {}
    
    # file_name = 'bifurcating_1000_2000_2'
    with h5py.File(os.path.join(path,file_name+'.h5'), 'r') as f:
        data['count'] = np.array(f['count'], dtype=np.float32)
        data['grouping'] = np.array(f['grouping']).astype('U')
        if 'gene_names' in f:
            data['gene_names'] = np.array(f['gene_names']) .astype('U')
        else:
            data['gene_names'] = None
        if 'cell_ids' in f:
            data['cell_ids'] = np.array(f['cell_ids']) .astype('U')
        else:
            data['cell_ids'] = None
        if 'milestone_net' in f:
            if file_name in ['linear','bifurcation','multifurcating','tree']:
                data['milestone_net'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype='U'), 
                    columns=['from','to','w']
                ).astype({'w':np.float32})
            else:
                data['milestone_net'] = pd.DataFrame(
                    np.array(np.array(list(f['milestone_network'])).tolist(), dtype='U'), 
                    columns=['from','to']
                )
            data['root_milestone_id'] = np.array(f['root_milestone_id']).astype('U')[0]
        else:
            data['milestone_net'] = None
            data['root_milestone_id'] = None
    
    data['type'] = type_dict[file_name]
    if data['type']=='non-UMI':
        scale_factor = np.sum(data['x'],axis=1, keepdims=True)/1e6
        data['x'] = data['x']/scale_factor
    
    return data['x']