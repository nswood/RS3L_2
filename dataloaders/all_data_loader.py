import numpy as np
import h5py
import glob
import tqdm
import torch
import random
import pyarrow.parquet as pq
from torch.utils.data.dataset import Dataset  # For custom datasets
from dataloaders import jetclass_dataloader


def load_W(paths,max_num_particles = 25):
    label = '/WToQQ_0'
    dt_a = []
    dt_b = []
    dt_c = []
    dt_d = []
    for cur_path in paths:
        
        a,b,c,d = jetclass_dataloader.read_file(cur_path,max_num_particles=max_num_particles,
                                             particle_features = ['part_energy', 'part_deta',
       'part_dphi', 'part_isNeutralHadron', 'part_isPhoton',
       'part_isChargedHadron', 'part_isElectron', 'part_isMuon',
       'part_charge'])
        dt_a.append(a)
        dt_b.append(b)
        dt_c.append(c)
        dt_d.append(d)
    a = np.array(dt_a)
    b = np.array(dt_b)
    c = np.array(dt_c)
    d = np.array(dt_d)
    
    a = np.vstack(a)
    b = np.vstack(b)
    c = np.vstack(c)
    d = np.vstack(d)
    
    
    a = np.transpose(a,(0, 2, 1))
    d = np.transpose(d,(0, 2, 1))
    
    x_energy = a[:,:,0]
    jet_energy = b[:,3]
    percent_e = x_energy/ np.expand_dims(jet_energy,axis =1)
    percent_e = np.expand_dims(percent_e,axis = 2)
    part_e = np.expand_dims(x_energy,axis = 2)/1000
    
    a = a[:,:,1:]
    a = np.concatenate((part_e, percent_e,a),axis = 2)
    return a,b,c,d


def load_file_qcd(file_path,max_num_particles = 25):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    df.drop('part_pid', axis=1, inplace=True)
    part_4vec = df[['part_px', 'part_py', 'part_pz', 'part_energy']].to_numpy()
    part_4vec = get_4vec(part_4vec,max_num_particles)
    a = df.to_numpy()
    jet_label = np.squeeze(a[:,0])
    zeros = np.zeros((len(jet_label),2))
    zeros[:,0] = 1
    jet_label = zeros
    jet_features = a[:,1:7]
    x_pf = get_xpart(a,max_num_particles)
    part_e = x_pf[:,:,3]
    x_pf = x_pf[:,:,4:]
    jet_e = jet_features[:,3]
    
    percent_e = part_e/ np.expand_dims(jet_e,axis =1)
    percent_e = np.expand_dims(percent_e,axis = 2)
    part_e = np.expand_dims(part_e,axis = 2)/1000
    x_pf = np.concatenate((part_e,percent_e,x_pf),axis = 2)
    
    return [x_pf, jet_features, jet_label,part_4vec]


def get_xpart(a,max_num_particles):
    stacked_arrays = [np.stack(row[7:], axis=1) for row in a]
    resulting_array = np.array(stacked_arrays,dtype=object)
    padded_array = np.zeros((99999, max_num_particles, 12), dtype=float)

    # Copy the data from 'data' into the padded_array
    for i, entry in enumerate(resulting_array):
        n = min(len(entry),max_num_particles)
        if n < max_num_particles:
            # If the sub-arrays need padding, append zero-filled sub-arrays
            padding = [np.zeros((12,), dtype=int)] * (max_num_particles - n)
            entry = np.vstack((entry, padding))
        padded_array[i, :max_num_particles, :] = entry[0:max_num_particles]
    return padded_array
def get_4vec(a,max_num_particles):
    stacked_arrays = [np.stack(row, axis=1) for row in a]
    resulting_array = np.array(stacked_arrays,dtype=object)
    padded_array = np.zeros((99999, max_num_particles, 4), dtype=float)

    # Copy the data from 'data' into the padded_array
    for i, entry in enumerate(resulting_array):
        n = min(len(entry),max_num_particles)
        if n < max_num_particles:
            # If the sub-arrays need padding, append zero-filled sub-arrays
            padding = [np.zeros((4,), dtype=int)] * (max_num_particles - n)
            entry = np.vstack((entry, padding))
        padded_array[i, :max_num_particles, :] = entry[0:max_num_particles]
    return padded_array
def extract_number(filename):
    # Extract the last 7 characters, then take the first 3 of those
    return int(filename[-7:-5])

class zpr_loader(Dataset):
    def __init__(self, all_raw_paths,qcd_raw_paths, small_feature = False, pf_size = 13,sv_size= 16,  qcd_only=True, transform=None,maxfiles=None,small_QCD= False, max_jets = None,equal_qcd=False,max_num_particles = 25):
        #super(zpr_loader, self).__init__(raw_paths)
        
        self.qcd_raw_paths = sorted(glob.glob(qcd_raw_paths+'/*parquet'))[0:maxfiles]
        self.all_raw_paths = sorted(glob.glob(all_raw_paths+'/*root'))

        if equal_qcd:
            maxfiles_jet = int(np.ceil(maxfiles/9))
            self.all_raw_paths = [f for f in self.all_raw_paths if extract_number(f)%16 < maxfiles_jet]
        else:
            self.all_raw_paths = [f for f in self.all_raw_paths if extract_number(f)%16 < maxfiles]
            
        self.max_num_particles = max_num_particles
        self.maxfiles = maxfiles
        self.small_feature =small_feature
        self.pf_size = pf_size
        self.sv_size = sv_size
        self.max_jets =max_jets
        self.small_QCD = small_QCD
        self.equal_qcd = equal_qcd
        self.fill_data()
        self.normalize_data()
        
        
        
    def calculate_offsets(self):
        for path in self.raw_paths:
            
            with h5py.File(path, 'r') as f:
                self.strides.append(f['features'].shape[0])
        self.strides = np.cumsum(self.strides)
    def is_binary(self, series):
        return np.all(np.isin(np.abs(series), [0, 1]))

    def normalize_data(self):
        self.mu =[]
        self.std = []
        
        for i in range(self.data_features.shape[2]):
            if not self.is_binary(self.data_features[:,:,i]):
#                 print(i)
                self.std.append(np.std(np.ravel(self.data_features[:,:,i])))
                self.mu.append(np.mean(np.ravel(self.data_features[:,:,i])))
                self.data_features[:,:,i] = (self.data_features[:,:,i] -self.mu[i])/(self.std[i])
#                 print(self.data_features[:,:,i])
            
            
        
    def fill_data(self):
        
        self.data_features = []
        self.data_jetfeatures = []
        self.data_truthlabel = [] 
        self.data_4vec = []
        for fi,path in enumerate(tqdm.tqdm(self.qcd_raw_paths)):
            tmp_features,tmp_jetfeatures,tmp_truthlabel, tmp_4vec = load_file_qcd(path,max_num_particles =self.max_num_particles)
            tmp_truthlabel = np.zeros((len(tmp_features),10),int)
            tmp_truthlabel[:,0] = 1
            if self.max_jets is not None:
                
                self.data_features.append(tmp_features.astype(np.float64)[0:self.max_jets])
                self.data_jetfeatures.append(tmp_jetfeatures.astype(np.float64)[0:self.max_jets])
                self.data_truthlabel.append(tmp_truthlabel.astype(np.float64)[0:self.max_jets])
                self.data_4vec.append(tmp_4vec.astype(np.float64)[0:self.max_jets])
            else:
                self.data_features.append(tmp_features.astype(np.float64))
                self.data_jetfeatures.append(tmp_jetfeatures.astype(np.float64))
                self.data_truthlabel.append(tmp_truthlabel.astype(np.float64))
                self.data_4vec.append(tmp_4vec.astype(np.float64))
        tmp_features,tmp_jetfeatures,tmp_truthlabel,tmp_4vec = load_W(self.all_raw_paths,max_num_particles =self.max_num_particles)
        
        perm_indices = np.random.permutation(len(tmp_features))
        
        tmp_features,tmp_jetfeatures,tmp_truthlabel,tmp_4vec = tmp_features[perm_indices],tmp_jetfeatures[perm_indices],tmp_truthlabel[perm_indices],tmp_4vec[perm_indices]
        if self.equal_qcd:
            n_qcd = len(self.data_truthlabel[0])*len(self.data_truthlabel)   
            tmp_features,tmp_jetfeatures,tmp_truthlabel,tmp_4vec = tmp_features[0:n_qcd],tmp_jetfeatures[0:n_qcd],tmp_truthlabel[0:n_qcd], tmp_4vec[0:n_qcd]
        
        
        if self.max_jets is not None:
            self.data_features.append(tmp_features.astype(np.float64)[0:9*self.max_jets*self.maxfiles])
            self.data_jetfeatures.append(tmp_jetfeatures.astype(np.float64)[0:9*self.max_jets*self.maxfiles])
            self.data_truthlabel.append(tmp_truthlabel.astype(np.float64)[0:9*self.max_jets*self.maxfiles])
            self.data_4vec.append(tmp_4vec.astype(np.float64)[0:9*self.max_jets*self.maxfiles])
        else:
            self.data_features.append(tmp_features.astype(np.float64))
            self.data_jetfeatures.append(tmp_jetfeatures.astype(np.float64))
            self.data_truthlabel.append(tmp_truthlabel.astype(np.float64))
            self.data_4vec.append(tmp_4vec.astype(np.float64))
            

        self.data_features = np.vstack(self.data_features)
        self.data_jetfeatures = np.vstack(self.data_jetfeatures)
        self.data_truthlabel = np.concatenate((self.data_truthlabel))
        self.data_4vec = np.concatenate(self.data_4vec)
        print("self.data_features.shape",self.data_features.shape)
     
        self.data_features = torch.DoubleTensor(self.data_features)
        self.data_jetfeatures = torch.DoubleTensor(self.data_jetfeatures)
        self.data_truthlabel = torch.DoubleTensor(self.data_truthlabel)
        self.data_4vec = torch.DoubleTensor(self.data_4vec)
        perm = np.random.permutation(len(self.data_features))
                                     
        self.data_features = self.data_features[perm]
        self.data_jetfeatures = self.data_jetfeatures[perm]
        self.data_truthlabel = self.data_truthlabel[perm]
        self.data_4vec = self.data_4vec[perm]
#         np.save('/n/home11/nswood/Mixed_Curvature/ex_out',self.data_truthlabel)
        
    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.h5')))
        return raw_files

    @property
    def processed_file_names(self):
        return []

    def __len__(self):
        return self.data_jetfeatures.shape[0]#self.strides[-1]

    def __getitem__(self, idx):
        x_pf = self.data_features[idx,:,:]
        x_jet = self.data_jetfeatures[idx,:]
        y = self.data_truthlabel[idx]
        v = self.data_4vec[idx,:,:]
        return x_pf, x_jet,y, v
        