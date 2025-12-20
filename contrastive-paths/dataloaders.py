from imports import *


# Option 1: For contrastive Learning

class PathPairDataset(Dataset):
    def __init__(self, all_samples_dicts):
        self.all_samples_dicts = all_samples_dicts
        self.num_samples = 0
        for phrase_dict in self.all_samples_dicts:
            self.num_samples += len(phrase_dict["good_paths"]) + len(phrase_dict["bad_paths"])

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        phrase_idx = np.random.randint(len(self.all_samples_dicts))
        good_paths = self.all_samples_dicts[phrase_idx]["good_paths"]
        bad_paths = self.all_samples_dicts[phrase_idx]["bad_paths"]
        return {
            "good_path": good_paths[idx % len(good_paths)],
            "bad_path": bad_paths[idx % len(bad_paths)]
        }
        
        
def pad_and_stack(paths):
    lengths = torch.tensor([len(p) for p in paths])
    max_len = lengths.max() + 1  # to include an "end token" for each phrase
    padded = torch.stack([
        F.pad(torch.Tensor(p), (0, 0, 0, max_len - len(p)))  # pad rows (dim=0)
        for p in paths
    ])
    return padded, lengths

def collate_fn(batch):
    """ Used by the dataloader
        to colalte a batch of good and bad paths
    """
    # 1. get good and bad paths
    
    good_paths = [torch.tensor(item['good_path'], dtype=torch.float32) for item in batch]
    bad_paths = [torch.tensor(item['bad_path'], dtype=torch.float32) for item in batch]
    
    # 2. pad and stack the paths
    
    good_feats, good_lens = pad_and_stack(good_paths)
    bad_feats, bad_lens = pad_and_stack(bad_paths)
    
    # 3. return a dictionary with good and bad features and lengths
    
    return {
        'good_feats': good_feats,   # [batch_size, max_path_len_good, 3]
        'good_len': good_lens,      # [batch_size]
        'bad_feats': bad_feats,     # [batch_size, max_path_len_bad, 3]
        'bad_len': bad_lens         # [batch_size]
    }


        
        
        
# # Option 2: For classification

# class PathExamplesDataset(Dataset):
#     def __init__(self, all_samples_dict):
#         self.all_samples_dicts = all_samples_dicts
#         self.num_samples = 0
#         for phrase_dict in self.all_samples_dicts:
#             self.num_samples += len(phrase_dict["good_paths"]) + len(phrase_dict["bad_paths"])
#         self.p_positive_sampling = 0.5  # Probability of sampling a positive example

#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, idx):
#         # Randomly sample a good or bad path
#         if np.random.rand() < self.p_positive_sampling:
#             path_type = "good_paths"
#         else:
#             path_type = "bad_paths"
#         phrase_idx = np.random.randint(len(self.all_samples_dicts))
#         paths = self.all_samples_dicts[phrase_idx][path_type]
#         path = paths[idx % len(paths)]
#         path_tensor = torch.tensor(path, dtype=torch.float32)
#         label = bool(path_type == "good_paths")
#         return {
#             "path": path_tensor,
#             "label": label
#         } 