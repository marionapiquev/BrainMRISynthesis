import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import re


class PatientSliceDataset(Dataset):
    def __init__(self, data_dir, seq_len=5,  slice_shape=(256, 256)):
        """
        Args:
            data_dir (str): Path to the directory containing all `.npz` slice files.
            transform (callable, optional): Optional transform to be applied on a sample.
            slice_shape (tuple): Shape of the slices (height, width), used for padding.
            prev_cond_slices (int): number of previous slices used for conditioning.
            post_cond_slices (int): number of post slices used for conditioning.
        """
        self.data_dir = data_dir
        self.slice_shape = slice_shape  
        self.seq_len = seq_len
        self.slices_by_patient = self._group_slices_by_patient()
        self.sample_indices = self._get_sample_indices()

    def _group_slices_by_patient(self):
        """
        Groups all slices by patient ID and sorts them by slice number.
        """
        slices_by_patient = {}

        # Updated regex pattern to capture patient ID and slice number
        pattern = r'^(.*)_struc_brain_(\d+)\.npz$'

        for file_name in os.listdir(self.data_dir):
            match = re.match(pattern, file_name)
            if match:
                patient_id, slice_num = match.groups()
                slice_num = int(slice_num)  # Convert slice number to an integer for sorting

                if patient_id not in slices_by_patient:
                    slices_by_patient[patient_id] = []

                slices_by_patient[patient_id].append((slice_num, file_name))

        # Sort slices by slice number for each patient
        for patient_id in slices_by_patient:
            slices_by_patient[patient_id].sort(key=lambda x: x[0])

        return slices_by_patient

    def _get_sample_indices(self):
        """
        Generates a list of tuples where each tuple represents a valid sample:
        (patient_id, starting slice index) -> for the first valid input-target slice combination.
        """
        sample_indices = []

        for patient_id, slices in self.slices_by_patient.items():
            # Every slice is now a valid target (since we'll pad the first two slices)
            for i in range(len(slices)):
                sample_indices.append((patient_id, i))
        return sample_indices

    def _create_black_slice(self):
        """
        Creates a black (empty) slice as a NumPy array with the specified shape.
        """
        return np.full(self.slice_shape, -1, dtype=np.float32)

    def __len__(self):
        return len(self.sample_indices)

    def _get_slice_indices_list(self, slice_idx):
        start = slice_idx - self.seq_len - 1
        end = slice_idx
        return list(range(start, end + 1))

    def __getitem__(self, idx):
        patient_id, slice_idx = self.sample_indices[idx]
        slices = self.slices_by_patient[patient_id]

        slice_indices = self._get_slice_indices_list(slice_idx)
        loaded_slices = []

        for idx in slice_indices:
            if idx < 0 or idx >= len(slices):  # Out of bounds, use black slice for padding
                slice_data = self._create_black_slice()
                loaded_slices.append(slice_data)
            else:  # Load actual slice data
                slice_path = os.path.join(self.data_dir, slices[idx][1])
                slice_data = np.load(slice_path)['data']
                loaded_slices.append(slice_data)

        # Convert to PyTorch tensors
        sequence_tensor = torch.tensor(np.stack(loaded_slices, axis=0)).float().unsqueeze(
            3)  # Shape: (seq_len, H, W, 1)


        # Return condition slices (2 previous slices) and current slice
        return {
            "block": sequence_tensor,
            "positions": torch.tensor(slice_indices)
        }



