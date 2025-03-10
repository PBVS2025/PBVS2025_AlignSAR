from torchsampler import ImbalancedDatasetSampler
from eo_sar_data import SarEODataset, PairedTransform
import torch
import pandas as pd

transform = PairedTransform(resize_size=224, flip_prob=0.5, vertical_flip_prob=0.5)

dataset_train = SarEODataset(root_dir='/workspace/hjkim/PBVS_dataset/Unicorn_Dataset', transform=transform)

print(type(dataset_train))
print(isinstance(dataset_train, torch.utils.data.Dataset))

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    sampler=ImbalancedDatasetSampler(dataset_train),
    batch_size=64,
    drop_last=True,
)

print(len(train_loader))

sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=1, rank=0, shuffle=True
        )
data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=64,
        num_workers=10,
        pin_memory=1,
        drop_last=True,
    )

print(len(data_loader_train))

print(ImbalancedDatasetSampler(dataset_train).weights)
print(ImbalancedDatasetSampler(dataset_train).weights.shape)
print(ImbalancedDatasetSampler(dataset_train).num_samples)

df = pd.DataFrame()
labels = dataset_train.get_labels()
df["label"] = labels
df = df.sort_index()

# print(df["label"])

label_to_count = df["label"].value_counts()
weights = 1.0 / label_to_count[df["label"]]
weights = torch.DoubleTensor(weights.to_list())

# print(labels)
# print(type(labels))
