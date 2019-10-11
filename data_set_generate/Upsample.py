import torch


m = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

input_pose_f = torch.arange(0, 256).view(1, 1, 1, 256).float()
target_pose_f = torch.arange(0, 256).view(1, 1, 1, 256).float()
pose_vector = torch.cat((input_pose_f, target_pose_f), 1)
print(pose_vector.shape)
x = m(pose_vector)
print(x.shape)
x = m(x)
print(x.shape)