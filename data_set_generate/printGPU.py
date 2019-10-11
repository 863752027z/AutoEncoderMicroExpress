import torch


#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
print(torch.cuda.is_available())
print('hello pytorch:' + torch.__version__)
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(0))

