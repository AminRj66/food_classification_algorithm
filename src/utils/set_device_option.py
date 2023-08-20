import os


# set device (gpu ro cpu)
def set_device_option(device):
    if device.split(':')[0] == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[-1]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
