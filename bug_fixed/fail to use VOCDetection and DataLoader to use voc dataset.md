## 🐛 Bug

<!-- A clear and concise description of what the bug is. -->

## To Reproduce

Steps to reproduce the behavior:

1. when use ```VOCDetection``` to load voc dataset and  use ```DataLoader``` to extract the data,an issue occurd when running
```
TypeError: list indices must be integers or slices, not str
```
when debugging to the code,it turns out that the function```default_collate(batch)``` is unabel to deal with the data type  returned by ``` VOCDetection.__getitem__```

2. I write a function to deal with this problem 

```python
def voc_collate_fn(batch_list):
    print(batch_list)
    images = default_collate([batch_list[i][0] for i in range(len(batch_list))])
    annotations = {}
    for k in batch_list[0][1]['annotation']:
        annotations[k] = [batch_list[i][1]['annotation'][k] for i in range(len(batch_list))]
    object_list = []
    for i in annotations['object']:
        if type(i)==list:
            object_list.append(i)
        else:
            l = []
            l.append(i)
            object_list.append(l)
    annotations['object'] = object_list
    return {'images':images,'annotations':annotations}
```

<!-- If you have a code sample, error messages, stack traces, please provide it here as well -->

## Expected behavior

<!-- A clear and concise description of what you expected to happen. -->

I hope the ```default_collate(batch)``` function in file  ```torch\utils\data\_utils\collate.py``` can deal with the voc dataset

## Environment

Please copy and paste the output from our
[environment collection script](https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py)
(or fill out the checklist below manually).

You can get the script and run it with:
```
wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py
# For security purposes, please check the contents of collect_env.py before running it.
python collect_env.py
```

 - PyTorch Version (e.g., 1.0): 1.1.0
 - OS (e.g., Linux):Microsoft Windows 10 
 - How you installed PyTorch (`conda`, `pip`, source):conda
 - Build command you used (if compiling from source):
 - Python version:3.7
 - CUDA runtime version: 8.0.60
 - cuDNN version: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\cudnn64_6.dll
 - GPU models and configuration: GeForce GTX 1050
 - Any other relevant information:

## Additional context

<!-- Add any other context about the problem here. -->