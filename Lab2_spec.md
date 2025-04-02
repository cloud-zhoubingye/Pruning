# Pruning
Prune mobilenetv2 by magnitude-based and taylor-based methods
This is Lab 1: Prune Vision Model of Edge AI course, NYCU, 2025 spring.

<span style="color:Red;">**Due Date: 3/31 23:59**</span>

:::warning
**Individual Assignment**
:::
:::danger
**Reminder:** Part 2 may require a decent amount of time on experimenting and retraining model. Start early!
:::

## Introduction

This lab aims to prune the MobileNetV2 model, resulting in a significant speedup while maintaining satisfactory accuracy.

* Please download the provided Jupyter Notebook file using the link below.
Follow the prompts and hints provided within the notebook to fill in the empty blocks and answer the questions.

    > [lab1.ipynb](https://drive.google.com/file/d/1_3cyChpEVhSb_pbOhLpUDDzL42T_OaDi/view?usp=sharing)

* Below is a MobileNetV2 with 96.3% accuarcy on CIFAR10, finetuned by TAs. Feel free to use it as a starting point for pruning. It's also fine if you want to use your own model:

    > [mobilenetv2_0.963.pth](https://drive.google.com/file/d/1wJuibk5ogSWQYePgLRtQiRFdokDXJfnZ/view?usp=sharing)

    You can load the above model with the following snippet:
    ```python
    import torch
    from torchvision.models import mobilenet_v2

    model = torch.load('./mobilenetv2_0.963.pth', map_location="cpu", weights_only=False)
    ```

## Part 1: Finegrained-Pruning (Unstructered Pruning) (50%)

In this part, you will learn the pruning basics by implementing various methods to sparsify the model. 
![image](https://hackmd.io/_uploads/SkaddPFA6.png)


Refer to **"Part1"** in the provided notebook (ipynb).

## Part 2: Channel Pruning using Torch-Pruning  (50% +)

You will apply modern channel pruning techniques on MobileNetV2 using the [Torch-Pruning library](https://github.com/VainF/Torch-Pruning). This library captures the dependencies between adjacent model layers, which allows us to perform consistent structural pruning easily.

![torch_pruning](https://hackmd.io/_uploads/SyBCDvtCa.png)

Refer to **"Part2"** in the provided notebook (ipynb).

## Hand-In Policy

You will need to hand-in:
* Your pruned model (***.pth***)
* Fill out ***lab1.ipynb***, and rename it to ***```<YourID>```.ipynb***

Please organize your submission files into a zip archive structured as follows:

```scss
YourID.zip
    ├── model/
    │     └── mobilenet.pth (sparsified model from part 2)
    └── YourID.ipynb
```

## Evaluation Criteria

Upon receiving your zip file, we will:
1. Verify the structure of your pruned model by inspecting ```mobilenet.pth``` using [Netron](https://netron.app/). Your pruned model should originate from MobileNetV2 or you will get no points.
2. Obtain the number of MFLOPs and accuracy of your pruned model using the provided code snippet:
    ```python
    import torch
    import torch_pruning as tp
    from torchvision.models import mobilenet_v2

    model = torch.load('./model/mobilenet.pth', map_location="cpu", weight_only=False)
    example_inputs = torch.zeros(1, 3, 224, 224).to(args.device)
    ops, size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)

    accuracy = evaluate(model, dataloader['test']) / 100  # Range from 0 to 1
    MFLOPs = ops/1e6
    print(MFLOPs)
    print(accuracy)
    ```

Accuracy and MFLOPs obtained above will be used in the grading formula for **Part 2**:

$$
  Score = (\dfrac{200 - MFLOPs}{200 - 45} - 10\times Clamp(0.92-Accuracy,\ 0,\ 1)) \times 45\%
$$

- It is possible to obtain higher than 45%.

:::info
:warning: Note: Accuracy beyond 0.92 (92%) won't contribute to a higher score!
:::

:::danger
**Reminder:** Go through the evaluation steps yourself before submission.
:::

## Tips and Recommendations

* Retraining after pruning is essential to recover model's accuracy.
* A moderate number of epochs (around 10) should suffice for finetuning/retraining.
* Experiment with multiple pruning and retraining iterations.
* Experiment with learning rate of different order of magnitude.

