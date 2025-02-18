import torch
import torch.nn as nn
import torchvision
from torchvision import models

# def get_last_layers(model):
#     """
#     :param model: torch.nn.Module
#     :return: list, last layers' names
#     """
#     # Get all layers using named_modules
#     layer_names = [name for name, layer in model.named_modules()]
    
#     # Return the last 4 layers (example)
#     last_layers = layer_names[-6:-2]
#     return last_layers


def process_layers(model):
    """
    Process each layer of the model
    :param model: torch.nn.Module
    """
    # # Get the last layers from the model
    # last_layers = get_last_layers(model)

    # Print all layers and their names
    # print("Last layers' names:", last_layers)
    dummy_input = torch.randn(1, 3, 224, 224)
    # Iterate over each layer in the model
    for name, layer in model.named_modules():
        # hook
        print(name, layer)
        if name == 'layer4.1.conv1':
            layer.register_forward_hook(hook_fn)
    
    print(model(dummy_input))
        
def hook_fn(module, input, output):
    print(f"Layer: {module}, Input shape: {output.shape}")


# Example model (ResNet18 or any other model)
model = models.resnet18(pretrained=True)  # Updated to use 'weights' instead of 'pretrained'

# # Register hooks to the last layers
# for layer in get_last_layers(model):
#     getattr(model, layer).register_forward_hook(hook_fn)
# Process and print last layers' output
process_layers(model)
