import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

# Define your custom VGG architecture
class VGGCustom(nn.Module):
    def __init__(self, num_classes=101):
        super(VGGCustom, self).__init__()
        # Load the pretrained VGG16_bn model with updated weights parameter
        vgg16_bn = models.vgg16_bn(pretrained=True)
        
        # Extract the features and avgpool layers
        self.features = vgg16_bn.features
        self.avgpool = vgg16_bn.avgpool
        
        # Extract the classifier and replace only the last FC layer
        self.classifier = vgg16_bn.classifier
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x, is_feat=False, is_feats=False):
        if is_feats:
            # Extract intermediate features
            f0 = self.features[:6](x)
            f1 = self.features[6:13](f0)
            f2 = self.features[13:23](f1)
            f3 = self.features[23:33](f2)
            f4 = self.features[33:43](f3)
            f5 = self.features[43:](f4)
            x = self.features[43:](x)
            x = x.view(x.size(0), -1)
            f5 = x
            out = self.classifier(x)
            return out, [f0, f1, f2, f3, f4, f5]
        else:
            out = self.features(x)
            feat = out.view(out.size(0), -1)
            out = self.classifier(feat)
            if is_feat:
                return out, feat
            else:
                return out

def get_modified_vgg16(num_classes=101):
    return VGGCustom(num_classes=num_classes)

def print_model_parameters(model):
    total_params = 0
    #print("\nModel Parameters:")
    for name, param in model.named_parameters():
        param_count = param.numel()
        #print(f"{name}: {param_count} parameters")
        total_params += param_count
    print(f"\nTotal parameters: {total_params}\n")

# Example usage for debugging:
def test():
    num_classes = 101
    model = get_modified_vgg16(num_classes)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    
    #summary(model, (3, 224, 224))
    
    # Freeze all parameters after building the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the last classifier layer
    for param in model.classifier[-4:].parameters():
        param.requires_grad = True
        
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", total_params)
    
    num_frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Number of frozen parameters:", num_frozen_params)
    
    num_frozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of unfrozen parameters:", num_frozen_params)
    
    # Display model architecture
    #print(model)
    
    # Dummy input for testing
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass to check for errors
    try:
        output = model(x)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        
test()