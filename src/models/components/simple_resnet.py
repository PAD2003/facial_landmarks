import torch.nn as nn
from torchvision import models

class SimpleResnet(nn.Module):
    def __init__(self,
                 model_name: str ="resnet18",
                 weights: str ="DEFAULT",
                 output_shape: list =[68, 2]):
        super().__init__()

        # get the output_shape
        self.output_shape = output_shape

        # init a pretrained resnet
        backbone = models.get_model(name=model_name, weights=weights)

        # get feature extractor
        layers = list(backbone.children())[:-1] # not include output layers
        self.feature_extractor = nn.Sequential(*layers)

        # freeze self.feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # unfreeze some layer # problem: too much param to train
        for param in self.feature_extractor[-2][1].parameters():
            param.requires_grad = True

        # get in_features of resnet50's output layer (2048)
        num_filters = backbone.fc.in_features

        # create output layer to predict landmarks
        self.output_layer = nn.Linear(in_features=num_filters,
                                      out_features=output_shape[0]*output_shape[1])
        
    def forward(self, x):
        # ### DETAIL
        # # forward pass the feature_extractor
        # x = self.feature_extractor(x)

        # # (B, 3, 224, 224) -> (B, 3 * 224 *224)
        # batch_size, channels, width, height = x.size()
        # x = x.view(batch_size, -1)

        # # forward pass the output_layer
        # x = self.output_layer(x)

        # # (B, 68*2) -> (B, 68, 2)
        # batch_size, _ = x.size()
        # x.reshape(batch_size, self.output_shape[0], self.output_shape[1])

        # return x

        return self.output_layer(self.feature_extractor(x).view(x.size(0), -1)).reshape(x.size(0), self.output_shape[0], self.output_shape[1])

if __name__ == "__main__":
    from torchinfo import summary
    import torch

    # create a model
    simple_resnet = SimpleResnet()

    # show model
    batch_size = 16
    summary(model=simple_resnet,
            input_size=(batch_size, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    # test input & output shape
    random_input = torch.randn([16, 3, 224, 224])
    output = simple_resnet(random_input)
    print(f"\n\nINPUT SHAPE: {random_input.shape}")
    print(f"OUTPUT SHAPE: {output.shape}")