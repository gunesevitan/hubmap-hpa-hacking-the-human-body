import torch.nn as nn
import segmentation_models_pytorch as smp


class WeightedSegmentationLoss(nn.Module):

    def __init__(self,
                 distribution_based_loss_function, distribution_based_loss_function_args, distribution_based_loss_function_weight,
                 region_based_loss_function, region_based_loss_function_args, region_based_loss_function_weight):

        super(WeightedSegmentationLoss, self).__init__()

        self.distribution_based_loss_function = getattr(smp.losses, distribution_based_loss_function)(**distribution_based_loss_function_args)
        self.distribution_based_loss_function_weight = distribution_based_loss_function_weight
        self.region_based_loss_function = getattr(smp.losses, region_based_loss_function)(**region_based_loss_function_args)
        self.region_based_loss_function_weight = region_based_loss_function_weight

    def forward(self, inputs, targets):

        loss = self.distribution_based_loss_function(inputs, targets) * self.distribution_based_loss_function_weight + self.region_based_loss_function(inputs, targets) * self.region_based_loss_function_weight

        return loss
