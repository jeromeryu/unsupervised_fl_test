import torch.nn as nn
import torch.nn.functional as F
import torch

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # ## encoder layers ##
        # # conv layer (depth from 3 --> 16), 3x3 kernels
        # self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        # # conv layer (depth from 16 --> 4), 3x3 kernels
        # self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # # pooling layer to reduce x-y dims by two; kernel and stride of 2
        # self.pool = nn.MaxPool2d(2, 2)
        
        # ## decoder layers ##
        # ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        # self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        # self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # ## encode ##
        # # add hidden layers with relu activation function
        # # and maxpooling after
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # # add second hidden layer
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)  # compressed representation
        
        # ## decode ##
        # # add transpose conv layers, with relu activation function
        # x = F.relu(self.t_conv1(x))
        # # output layer (with sigmoid for scaling from 0 to 1)
        # x = F.sigmoid(self.t_conv2(x))

        x = self.encoder(x)
        x = self.decoder(x)        
        
        return x


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive



# import torch.nn as nn

# '''
# modified to fit dataset size
# '''
# NUM_CLASSES = 10


# class AlexNet(nn.Module):
#     def __init__(self, num_classes=NUM_CLASSES):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 2 * 2, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 2 * 2)
#         x = self.classifier(x)
#         return x
