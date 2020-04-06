import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, input, outputs):
        super(Classifier, self).__init__()
        size = list(input.size())[-1]
        self.seq_nlp = nn.Sequential(
            nn.Linear(size, 2048),
            nn.ReLU(),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, outputs.shape[-1])
        ).cuda()

    def conv2d_size_out(self, size, kernel_size=3, stride=2):
        toreturn = size - (kernel_size - 1)
        toreturn = toreturn - 1
        toreturn = toreturn // stride
        toreturn += 1
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        nlp_output = self.seq_nlp(x)
        return nlp_output

    def _get_size(self, w, h):
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w)))
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h)))
        return convw * convh * 32