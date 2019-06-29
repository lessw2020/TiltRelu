# TiltRelu
Pytorch impl of TiltRelu
TiltRelu is the "static activation normalization" of Relu.
https://arxiv.org/abs/1905.01369v1

Code for now until I checkin:

    class TiltRelu(nn.Module):
        def __init__(self, sub=, mean_shift=-.03):
            super().__init__()
            self.sub = sub
            self.mean_shift = mean_shift
            self.constant = .79788456

        def forward(self,x):
            x = torch.abs(x)

            if self.sub:
                x.sub_(self.constant)

                #pi = 3.14159265359
                #sqrt (2 / pi) = .79788456080283909979633587119266‬

            if self.mean_shift is not None:
                x.sub_(self.mean_shift)

            return x

