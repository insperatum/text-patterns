import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import util

class RobustFill(nn.Module):
    """
    
    """
    
    def __init__(self):
        super(RobustFill, self).__init__()
        self.h_encoder_size = 100
        self.h_decoder_size = 100
        self.embedding_size = 100
        self.max_length_input = 10
        self.max_length_output = 10
        self.v_input = 10 # Number of tokens in input vocabulary
        self.v_output = 10 # Number of tokens in output vocabulary
        self.encoder_cell = nn.GRUCell(input_size=self.v_input+1, hidden_size=self.h_encoder_size, bias=True)
        self.decoder_cell = nn.GRUCell(input_size=self.v_output+1, hidden_size=self.h_decoder_size, bias=True)
        self.W = nn.Linear(self.h_encoder_size + self.h_decoder_size, self.embedding_size)
        self.V = nn.Linear(self.embedding_size, self.v_output+1)
        self.A = nn.Bilinear(self.h_encoder_size, self.h_decoder_size, 1, bias=False) # attention: general https://arxiv.org/pdf/1508.04025.pdf
        # self.input_EOS = 
        self.output_SOS = Variable(torch.zeros(1, self.v_input+1))
        self.output_SOS[:, -1] = 1

    def score(self, inputs, target):
        """
        :param List[LongTensor] inputs: n_examples * (max_length_input * batch_size)
        :param List[LongTensor] target: max_length_output * batch_size
        """
        n_examples = len(inputs)
        batch_size = inputs[0].size(1)

        score = Variable(torch.zeros(batch_size))
        inputs_scatter = [Variable(torch.zeros(self.max_length_input, batch_size, self.v_input+1).scatter_(2, inputs[j][:, :, None], 1)) for j in range(n_examples)] # n_examples * (max_length_input * batch_size * v_input+1)
        target_scatter = Variable(torch.zeros(self.max_length_output, batch_size, self.v_output+1).scatter_(2, target[:, :, None], 1)) # max_length_output * batch_size * v_output+1
        
        H = [] # n_examples * (max_length_input * batch_size * h_encoder_size)
        for j in range(n_examples):
            h = Variable(torch.zeros(batch_size, self.h_encoder_size))
            hs = []
            for i in range(self.max_length_input):
                h = self.encoder_cell(inputs_scatter[j][i, :, :], h)
                hs.append(h[None, :, :])
            H.append(torch.cat(hs, 0))

        def attend(j, h_dec):
            """
            :param j: Index of example
            :param h_dec: batch_size * h_decoder_size
            """
            scores = self.A(
                H[j].view(self.max_length_input * batch_size, self.h_encoder_size),
                h_dec.view(batch_size, self.h_decoder_size).repeat(self.max_length_input, 1)
            ).view(self.max_length_input, batch_size)
            c = (F.log_softmax(scores[:, :, None], dim=0) * H[j]).sum(0)
            return c#torch.cat([h_dec, c], dim=1)

        # Multi-example pooling: Figure 3, https://arxiv.org/pdf/1703.07469.pdf
        P = [self.decoder_cell(self.output_SOS, H[j][-1]) for j in range(n_examples)]
        for i in range(self.max_length_output):
            FC = []
            for j in range(n_examples):
                p_aug = torch.cat([P[j], attend(j, P[j])], 1)
                FC.append(F.tanh(self.W(p_aug)[None, :, :]))
            m = torch.max(torch.cat(FC, 0), 0)[0] # batch_size * embedding_size
            output = F.log_softmax(self.V(m), dim=1)
            score = score + util.choose(output, target[i, :])
            for j in range(n_examples):
                P[j] = self.decoder_cell(target_scatter[i, :, :], attend(j, P[j]))

        return score
        #todo: variable length
        #todo: 'init' parameter rather than zeros


if __name__ == "__main__":
    net = RobustFill()
    params = net.parameters()
    opt = torch.optim.SGD(params, lr=0.001)
    for i in range(1000):
        opt.zero_grad()
        target = (10*torch.rand(2, 10).repeat(5,1)).long()
        inputs = [target for _ in range(5)]
        # inputs = [target.scatter_(1, (20*torch.rand(10,1)).long(), 0) for _ in range(5)]
        score = net.score(inputs, target).sum()
        print(score.data[0])
        (-score).backward()
        opt.step()