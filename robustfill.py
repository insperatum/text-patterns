import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
v = 10
l = 10
b = 500
n_examples=3

class RobustFill(nn.Module):
    """
    
    """
    
    def __init__(self):
        super(RobustFill, self).__init__()
        self.h_encoder_size = 100
        self.h_decoder_size = 100
        self.embedding_size = 100
        self.max_length_input = l
        self.max_length_output = l
        self.v_input = v # Number of tokens in input vocabulary
        self.v_output = v # Number of tokens in output vocabulary
        self.encoder_cell = nn.GRUCell(input_size=self.v_input+1, hidden_size=self.h_encoder_size, bias=True)
        self.encoder_init = Parameter(torch.rand(1, self.h_encoder_size))
        self.decoder_cell = nn.GRUCell(input_size=self.v_output+1, hidden_size=self.h_decoder_size, bias=True)
        self.W = nn.Linear(self.h_encoder_size + self.h_decoder_size, self.embedding_size)
        self.V = nn.Linear(self.embedding_size, self.v_output+1)
        self.A = nn.Bilinear(self.h_encoder_size, self.h_decoder_size, 1, bias=False) 
        self.input_EOS = torch.zeros(1, self.v_input+1)
        self.input_EOS[:, -1] = 1
        self.input_EOS = Parameter(self.input_EOS)
        self.output_SOS = torch.zeros(1, self.v_output+1)
        self.output_SOS[:, -1] = 1
        self.output_SOS = Parameter(self.output_SOS)


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
        embeddings = [] # h for example at INPUT_EOS
        attention_mask = [] # 0 until (and including) INPUT_EOS, then -inf
        for j in range(n_examples):
            active = torch.Tensor(self.max_length_input, batch_size).byte()
            active[0, :] = 1
            h = self.encoder_init.repeat(batch_size, 1)
            hs = []
            for i in range(self.max_length_input):
                h = self.encoder_cell(inputs_scatter[j][i, :, :], h)
                if i+1 < self.max_length_input: active[i+1, :] = active[i, :] * (inputs[j][i, :] != self.v_input)
                hs.append(h[None, :, :])
            H.append(torch.cat(hs, 0))
            embedding_idx = active.sum(0).long() - 1
            embedding = H[j].gather(0, Variable(embedding_idx[None, :, None].repeat(1, 1, self.h_encoder_size)))[0]
            embeddings.append(embedding)
            attention_mask.append(Variable(active.float().log()))

        def attend(j, h_dec):
            """
            'general' attention from https://arxiv.org/pdf/1508.04025.pdf
            :param j: Index of example
            :param h_dec: batch_size * h_decoder_size
            """
            scores = self.A(
                H[j].view(self.max_length_input * batch_size, self.h_encoder_size),
                h_dec.view(batch_size, self.h_decoder_size).repeat(self.max_length_input, 1)
            ).view(self.max_length_input, batch_size) + attention_mask[j]
            c = (F.softmax(scores[:, :, None], dim=0) * H[j]).sum(0)
            return c

        # Multi-example pooling: Figure 3, https://arxiv.org/pdf/1703.07469.pdf
        P = [self.decoder_cell(self.output_SOS.repeat(batch_size,1), embeddings[j]) for j in range(n_examples)]
        active = torch.ones(batch_size).byte()
        for i in range(self.max_length_output):
            FC = []
            for j in range(n_examples):
                p_aug = torch.cat([P[j], attend(j, P[j])], 1)
                FC.append(F.tanh(self.W(p_aug)[None, :, :]))
            m = torch.max(torch.cat(FC, 0), 0)[0] # batch_size * embedding_size
            output = F.log_softmax(self.V(m), dim=1)
            score = score + util.choose(output, target[i, :]) * Variable(active.float())
            active *= (target[i, :] != self.v_output)
            for j in range(n_examples):
                P[j] = self.decoder_cell(target_scatter[i, :, :], attend(j, P[j]))
        return score

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net = RobustFill()
    params = net.parameters()
    opt = torch.optim.Adam(params, lr=0.01)
    def genString():
        t = (torch.rand(l, b)*v).long()
        lengths = (torch.rand(1, b)*l).long()
        t.scatter_(0, lengths, v)
        return t

    xs = []
    ys = []
    n_iters = 300
    for i in range(n_iters):
        opt.zero_grad()
        target = genString()
        corrupt_idx = torch.zeros(n_examples, 1, b).scatter_(0, (torch.rand(1, 1, b)*n_examples).long(), 1).long()
        inputs = [target*(1-corrupt_idx[j, :, :]) + genString()*corrupt_idx[j, :, :] for j in range(n_examples)] #For each character, corrupt one example
        score = net.score(inputs, target).mean()
        avgscore = score.data[0] if i==0 else avgscore*0.9 + 0.1*score.data[0]
        print(score.data[0])
        (-score).backward()
        opt.step()

        xs.append(i)
        ys.append(-avgscore)
        plt.clf()
        plt.plot(xs, ys)
        plt.xlim(xmin=0, xmax=n_iters)
        plt.ylim(ymin=0)
        plt.xlabel('iteration')
        plt.ylabel('NLL')
        plt.savefig("results/robustfill.png")