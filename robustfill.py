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
import random


class RobustFill(nn.Module):
    """
    Todo:
    - check if this is right? attend during P->FC rather than during softmax->P?
    - allow length 0 inputs/outputs
    - give n_examples as input to FC
    """
    
    def __init__(self, vocab_size=128, hidden_size=512, embedding_size=128, cell_type="LSTM"):
        super(RobustFill, self).__init__()
        self.h_encoder_size = hidden_size
        self.h_decoder_size = hidden_size
        self.embedding_size = embedding_size
        self.v_input = vocab_size # Number of tokens in input vocabulary
        self.v_output = vocab_size # Number of tokens in output vocabulary

        self.cell_type=cell_type
        if cell_type=='GRU':
            self.encoder_cell = nn.GRUCell(input_size=self.v_input+1, hidden_size=self.h_encoder_size, bias=True)
            self.encoder_init = Parameter(torch.rand(1, self.h_encoder_size))
            self.decoder_cell = nn.GRUCell(input_size=self.v_output+1, hidden_size=self.h_decoder_size, bias=True)
        if cell_type=='LSTM':
            self.encoder_cell = nn.LSTMCell(input_size=self.v_input+1, hidden_size=self.h_encoder_size, bias=True)
            self.encoder_init = (Parameter(torch.rand(1, self.h_encoder_size)), Parameter(torch.rand(1, self.h_encoder_size)))
            self.decoder_cell = nn.LSTMCell(input_size=self.v_output+1, hidden_size=self.h_decoder_size, bias=True)
            self.decoder_init_c = Parameter(torch.rand(1, self.h_decoder_size))
            
        self.W = nn.Linear(self.h_encoder_size + self.h_decoder_size, self.embedding_size)
        self.V = nn.Linear(self.embedding_size, self.v_output+1)
        self.A = nn.Bilinear(self.h_encoder_size, self.h_decoder_size, 1, bias=False) 
        self.input_EOS = torch.zeros(1, self.v_input+1)
        self.input_EOS[:, -1] = 1
        self.input_EOS = Parameter(self.input_EOS)
        self.output_SOS = torch.zeros(1, self.v_output+1)
        self.output_SOS[:, -1] = 1
        self.output_SOS = Parameter(self.output_SOS)


    def encoder_get_init(self, batch_size):
        if self.cell_type=="GRU": return self.encoder_init.repeat(batch_size, 1)
        if self.cell_type=="LSTM": return tuple(x.repeat(batch_size, 1) for x in self.encoder_init)

    def decoder_get_init(self, encoder_h):
        if self.cell_type=="GRU": return encoder_h
        if self.cell_type=="LSTM": return (encoder_h, self.decoder_init_c.repeat(encoder_h.size(0), 1))

    def cell_get_h(self, cell_state):
        if self.cell_type=="GRU": return cell_state
        if self.cell_type=="LSTM": return cell_state[0]

    def score(self, inputs, target):
        output, score = self.run(inputs, target=target, mode="score")
        return score

    def sample(self, inputs):
        output, score = self.run(inputs, mode="sample")
        return output

    def run(self, inputs, target=None, mode="sample"):
        """
        :param mode: "score" returns log p(target|input), "sample" returns output ~ p(-|input) 
        :param List[LongTensor] inputs: n_examples * (max_length_input * batch_size)
        :param List[LongTensor] target: max_length_output * batch_size
        """
        assert((mode=="score" and target is not None) or mode=="sample")

        n_examples = len(inputs)
        max_length_input = [inputs[j].size(0) for j in range(n_examples)]
        batch_size = inputs[0].size(1)
        max_length_output = target.size(0) if target is not None else 20

        score = Variable(torch.zeros(batch_size))
        inputs_scatter = [Variable(torch.zeros(max_length_input[j], batch_size, self.v_input+1).scatter_(2, inputs[j][:, :, None], 1)) for j in range(n_examples)] # n_examples * (max_length_input * batch_size * v_input+1)
        if target is not None: target_scatter = Variable(torch.zeros(max_length_output, batch_size, self.v_output+1).scatter_(2, target[:, :, None], 1)) # max_length_output * batch_size * v_output+1
        
        H = [] # n_examples * (max_length_input * batch_size * h_encoder_size)
        embeddings = [] # h for example at INPUT_EOS
        attention_mask = [] # 0 until (and including) INPUT_EOS, then -inf
        for j in range(n_examples):
            active = torch.Tensor(max_length_input[j], batch_size).byte()
            active[0, :] = 1
            encoder_state = self.encoder_get_init(batch_size)
            hs = []
            for i in range(max_length_input[j]):
                encoder_state = self.encoder_cell(inputs_scatter[j][i, :, :], encoder_state)
                if i+1 < max_length_input[j]: active[i+1, :] = active[i, :] * (inputs[j][i, :] != self.v_input)
                h = self.cell_get_h(encoder_state) 
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
                H[j].view(max_length_input[j] * batch_size, self.h_encoder_size),
                h_dec.view(batch_size, self.h_decoder_size).repeat(max_length_input[j], 1)
            ).view(max_length_input[j], batch_size) + attention_mask[j]
            c = (F.softmax(scores[:, :, None], dim=0) * H[j]).sum(0)
            return c

        # Multi-example pooling: Figure 3, https://arxiv.org/pdf/1703.07469.pdf
        output = target if mode=="score" else torch.zeros(max_length_output, batch_size).long()
        decoder_states = [self.decoder_cell(self.output_SOS.repeat(batch_size,1), self.decoder_get_init(embeddings[j])) for j in range(n_examples)] #P
        active = torch.ones(batch_size).byte()
        for i in range(max_length_output):
            FC = []
            for j in range(n_examples):
                h = self.cell_get_h(decoder_states[j])
                p_aug = torch.cat([h, attend(j, h)], 1)
                FC.append(F.tanh(self.W(p_aug)[None, :, :]))
            m = torch.max(torch.cat(FC, 0), 0)[0] # batch_size * embedding_size
            logsoftmax = F.log_softmax(self.V(m), dim=1)
            if mode=="sample": output[i, :] = torch.multinomial(logsoftmax.data.exp(), 1)[:, 0]
            score = score + util.choose(logsoftmax, output[i, :]) * Variable(active.float())
            active *= (output[i, :] != self.v_output)
            for j in range(n_examples):
                if mode=="score":
                    output_scatter = target_scatter[i, :, :]
                elif mode=="sample":
                    output_scatter = Variable(torch.zeros(batch_size, self.v_output+1).scatter_(1, output[i, :, None], 1))
                decoder_states[j] = self.decoder_cell(output_scatter, decoder_states[j]) 
        return output, score


def tensorToStrings(vocab_size, tensor):
    """
    :param tensor: max_length * batch_size
    """
    return [''.join(chr(x) for x in tensor[:, i]).partition(chr(vocab_size))[0] for i in range(tensor.size(1))]

def stringsToTensor(vocab_size, strings):
    """
    :param vocab_size: vocabulary size
    :param strings: batch of strings
    """
    maxlen = max(len(s) for s in strings)
    t = torch.ones(1 if maxlen==0 else maxlen, len(strings)).long()*vocab_size
    for i in range(len(strings)):
        s = strings[i]
        if len(s)>0: t[:len(s), i] = torch.LongTensor([ord(x) for x in s])
    return t






# ------------------------------------------------------------------------

if __name__ == "__main__":
    b = 500
    max_examples=5
    v=128
    l=10

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
        n_examples = random.randint(1, max_examples)
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