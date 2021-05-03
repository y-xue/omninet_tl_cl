import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .resnet import resnet152

class VideoInputPeripheral(nn.Module):
    def __init__(self,output_dim,dropout=0,weights_preload=True,freeze_layers=True):

        self.feature_dim=1024  
        super(VideoInputPeripheral,self).__init__()
        self.image_model=resnet152(pretrained=weights_preload, small=True)
        if freeze_layers:
            self.image_model=self.image_model.eval()
            #Override the train mode So that it does not change when switching OmniNet to train mode
            self.image_model.train=self.empty_fun
            self.image_model.eval=self.empty_fun
            for param in self.image_model.parameters():
                param.requires_grad = False
        self.enc_dropout=nn.Dropout(dropout)   
        self.output_fc=nn.Linear(self.feature_dim,output_dim)

    def encode(self,image_tensor):
        # print('image peripheral:')
        # print('image_tensor[0] and [-1]:')
        # print(image_tensor[0])
        # print(image_tensor[-1])
        shape=image_tensor.shape
        if len(shape)==5:
            t_dim=image_tensor.shape[1]
            image_tensor=torch.reshape(image_tensor,(-1,3,shape[3],shape[4])) 
            # print('r1:', image_tensor.shape)   # [b*t, c, h, w]
        batch_size=image_tensor.shape[0]
        image_enc=self.image_model(image_tensor)
        # print('r2:', image_enc.shape) # [b*t, f, h, w]
        h,w = image_enc.shape[2], image_enc.shape[3]
        # print('image_enc:')
        # print(image_enc[0])
        # print(image_enc[-1])
        # enc_reshape=torch.reshape(image_enc,[batch_size,self.feature_dim,-1])
        # print(enc_reshape.shape) # [b*t, f, h*w]
        # enc_transposed=torch.transpose(enc_reshape,1,2)
        enc_transposed=image_enc.permute(0,2,3,1) 
        # print(enc_transposed.shape) # [b*t, h, w, f]
        output_enc=self.enc_dropout(enc_transposed)
        # output_enc=self.output_fc(output_enc)
        # print(output_enc.shape) # [b*t, h, w, out_dim]
        # print('output_enc:')
        # print(output_enc[0])
        # print(output_enc[-1])
        if len(shape)==5:
            output_enc=torch.reshape(output_enc,(-1,t_dim,output_enc.shape[1],output_enc.shape[2],output_enc.shape[3]))
            # print('r3:', output_enc.shape) # [b, t, h, w, out_dim]
        else:
            output_enc=output_enc.unsqueeze(1)
        return output_enc

    def empty_fun(self,mode):
        pass


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        # ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c.detach() * self.Wci)
        # see https://github.com/automan000/Convolutional_LSTM_PyTorch/issues/20

        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], pool_window_size=None):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self.pool_window_size = pool_window_size
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, inputs):
        """
        inputs: [sequence, bsize, channel, x, y]
        """
        # print(inputs.shape)
        # print(self.step)
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = inputs[step] # changed from x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

class ConvLSTMCLF(nn.Module):
    def __init__(self, num_classes, 
                input_channels, hidden_channels, kernel_size, 
                step=1, effective_step=[1], 
                hidden_sizes=[16384,128], pool_window_size=2):
        super(ConvLSTMCLF, self).__init__()
        self.convlstm = ConvLSTM(input_channels=input_channels, hidden_channels=hidden_channels, kernel_size=kernel_size, step=step,
                        effective_step=effective_step)
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], num_classes)
        self.pool_window_size = pool_window_size

    def forward(self, inputs):
        inputs = inputs.permute(1,0,2,3,4)
        # print(inputs.shape)
        # inputs: [seq, b, c, h, w] (e.g. [5, 1, 512, 64, 32])
        xs = self.convlstm(inputs)[0]

        return [F.max_pool2d(F.relu(x), self.pool_window_size) for x in xs]

def ConvLSTM_HMDB2(pretrained='/mnt-gluster/data/yxue/research/allstate'):
    hidden_channels = [32]
    pool_window_size = 56
    hidden_sizes = [hidden_channels[-1]*(224//pool_window_size)*(224//pool_window_size), 1024]
    model = ConvLSTMCLF(2, input_channels=3, hidden_channels=hidden_channels, kernel_size=3,
            step=16, effective_step=list(range(16)), hidden_sizes=hidden_sizes, pool_window_size=pool_window_size)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(pretrained))

    return model

class ConvLSTMCLFWithResNet(nn.Module):
    def __init__(self, num_classes, 
                input_channels, hidden_channels, kernel_size, 
                step=1, effective_step=[1], 
                hidden_sizes=[16384,128], pool_window_size=2,
                dropout=0.1):
        super(ConvLSTMCLFWithResNet, self).__init__()

        self.video_input_perph = VideoInputPeripheral(output_dim=input_channels,
                                                      dropout=dropout,
                                                      freeze_layers=True)

        self.convlstm = ConvLSTM(input_channels=input_channels, hidden_channels=hidden_channels, kernel_size=kernel_size, step=step,
                        effective_step=effective_step)
        # self.dropout1 = nn.Dropout2d(0.3)
        # # self.dropout2 = nn.Dropout2d(0.5)
        # self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], num_classes)
        self.pool_window_size = pool_window_size

    def forward(self, inputs):
        # inputs: [b, seq, c, h, w]
        # print('0: ', inputs.shape)
        inputs = self.video_input_perph.encode(inputs)
        # inputs: [b, seq, h, w, out_dim]
        # print('1: ', inputs.shape)
        inputs = inputs.permute(1,0,4,2,3)
        # print('2: ', inputs.shape)
        # inputs: [seq, b, out_dim h, w]

        # inputs: [seq, b, c, h, w] (e.g. [5, 1, 512, 64, 32])
        xs = self.convlstm(inputs)[0]
        # x: [b, c, h, w] (e.g. [1, 32, 64, 32])
        # print('0:', x.shape)

        return [F.max_pool2d(F.relu(x), self.pool_window_size) for x in xs]
        # x = F.relu(x)
        # x = F.max_pool2d(x, self.pool_window_size)
        # # x: [1, 32, 32, 16]
        # # print('1:', x.shape)

        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # # print('2:', x.shape)
        # # x: [1, 16384]

        # x = self.fc1(x)
        # # x: [1, 128]

        # x = F.relu(x)
        # # x = self.dropout2(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # # x: [1, 2]

        # output = F.log_softmax(x, dim=1)
        # output: [1, 2]

        # return output

def ConvLSTM_HMDB(pretrained='/files/yxue/research/allstate'):
    hidden_channels = [64]
    hidden_sizes = [hidden_channels[-1]*4*4, 512]
    model = ConvLSTMCLFWithResNet(2, input_channels=1024, hidden_channels=hidden_channels, kernel_size=3, 
        step=16, effective_step=list(range(16)), hidden_sizes=hidden_sizes, pool_window_size=3,
        dropout=0.5)
    model.load_state_dict(torch.load(pretrained)['model'])

    return model


if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
                        effective_step=[4]).cuda()
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(5, 1, 512, 64, 32)).cuda()
    target = Variable(torch.randn(5, 1, 32, 64, 32)).double().cuda()

    output = convlstm(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
    