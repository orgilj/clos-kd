import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, LBFGS
import torch
import torch.nn as nn
import math
from torch import Tensor
def make_fourier_eye(
    hidden_dim: int = 768,
    seq_len: int = 512,
    device: str = "cuda",
    base: float = 10000.0,      # LLaMA/RoPE-той яг ижил base
    include_sin_cos: bool = True  # True → sin + cos (хамгийн сайн)
) -> torch.Tensor:
    """
    Returns: [seq_len * hidden_dim, hidden_dim] 
    Fourier features stacked over sequence positions (RoPE-style)
    """
    pos = torch.arange(seq_len, device=device).float()           # [S]
    dim = torch.arange(hidden_dim, device=device).float()        # [H]

    # RoPE шиг 10000^(-2i/d) frequency
    inv_freq = 1.0 / (base ** (torch.arange(0, hidden_dim, 2, device=device).float() / hidden_dim))
    inv_freq = inv_freq[:hidden_dim // 2]  # [H/2]

    # [S, H/2]
    angles = pos[:, None] * inv_freq[None, :]

    if include_sin_cos:
        # Sin + Cos concatenation → [S, H]
        fourier = torch.cat([angles.cos(), angles.sin()], dim=-1)
    else:
        # Зөвхөн sin эсвэл cos
        fourier = angles.sin()  # эсвэл .cos()

    # [S, H] → [S*H, H] (CLOS-д шууд оруулахад бэлэн)
    # fourier = fourier.reshape(seq_len * hidden_dim, hidden_dim)
    return fourier.contiguous()


class Clos(nn.Module):
    def __init__(self, in_features=768, out_features=None, channel=3, switches=None, bias=True, middle_switch_multiplier=4):
        """switches={int: bin, b1, b2, b3, bout}
        in_features=768, out_features=768,
        channel=3,  bias=True"""
        super(Clos, self).__init__()

        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features
        self.channel = channel
        self.bias = bias
        self.middle_switch_multiplier = middle_switch_multiplier
        self.switches = {}

        # orolt garaltiin tootsoo hiine, custom input ugvul update hiine
        self.find_factors()
        
        if switches is not None:
            self.switches.update(switches)

        # weightuud
        k = 1.0 / math.sqrt(in_features)       
        self.weight1 = nn.Parameter(torch.Tensor( 
                        self.switches['bin'], 
                        self.switches['b1'], 
                        self.switches['b2']
                        ))

        self.weight2 = nn.Parameter(torch.Tensor(   
                        self.switches['b1'],
                        self.switches['b2'],
                        self.switches['b3'],
                        ))

        self.weight3 = nn.Parameter(torch.Tensor(
                        self.switches['b2'],
                        self.switches['b3'], 
                        self.switches['bout']
                        ))

        # bias
        if self.bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.switches['b1']))
            self.bias2 = nn.Parameter(torch.Tensor(self.switches['b2']))
            self.bias3 = nn.Parameter(torch.Tensor(self.switches['b3']))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)
            self.register_parameter('bias3', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight3, a=math.sqrt(5))
        
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias1, -bound, bound)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias2, -bound, bound)

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight3)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias3, -bound, bound)

    def find_factors(self):
        for i in range(int(math.sqrt(self.in_features)), 0, -1):
            if self.in_features % i == 0:
                self.switches['bin'] = i
                self.switches['b1'] = self.in_features // i
                break
        
        for i in range(int(math.sqrt(self.out_features)), 0, -1):
            if self.out_features % i == 0:
                self.switches['bout'] = i
                self.switches['b3'] = self.out_features // i
                break

        self.switches['b2'] = self.middle_switch_multiplier * self.switches['bin'] #Middle switch multiplier 4

    def __repr__(self):
        return (
            f"Clos(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias}, "\
            f"bin={self.switches['bin']}, "\
            f"b1={self.switches['b1']}, "\
            f"b2={self.switches['b2']}, "\
            f"b3={self.switches['b3']}, "\
            f"bout={self.switches['bout']}, "\
            f"channel={self.channel})"
        )

    def channel2(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.view(b, self.switches['bin'], self.switches['b1'])

        if self.bias:
            x = torch.einsum('bnr,nrm->bmr', x, self.weight1) + self.bias1
            x = torch.einsum('bmr,rmn->bnm', x, self.weight2) + self.bias2
            x = torch.einsum('bnm,mro->bor', x, self.weight3) + self.bias3
        else:
            x = torch.einsum('bnr,nrm->bmr', x, self.weight1)
            x = torch.einsum('bmr,rmn->bnm', x, self.weight2)
            x = torch.einsum('bnm,mro->bor', x, self.weight3)

        return x.reshape(b, -1)

    # Channel=3: for attention-style [B, C, H] (e.g., batch, heads, dim)
    def channel3(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        x = x.view(b, c, self.switches['bin'], self.switches['b1'])

        if self.bias:
            x = torch.einsum('bcnr,nrm->bcmr', x, self.weight1) + self.bias1
            x = torch.einsum('bcmr,rmn->bcnm', x, self.weight2) + self.bias2
            x = torch.einsum('bcnm,mro->bcor', x, self.weight3) + self.bias3
        else:
            x = torch.einsum('bcnr,nrm->bcmr', x, self.weight1)
            x = torch.einsum('bcmr,rmn->bcnm', x, self.weight2)
            x = torch.einsum('bcnm,mro->bcor', x, self.weight3)

        return x.reshape(b, c, -1)
    def forward(self, input: Tensor) -> Tensor:
        if self.channel == 2:
            return self.channel2(input)
        elif self.channel == 3:
            return self.channel3(input)


def transfer_fc_to_clos(fc: nn.Linear, channel: int = 2, max_steps: int = 2500, W_lr: float = 1e-3, B_lr: float = 1e-5, verbose: bool = False, save_path: str="./tmp/clos.pth"):
    """
    input:
        - Fc: Learned Fully connected layers from pytorch. What linear want to clone.
        - channel: Fc layer processing for sequential or in attention mechanism channel=3 [B,S,H]. Otherwise channel=2 [B,H]
        - max_steps: optimization steps for clone: learn knowledge from linear from dummy[eye matrix and zeros] inputs processed output of linear layer
        - W_lr: learning rate of w optimization. If want any optimization can be used. Found AdamW for best in this time.
        - B_lr: bais optimization learning and final optimization.
        - verbose: print cloning process.
    output:
        - clos: cloned from linear layer
    """
    device = fc.weight.device
    in_f, out_f = fc.in_features, fc.out_features
    clos = Clos(in_features=in_f, out_features=out_f, channel=channel, bias=True).to(device)
    # print(clos)
    clos.train()
    target_for_mse = fc.weight.T.contiguous()
    # ========== ЗӨВ EYE MATRIX ҮҮСГЭХ ==========
    if channel == 2:
        # MLP шиг: [batch, hidden]
        eye = torch.eye(in_f, device=device)  # [1, in_f] → clos(eye) → [1, out_f]
        # target_for_mse = target_weight  # [in_f, out_f]

    elif channel == 3:
        # BERT Attention шиг: [batch, seq_len, hidden]  
        target_for_mse = target_for_mse.unsqueeze(0)  # [S*in_f, out_f] # in here the main problem occures. how to process channel 3 input of eye matrix in bert like structure.
        eye = make_fourier_eye(
                                    hidden_dim=in_f,   # 768
                                    seq_len=768,
                                    device=fc.weight.device
                                ).unsqueeze(0)   # [1, S*H, H]
    
    optimizer = AdamW(clos.parameters(), lr=W_lr, weight_decay=1e-3)
    for step in range(max_steps):
        optimizer.zero_grad()
        pred = clos(eye)
        # print(pred.shape, target_for_mse.shape)
        loss = F.mse_loss(pred, target_for_mse)
        loss.backward()
        optimizer.step()
        if verbose:
            if step % 1000 == 0 or step == max_steps - 1:
                print(f"  step {step:4d} │ loss {loss.item():.10f}")
    print("LBFGS polish...")
    optimizer = LBFGS(clos.parameters(), lr=0.1, max_iter=1000, history_size=40, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        pred = clos(eye)
        if channel ==2:
            loss = F.mse_loss(pred, target_for_mse)
        else:
            loss = F.mse_loss(pred, target_for_mse.unsqueeze(0))
        loss.backward()
        return loss
    for _ in range(int(8)):
        optimizer.step(closure)
    if verbose:
        print(f"Final weight MSE: {F.mse_loss(clos(eye), target_for_mse).item():.10f}")
    if fc.bias is not None:
        target_bias = fc.bias.clone().unsqueeze(0).expand(out_f, out_f)
        clos.bias1.requires_grad_(True)
        clos.bias2.requires_grad_(True)
        clos.bias3.requires_grad_(True)
        clos.train()
        if channel == 2:
            zero_input = torch.zeros(1, in_f, device=device, requires_grad=False)
        elif channel == 3:
            zero_input = torch.zeros(1, in_f, in_f, device=device, requires_grad=False)
        bias_params = [clos.bias1, clos.bias2, clos.bias3]
        bias_opt = AdamW(bias_params, lr=B_lr, weight_decay=1e-2)
        for _ in range(1000): # final optimization.
            bias_opt.zero_grad()
            out_at_zero = clos(zero_input)       # (784,)
            if channel == 2:
                # print("out at zero shape: ", out_at_zero.shape)
                b_loss = F.mse_loss(out_at_zero.unsqueeze(0), target_bias)
            else:  # channel 3
                b_loss = F.mse_loss(out_at_zero, target_bias.unsqueeze(0))
            b_loss.backward()
            bias_opt.step()
            if verbose:
                if _ % 100 == 0 or _ == 999:
                    print(f"    bias step {_}: loss = {b_loss.item():.2e}")
        print(f"Final MSE: {b_loss.item():.2e}")
    return clos
