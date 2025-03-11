import torch
import torch.nn as nn
try:
    from models.swin import swin_tiny_patch4_window7_224
except:
    from swin import swin_tiny_patch4_window7_224
import torch.nn.functional as F


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=196, K=640, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

        self.ce = nn.CrossEntropyLoss()

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return self.ce(logits, labels)


class PPCA_RGBT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cl = MoCo(base_encoder=swin_tiny_patch4_window7_224)

    def forward(self, r, b, t):
        # The latter parameter will enter the queue.
        # Taking BM (Meng. et al. ECCV 2024) for example, need to generate the broker modality and add them to the dataset folder first.
        # If using two-branch base approaches, remove the broker-related operation.

        # Setting data augment A(.) here.
        r_r_loss = self.cl(r, r)
        t_t_loss = self.cl(t, t)
        b_b_loss = self.cl(b, b)

        r_b_loss = self.cl(r, b)
        b_t_loss = self.cl(b, t)
        t_r_loss = self.cl(t, r)
        
        loss = r_r_loss+t_t_loss+b_b_loss+r_b_loss+b_t_loss+t_r_loss
        return loss/6


if __name__ == "__main__":
    model = PPCA_RGBT().to("cuda")
    r = torch.randn(8, 3, 224, 224).to("cuda")
    b = torch.randn(8, 3, 224, 224).to("cuda")
    t = torch.randn(8, 3, 224, 224).to("cuda")
    print(model(r, b, t))
