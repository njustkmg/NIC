import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from exceptions import ParamException
from crossentropy import CrossEntropy


class ProSelfLC(CrossEntropy):
    """
    The implementation for progressive self label correction (CVPR 2021 paper).
    The target probability will be corrected by
    a predicted distributions, i.e., self knowledge.
        1. ProSelfLC is partially inspired by prior related work,
            e.g., Pesudo-labelling.
        2. ProSelfLC is partially theorectically bounded by
            early stopping regularisation.

    Inputs: two tensors for predictions and target.
        1. predicted probability distributions of shape (N, C)
        2. target probability  distributions of shape (N, C)
        3. current time (epoch/iteration counter).
        4. total time (total epochs/iterations)
        5. exp_base: the exponential base for adjusting epsilon
        6. counter: iteration or epoch counter versus total time.

    Outputs: scalar tensor, normalised by the number of examples.
    """

    def __init__(
        self,
        params: dict = None,
    ) -> None:
        super().__init__()
        self.total_epochs = params["total_epochs"]
        self.exp_base = params["exp_base"]
        self.counter = params["counter"]
        self.epsilon = None
        self.epsilon_context = None
        self.epsilon_attention = None
        self.transit_time_ratio = params["transit_time_ratio"]

        # aoa
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        smoothing=0.0
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

        if not (self.exp_base >= 0):
            error_msg = (
                "self.exp_base = "
                + str(self.exp_base)
                + ". "
                + "The exp_base has to be no less than zero. "
            )
            raise (ParamException(error_msg))

        if not (isinstance(self.total_epochs, int) and self.total_epochs > 0):
            error_msg = (
                "self.total_epochs = "
                + str(self.total_epochs)
                + ". "
                + "The total_epochs has to be a positive integer. "
            )
            raise (ParamException(error_msg))

        if self.counter not in ["iteration", "epoch"]:
            error_msg = (
                "self.counter = "
                + str(self.counter)
                + ". "
                + "The counter has to be iteration or epoch. "
                + "The training time is counted by eithor of them. "
            )
            raise (ParamException(error_msg))

        if "total_iterations" in params.keys():
            # only exist when counter == "iteration"
            self.total_iterations = params["total_iterations"]

    def update_epsilon_progressive_adaptive(self, pred_probs, cur_time):
        with torch.no_grad():
            # global trust/knowledge
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )

            # 希望这个比较小
            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))
            # example-level trust/knowledge
            class_num = pred_probs.shape[1]

            H_pred_probs = torch.sum(
                -(pred_probs + 1e-12) * torch.log(pred_probs + 1e-12), 1
            )  # 熵下降的速度和loss差不多
            H_uniform = -torch.log(torch.tensor(1.0 / class_num))
            example_trust = 1 - H_pred_probs / H_uniform 
            self.epsilon = global_trust * example_trust 
            # from shape [N] to shape [N, 1]
            self.epsilon = self.epsilon[:, None]

            return H_pred_probs

    def context_update_epsilon(self, target_probs, pred_probs, pred_outputs, mask, cur_time):
        with torch.no_grad():
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )
            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))

            target_probs = target_probs
            pred_probs = pred_probs
            pred_outputs = pred_outputs
            mask_origin = mask
            mask =  mask[:, :pred_outputs.size(1)]
            mask = self.to_contiguous(mask).view(-1)
            
            pred_outputs = F.log_softmax(pred_outputs, dim=-1)        
            pred_outputs = self.to_contiguous(pred_outputs).view(-1, pred_outputs.size(-1))

            loss_word = self.criterion(pred_probs, target_probs)
            loss_word = loss_word.sum(1) * mask
            loss_word = loss_word.view(mask_origin.size())

            loss_word_pred = self.criterion(pred_outputs, target_probs)
            loss_word_pred = loss_word_pred.sum(1) * mask
            loss_word_pred = loss_word_pred.view(mask_origin.size())

            cur_time_losses = mask_origin.new_zeros(mask_origin.size())
            next_time_losses = mask_origin.new_zeros(mask_origin.size())
            pred_time_losses = mask_origin.new_zeros(mask_origin.size())
            for i in range(mask_origin.size(1)):
                if i < (mask_origin.size(1)-1)-2:
                    cur_time_loss = loss_word[:, i]
                    next_time_loss = loss_word[:, i+1]
                    mask_sum = mask_origin[:, i+2:].sum(1) == 0
                    mask_sum  = mask_sum * 1.0
                    mask_sum = mask_sum + mask_origin[:, i+2:].sum(1)
                    pred_time_loss = loss_word[:, i+2:].sum(1) / mask_sum
                elif i == (mask_origin.size(1)-1)-1:
                    cur_time_loss = loss_word[:, i]
                    next_time_loss = loss_word[:, i+1]
                    pred_time_loss = loss_word[:, i].new_zeros(loss_word[:, i].size())
                else:
                    cur_time_loss = loss_word[:, i]
                    next_time_loss = loss_word[:, i].new_zeros(loss_word[:, i].size())
                    pred_time_loss = loss_word[:, i].new_zeros(loss_word[:, i].size())

                cur_time_losses[:, i] = cur_time_loss
                next_time_losses[:, i] = next_time_loss
                pred_time_losses[:, i] = pred_time_loss

            cur_time_losses = cur_time_losses.view(-1)
            next_time_losses = next_time_losses.view(-1)
            pred_time_losses = pred_time_losses.view(-1)

            batch_losses = (cur_time_losses + next_time_losses + pred_time_losses) / 30.0
            self.epsilon_context = global_trust * batch_losses
            # from shape [N] to shape [N, 1]
            self.epsilon_context = self.epsilon_context[:, None]
              

    def attention_update_epsilon(self, atts_distribution, att_masks, cur_time):
        """
        对于注意力8个头的分布, 是采用分别算KL再求和, 还是拼接之后再算KL呢?
        感觉拼接更合理一些.
        """
        
        with torch.no_grad():
            # global trust/knowledge
            # 时间维度上的
            if self.counter == "epoch":
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_epochs - self.transit_time_ratio
                )
            else:
                time_ratio_minus_half = torch.tensor(
                    cur_time / self.total_iterations - self.transit_time_ratio
                )
            global_trust = 1 / (1 + torch.exp(-self.exp_base * time_ratio_minus_half))

            atts_distribution_origin = atts_distribution
            att_masks_origin = att_masks
            batch_att_kl = atts_distribution.new_zeros(atts_distribution.size(0), atts_distribution.size(1))

            i = 0
            for sample_att, sample_mask in zip(atts_distribution, att_masks):
                sample_att = sample_att
                sample_mask = sample_mask
                
                sample_att_size = sample_att.size()
                sample_att_sum = self.to_contiguous(sample_att[:, :, :sample_mask.sum(0).int()]).view(sample_att_size[0], -1)
                sample_att_sum = F.log_softmax(sample_att_sum, dim=-1)
                class_att = torch.full_like(sample_att_sum, 1/(sample_att_sum.size(-1)/sample_att_size[1]))
                class_att = F.softmax(class_att, dim=-1)
                sample_attention_kl = (self.criterion(sample_att_sum, class_att).sum(-1))

                batch_att_kl[i, :] = sample_attention_kl
                i += 1

            batch_att_kl = batch_att_kl.view(-1)
            # batch_att_kl = 1.0 - batch_att_kl
            self.epsilon_attention = global_trust * batch_att_kl 
            # from shape [N] to shape [N, 1]
            self.epsilon_attention = self.epsilon_attention[:, None]


    def to_contiguous(self, tensor):
        if tensor.is_contiguous():
            return tensor
        else:
            return tensor.contiguous()

    def logits2probs_softmax(self, logits):
            """
            Transform logits to probabilities using exp function and normalisation

            Input:
                logits with shape: (N, C)
                N means the batch size or the number of instances.
                C means the number of training classes.

            Output:
                probability vectors of shape (N, C)
            """
            # reimplementation of F.softmax(logits)
            # or: nn.Softmax()(logits)
            # per instance:
            # subtract max logit for numerical issues
            subtractmax_logits = logits - torch.max(logits, dim=-1, keepdim=True).values
            exp_logits = torch.exp(subtractmax_logits)
            sum_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
            return exp_logits / sum_logits

    def forward(
        self, pred_probs: Tensor, target_probs: Tensor, mask: Tensor, 
        pred_outputs: Tensor, atts_distribution: Tensor, att_masks: Tensor, cur_time: int, epoch: int
    ) -> Tensor:
        """
        Inputs:
            1. predicted probability distributions of shape (N, C)
            2. target probability  distributions of shape (N, C)
            3. current time (epoch/iteration counter).

        Outputs:
            Loss: a scalar tensor, normalised by N.
        """
        if self.counter == "epoch":
            # cur_time indicate epoch
            if not (cur_time <= self.total_epochs and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_epochs)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))
        else:  # self.counter == "iteration":
            # cur_time indicate iteration
            if not (cur_time <= self.total_iterations and cur_time >= 0):
                error_msg = (
                    "The cur_time = "
                    + str(cur_time)
                    + ". The total_time = "
                    + str(self.total_iterations)
                    + ". The cur_time has to be no larger than total time "
                    + "and no less than zero."
                )
                raise (ParamException(error_msg))

        pred_probs_origin = pred_probs
        # truncate to the same size
        lc_pred_probs = self.logits2probs_softmax(pred_probs)
        aoa_pred_probs = F.log_softmax(pred_probs, dim=-1)
        
        target_probs = target_probs[:, :pred_probs.size(1)]
        mask =  mask[:, :pred_probs.size(1)]
        mask_origin = mask

        lc_pred_probs = self.to_contiguous(lc_pred_probs).view(-1, lc_pred_probs.size(-1))
        aoa_pred_probs = self.to_contiguous(aoa_pred_probs).view(-1, aoa_pred_probs.size(-1))
        target_probs = self.to_contiguous(target_probs).view(-1)
        mask = self.to_contiguous(mask).view(-1)
        
        self.size = lc_pred_probs.size(1)
        true_dist = lc_pred_probs.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target_probs.data.unsqueeze(1), self.confidence)

        # update self.epsilon
        H_pred = self.update_epsilon_progressive_adaptive(lc_pred_probs, cur_time)
        self.epsilon = self.epsilon * 0.01
        H_pred = H_pred.view(pred_probs_origin.size(0), pred_probs_origin.size(1))
        H_pred = H_pred.cpu().numpy().tolist()

        if epoch >= 0:
            self.attention_update_epsilon(atts_distribution, att_masks, cur_time)
            self.context_update_epsilon(true_dist, aoa_pred_probs, pred_outputs, mask_origin, cur_time)
            self.epsilon_context = self.epsilon_context*0.01
            self.epsilon_attention = self.epsilon_attention*0.01
        else:
            self.epsilon_context = self.epsilon
            self.epsilon_attention = self.epsilon
        
        # save self.epsilon
        save_epsilon = self.epsilon.squeeze() * mask
        save_epsilon = save_epsilon.view(pred_probs_origin.size(0), pred_probs_origin.size(1))[:, 0]

        save_epsilon_context = self.epsilon_context.squeeze() * mask
        save_epsilon_context = save_epsilon_context.view(pred_probs_origin.size(0), pred_probs_origin.size(1))[:, 0]

        save_epsilon_attention = self.epsilon_attention.squeeze() * mask
        save_epsilon_attention = save_epsilon_attention.view(pred_probs_origin.size(0), pred_probs_origin.size(1))[:, 0]

        if epoch <= 1:
            # self.epsilon = 0
            self.epsilon_context = 0
            self.epsilon_attention = 0
            
        # new_target_probs = (1 - (self.epsilon + self.epsilon_context)) * true_dist + (self.epsilon + self.epsilon_context) * lc_pred_probs
        new_target_probs = (1 - (self.epsilon + self.epsilon_context + self.epsilon_attention)/3) * true_dist + (self.epsilon + self.epsilon_context + self.epsilon_attention)/3 * lc_pred_probs
        # # sc
        # new_target_probs = (1 - self.epsilon) * true_dist + (self.epsilon) * lc_pred_probs
        # # cic
        # new_target_probs = (1 - (self.epsilon_context)) * true_dist + (self.epsilon_context) * lc_pred_probs
        # imc
        # new_target_probs = (1 - (self.epsilon_attention)) * true_dist + (self.epsilon_attention) * lc_pred_probs
        # # w/o imc
        # new_target_probs = (1 - (self.epsilon + self.epsilon_context)/2) * true_dist + (self.epsilon + self.epsilon_context)/2 * lc_pred_probs
        # # w/o cic
        # new_target_probs = (1 - (self.epsilon + self.epsilon_attention)/2) * true_dist + (self.epsilon + self.epsilon_attention)/2 * lc_pred_probs
        # # w/o sc
        # new_target_probs = (1 - (self.epsilon_context + self.epsilon_attention)/2) * true_dist + (self.epsilon_context + self.epsilon_attention)/2 * lc_pred_probs

        loss_1 = self.criterion(aoa_pred_probs, new_target_probs)
        loss_2 = loss_1.sum(1) * mask
        loss_3 = loss_2.view(pred_probs_origin.size(0), pred_probs_origin.size(1)).sum(1) / mask_origin.sum(1)

        # reuse CrossEntropy's forward computation
        return (self.criterion(aoa_pred_probs, new_target_probs).sum(1) * mask).sum() / mask.sum(), save_epsilon, save_epsilon_context, save_epsilon_attention, loss_3, H_pred
