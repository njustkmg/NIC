import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

from proselflc import ProSelfLC

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.params = {
                'total_epochs': 25,
                'exp_base': 10,
                'counter': 'iteration',
                'transit_time_ratio': 0.5,
                'total_iterations': 25 * 11328 * 2
            }
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
            self.crit_lc = ProSelfLC(self.params)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, cur_time, epoch, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag):
        out = {}
        if not sc_flag:
            outputs, pred_outputs, atts_distribution, att_masks = self.model(fc_feats, att_feats, labels, att_masks)
            loss, epsilon, epsilon_context, epsilon_attention, \
                sentence_loss, H_pred = self.crit_lc(outputs, labels[:,1:], masks[:,1:], pred_outputs, atts_distribution, att_masks, cur_time, epoch)
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample')
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_method':'sample'}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out, epsilon, epsilon_context, epsilon_attention, sentence_loss, H_pred
