import torch.nn.functional as F
import torch
from mmengine.runner import Hook
from mmengine.registry import HOOKS
@HOOKS.register_module()
class KnowledgeDistillationHook(Hook):
    def __init__(self, teacher_model, alpha, temperature):
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

    def after_train_iter(self, runner,batch_idx, data_batch, outputs):
        student_output = outputs['preds']
        with torch.no_grad():
            teacher_output = self.teacher_model(runner.inputs['img'])

        loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction='batchmean',
        )
        outputs['loss'] = outputs['loss'] * (1 - self.alpha) + loss * self.alpha