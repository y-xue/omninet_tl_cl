import multiprocessing

def ensure_shared_grads(model, shared_model, gpu_id):
    shared_params=shared_model.parameters()
    print("Sharing")
    for param, shared_param in zip(model.parameters(),
                                   shared_params):
        if param.grad is not None:
            shared_param._grad = param.grad.to(0)

class Counter(object):
    def __init__(self, initval=0):
        self.val = multiprocessing.RawValue('i', initval)
        self.lock = multiprocessing.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1
            return self.val.value

    @property
    def value(self):
        return self.val.value

def overfitting(seq, last_val_accs, val_acc, decrease_step_cnt, overfitting_threshold):
    if seq not in last_val_accs:
        return False

    if (last_val_accs[seq] - val_acc)/val_acc > overfitting_threshold:
        decrease_step_cnt[seq] += 1
        if decrease_step_cnt[seq] > 1:
            return True
    return False