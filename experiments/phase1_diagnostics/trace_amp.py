"""Localize AMP failure with tensor-gradient hooks; no optimizer updates.

Wait for the primary run to release the GPU. Every condition restores epoch-18
weights and random state. Targeted FP32 islands and cuDNN-off are diagnostics,
not adopted training configurations or full runs.
"""
import json
import time
import types
import gc
import run_phase1 as run

torch = run.torch
HERE = run.HERE


def leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from leaves(item)


def cast(value):
    if torch.is_tensor(value) and value.is_floating_point():
        return value.float()
    if isinstance(value, tuple):
        return tuple(cast(x) for x in value)
    if isinstance(value, list):
        return [cast(x) for x in value]
    return value


def main():
    while not (HERE / 'preservation_verification.json').exists():
        time.sleep(5)
    if not (HERE / 'phase1_summary.json').exists():
        raise RuntimeError('Primary run did not finish; do not allocate a second model.')
    phase_summary = json.loads((HERE / 'phase1_summary.json').read_text())
    if phase_summary.get('training_restarted_after_oom') is False or (HERE / 'failure.json').exists():
        raise RuntimeError('OOM stop boundary is active; further GPU diagnostics require a new approved attempt.')
    before = json.loads((HERE / 'preservation_before.json').read_text())
    torch.set_num_threads(2)
    run.seed()
    saved = torch.load(run.CKPT, map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('model.'): v for k, v in saved['model'].items()}
    cfg = run.Config(data_root=run.DATA)
    vision, text_source = run._model_sources(cfg)
    tokenizer = run.AutoTokenizer.from_pretrained(text_source, trust_remote_code=True, local_files_only=True)
    data = run.QaTaDataset(run.CSV, run.DATA, 'train', tokenizer, augment=False, seed=run.SEED)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    item = data[protocol['subset_indices'][0]]
    model = run.create_model(vision, text_source).to(run.DEVICE)
    capture = run.LogitCapture(model)
    modules = dict(model.named_modules())
    conditions = [
        ('fp32_reference', False, None, True, True),
        ('amp_trace', True, None, True, True),
        ('amp_final_convblock_fp32', True, 'decoder4.decoder.conv_block', True, True),
        ('amp_final_transpose_fp32', True, 'decoder4.decoder.transp_conv', True, True),
        ('amp_final_decoder_fp32', True, 'decoder4.decoder', True, True),
        ('amp_cudnn_disabled', True, None, False, True),
        ('amp_nondeterministic_backend', True, None, True, False),
    ]
    results = []
    try:
        for label, amp, island, cudnn_enabled, deterministic in conditions:
            print('AMP LOCALIZATION', label, flush=True)
            run.restore(model, state)
            model.train()
            run.seed()
            run.reset_memory()
            events, hooks = [], []
            def hook_for(name):
                def forward_hook(module, inputs, output):
                    for i, tensor in enumerate(leaves(output)):
                        if tensor.requires_grad:
                            def backward_hook(gradient, name=name, i=i):
                                info = run.tensor_info(gradient)
                                events.append({'order': len(events), 'module': name, 'output_index': i, **info})
                            tensor.register_hook(backward_hook)
                return forward_hook
            for name, module in modules.items():
                if name.startswith(('decoder4', 'decoder1', 'out')):
                    hooks.append(module.register_forward_hook(hook_for(name)))
            original_forward = None
            if island:
                target_module = modules[island]
                original_forward = target_module.forward
                def float_forward(self, *args, **kwargs):
                    with torch.autocast('cuda', enabled=False):
                        return original_forward(*(cast(a) for a in args), **kwargs)
                target_module.forward = types.MethodType(float_forward, target_module)
            previous_cudnn = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = cudnn_enabled
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = deterministic
            torch.use_deterministic_algorithms(deterministic, warn_only=True)
            try:
                image, text, target = run.to_device(item)
                with torch.autocast('cuda', enabled=amp):
                    probability = model([image, text])
                loss, _, _, _ = run.stable_loss(capture.logits, target)
                loss.backward()  # Scale=1; no optimizer and no parameter update.
                torch.cuda.synchronize()
                info = run.grad_info(model)
                first_bad = next((r for r in events if r['nonfinite_elements']), None)
                row = {'condition': label, 'loss': float(loss.detach()), 'amp': amp,
                       'scale': 1, 'fp32_island': island, 'cudnn_enabled': cudnn_enabled,
                       'cudnn_deterministic': deterministic, 'deterministic_algorithms': deterministic,
                       'first_nonfinite_tensor_hook': first_bad, 'backward_events': events,
                       'optimizer_updates': 0, **info, **run.memory()}
                results.append(row)
                print(label, 'bad gradients', info['nonfinite_gradient_tensors'],
                      'first bad', first_bad, flush=True)
                run.dump('diagnostics/amp_localization.json', results)
                capture.clear()
                del image, text, target, probability, loss
            finally:
                torch.backends.cudnn.enabled = previous_cudnn
                if island:
                    target_module.forward = original_forward
                for hook in hooks:
                    hook.remove()
            gc.collect()
            torch.cuda.empty_cache()
        run.dump('diagnostics/amp_localization_complete.json', {'conditions': len(results), 'complete': True})
    except Exception as exc:
        run.dump('diagnostics/amp_localization_failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        run.preservation_end(before)


if __name__ == '__main__':
    main()
