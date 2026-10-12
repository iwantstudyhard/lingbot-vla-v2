"""Compare actual repository forward paths without optimizer/backward or weight writes."""
from contextlib import contextmanager
import torch


@contextmanager
def forward_mode(flow, attention, training=False, cache=False):
    backbone = flow.qwenvl_with_expert
    flags = [(m, m.training) for m in flow.modules()]
    config_fields = ('use_cache', 'align_params', 'sequence_wise_loss_coeff', 'router_z_loss_coeff')
    configs = {k: getattr(flow.config, k, None) for k in config_fields}
    old_attention, old_interface = backbone.config.attention_implementation, backbone.attention_interface
    # Only train-mode MoE monitor buffers mutate in this no-grad probe.
    buffers = [(v, v.clone()) for name, v in flow.named_buffers()
               if name.rsplit('.', 1)[-1] in ('tokens_per_expert', 'last_tokens_per_expert', 'avg_topk_sigmoid_score')]
    try:
        flow.train(training)
        backbone.config.attention_implementation = attention
        backbone.attention_interface = backbone.get_attention_interface()
        flow.config.use_cache = cache
        # Auxiliary losses are AFTER the shared action backbone. No teachers
        # needed; task tokens/heads remain instantiated and unchanged.
        flow.config.align_params = {}
        flow.config.sequence_wise_loss_coeff = 0.0
        flow.config.router_z_loss_coeff = 0.0
        with torch.no_grad():
            yield
    finally:
        for k, value in configs.items():
            setattr(flow.config, k, value)
        backbone.config.attention_implementation = old_attention
        backbone.attention_interface = old_interface
        for module, flag in flags:
            module.training = flag
        for destination, value in buffers:
            destination.copy_(value)


def full_velocity(flow, inputs, actions, noise, time, attention='eager', training=False):
    """Call actual FlowMatchingV2.forward and capture its action projection."""
    if getattr(flow.config, 'action_fp32', False):
        raise ValueError('This path probe currently supports action_fp32=False (the supplied 18k run) only')
    captured = []
    def capture(module, args, output):
        captured.append(output.detach().clone())
    handle = flow.action_out_proj.register_forward_hook(capture)
    try:
        with forward_mode(flow, attention, training=training):
            flow.forward(**inputs, actions=actions, noise=noise.clone(), time=time, loss_type='L1_fm')
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f'Expected exactly one action projection, got {len(captured)}')
    return captured[0]


def cached_velocity(flow, inputs, x_t, time):
    """Use the exact deployment prefix-prefill and predict_velocity methods."""
    from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks
    with forward_mode(flow, 'eager', cache=True):
        prefix, padding, attention, positions, visual_mask, deepstack = flow.embed_prefix(
            inputs['images'], inputs['img_masks'], inputs['lang_tokens'], inputs['lang_masks'],
            image_grid_thw=inputs.get('image_grid_thw'))
        _, cache, _ = flow.qwenvl_with_expert.forward(
            attention_mask=make_att_2d_masks(padding, attention),
            position_ids=positions, vlm_position_ids=positions, past_key_values=None,
            inputs_embeds=[prefix, None], use_cache=True, fill_kv_cache=True,
            visual_pos_masks=visual_mask, deepstack_visual_embeds=deepstack)
        return flow.predict_velocity(inputs['state'], padding, cache, x_t.clone(), time,
                                     prefix_position_ids=positions).detach().clone()
