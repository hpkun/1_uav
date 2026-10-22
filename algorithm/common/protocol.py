ALGORITHM_VERSION = 'mappo-coop3d-v3.0.0'
PROTOCOL = dict(actor_input=12, actor_action=3, critic_input=90, critic_output=5,
                current_state='pre-action 90D', next_state='real successor before auto-reset',
                current_mask='pre-action valid UAV', next_mask='real successor valid UAV',
                sampled_steps='sum of joint transitions across workers')
