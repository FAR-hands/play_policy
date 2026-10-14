"""DSP play-policy library: data loading, the DiT play policy and the DiT world model.

    state  = [hand_joint_pos (22), cube_pose (pos3 + quat_wxyz4)]   (29-D, normalized)
    action = cmd_q22 (22) joint-position commands

``play_policy.train`` trains the goal-free play policy on teleop takes (behavior cloning);
``play_policy.train_worldmodel`` trains the action-conditioned world model on autonomous play.
Loading for inference: ``isaac/deploy_core.load_policy`` and
``play_policy.model.load_world_model_ckpt``. See README.md in this directory.
"""
