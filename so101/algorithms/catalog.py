"""Built-in setup routines and learned policies, in lab order."""

from __future__ import annotations

from so101.algorithms.loaders import load_act, load_hybrid, load_smolvla, train_act
from so101.algorithms.registry import Algorithm


def builtin_algorithms() -> list[Algorithm]:
    setup = [
        Algorithm(
            name="find-port",
            family="setup",
            summary="Find the follower and leader USB serial ports and write config/ports.env.",
            run="bash scripts/01_find_port.sh",
            module="scripts/01_find_port.sh",
        ),
        Algorithm(
            name="setup-motors-follower",
            family="setup",
            summary="Assign follower motor IDs 1–6 (shoulder_pan through gripper). Power only the follower.",
            run="bash scripts/02_setup_motors_follower.sh",
            module="scripts/02_setup_motors_follower.sh",
        ),
        Algorithm(
            name="setup-motors-leader",
            family="setup",
            summary="Assign leader motor IDs 7–12. Power only the leader.",
            run="bash scripts/03_setup_motors_leader.sh",
            module="scripts/03_setup_motors_leader.sh",
        ),
        Algorithm(
            name="calibrate-follower",
            family="setup",
            summary="Record follower joint ranges. One arm powered, 12V STS3215 only.",
            run="bash scripts/04_calibrate_follower.sh",
            module="scripts/04_calibrate_follower.sh",
        ),
        Algorithm(
            name="calibrate-leader",
            family="setup",
            summary="Record leader joint ranges. One arm powered, 12V STS3215 only.",
            run="bash scripts/05_calibrate_leader.sh",
            module="scripts/05_calibrate_leader.sh",
        ),
        Algorithm(
            name="find-cameras",
            family="setup",
            summary="List cameras so config/camera.env can point at the Omen ZMQ stream.",
            run="bash scripts/07_find_cameras.sh",
            module="scripts/07_find_cameras.sh",
        ),
        Algorithm(
            name="omen-publish",
            family="setup",
            summary="Stream wrist RGB, wrist depth, and overhead JPEG from the Windows sensing PC.",
            run="python omen/publish_d405.py",
            module="omen/publish_d405.py",
        ),
        Algorithm(
            name="teleoperate",
            family="setup",
            summary="Leader drives the follower live, with no dataset logging.",
            run="bash scripts/06_teleoperate.sh",
            module="scripts/06_teleoperate.sh",
        ),
        Algorithm(
            name="record",
            family="setup",
            summary="Log leader actions, follower state, and cameras into data/datasets/<name>/.",
            run="python -m so101 record",
            module="so101.record",
        ),
        Algorithm(
            name="replay",
            family="setup",
            summary="Play one episode's action.npy open-loop on the follower.",
            run="python -m so101 replay",
            module="so101.replay",
        ),
        Algorithm(
            name="home",
            family="setup",
            summary="Move the follower in joint space to the mean demonstration start pose.",
            run="Called by eval before a rollout (so101.home.interpolate_to_qpos).",
            module="so101.home",
        ),
        Algorithm(
            name="depth-grasp",
            family="setup",
            summary="Near contact, override vertical joints from wrist depth (mm) and close the gripper.",
            run="Tuned in config/hybrid_grasp.yaml. Used only through --policy hybrid.",
            module="so101.depth_grasp",
        ),
        Algorithm(
            name="hybrid",
            family="setup",
            summary="ML policy for approach and place; depth-grasp takes the contact phase.",
            run=(
                "python -m so101 eval --policy hybrid --policy-base act "
                "--grasp-config config/hybrid_grasp.yaml"
            ),
            module="so101.hybrid_policy",
            load=load_hybrid,
        ),
        Algorithm(
            name="protocol",
            family="setup",
            summary="Repeat independent trials per scene condition and record binary y/n success.",
            run="python -m so101 eval --protocol config/eval/pen_to_keys.yaml",
            module="so101.eval_protocol",
        ),
        Algorithm(
            name="convert",
            family="setup",
            summary="Export demonstrations to LeRobot format (RGB only) for SmolVLA fine-tuning.",
            run="python -m so101 convert",
            module="so101.convert_lerobot",
        ),
    ]
    ml = [
        Algorithm(
            name="act",
            family="ml",
            summary="Action Chunking Transformer, PyTorch ResNet18, chunks of 100. Uses wrist depth when the dataset has it.",
            run=(
                "train: python -m so101 train --algo act\n"
                "eval:  python -m so101 eval --policy act"
            ),
            module="so101.models.act",
            load=load_act,
            train=train_act,
        ),
        Algorithm(
            name="act-jax",
            family="ml",
            summary="Same ACT objective with a Flax tiny vision backbone. The checkpoint config selects this backend at eval.",
            run=(
                "train: python -m so101 train --algo act-jax\n"
                "eval:  python -m so101 eval --policy act"
            ),
            module="so101.models.act_jax",
            load=load_act,
            train=train_act,
        ),
        Algorithm(
            name="smolvla",
            family="ml",
            summary="Language-conditioned vision-language-action policy. Wrist depth is not a model input.",
            run=(
                "train: bash scripts/13_convert_lerobot.sh && bash scripts/14_train_smolvla.sh\n"
                'eval:  python -m so101 eval --policy smolvla --task "Place keys into metal cup"'
            ),
            module="so101.models.smolvla_wrap",
            load=load_smolvla,
            train=None,
        ),
    ]
    return setup + ml
