"""GR00T modality config for the abc-sim bimanual YAM (bottles-in-bin).

Matches pi0.5 / DM0.5 on the same data: relative joints, absolute grippers.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ARMS = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
REL = ActionConfig(
    rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT
)
ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT
)

abc_sim_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["head", "wrist_left", "wrist_right"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=ARMS),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=ARMS,
        action_configs=[REL, ABS, REL, ABS],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

register_modality_config(abc_sim_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
