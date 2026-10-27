"""Unified label schema for Phase 1 (lungs + heart) and the organ -> disease map.

Every dataset is mapped onto these names. A label a dataset does not annotate is
left as NaN so the training loss can mask it out later.
"""

ORGAN_LABELS = {
    "lungs": [
        "Atelectasis", "Consolidation", "Edema", "Pneumonia", "Lung Opacity", "Lung Lesion",
        "Infiltration", "Mass", "Nodule", "Emphysema", "Fibrosis",
    ],
    "pleura": ["Pleural Effusion", "Pneumothorax", "Pleural Thickening"],   # the lining around the lungs
    "heart": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
}
# Models trained BEFORE this split (e.g. run "phase1") keep their own organ layout inside the checkpoint
# (lungs incl. pleural findings, heart), so they continue to work unchanged.

UNIFIED_LABELS = [l for labels in ORGAN_LABELS.values() for l in labels]
LABEL_TO_ORGAN = {l: o for o, labels in ORGAN_LABELS.items() for l in labels}

# Dataset-specific name -> unified name
NIH_RENAME = {"Effusion": "Pleural Effusion", "Pleural_Thickening": "Pleural Thickening"}
NIH_ANNOTATED = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion",
    "Emphysema", "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax",
]  # (NIH's Hernia is a diaphragm finding -> future work, dropped)

CHEXPERT_ANNOTATED = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion",
]  # (CheXpert's Fracture -> Phase 2)


def active_organ_labels(master) -> dict:
    """organ -> labels that have at least one annotation in the TRAIN split.

    With NIH only, 'Lung Opacity', 'Lung Lesion' and 'Enlarged Cardiomediastinum'
    are all-NaN, so they drop out automatically. When CheXpert is added later and
    master.csv is rebuilt, they come back with no code change.
    """
    train = master[master["split"] == "train"]
    out = {}
    for organ, labs in ORGAN_LABELS.items():
        keep = [l for l in labs if train[l].notna().any()]
        if keep:
            out[organ] = keep
    return out
