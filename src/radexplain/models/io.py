import torch

from radexplain.models.radexplain_net import RadExplainNet


def load_model(ckpt_path, device):
    """Rebuild the network exactly as it was trained and load its weights. Returns (model, checkpoint)."""
    ck = torch.load(ckpt_path, map_location="cpu")
    model = RadExplainNet(ck["organ_labels"], ck["backbone"], pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck
