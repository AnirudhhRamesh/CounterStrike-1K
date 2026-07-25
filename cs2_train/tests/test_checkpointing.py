from argparse import Namespace

import torch

from cs2_train.src.train import save_checkpoint


def test_checkpoint_is_atomic_and_contains_resume_state(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _step: 1.0,
    )
    path = tmp_path / "step_0010000.pt"

    save_checkpoint(
        path,
        model=model,
        optim=optimizer,
        sched=scheduler,
        step=10_000,
        best_val_mse=0.125,
        args=Namespace(seed=28, action_mode="true"),
    )

    assert path.is_file()
    assert not path.with_suffix(".pt.tmp").exists()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == 10_000
    assert payload["best_val_mse"] == 0.125
    assert payload["args"] == {"seed": 28, "action_mode": "true"}
    assert payload["model"].keys() == model.state_dict().keys()
    assert payload["optim"]["param_groups"] == optimizer.state_dict()["param_groups"]
    assert payload["sched"] == scheduler.state_dict()
