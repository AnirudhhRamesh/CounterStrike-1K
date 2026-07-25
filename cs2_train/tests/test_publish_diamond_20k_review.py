from cs2_train.scripts.publish_diamond_20k_review import update_index


def test_update_index_replaces_endpoint_event_and_marks_skipped_arm() -> None:
    raw = {
        "arms": {
            "true": {
                "events": [
                    {"step": 20_000, "kind": "confirmatory_midpoint", "old": True},
                    {"step": 17_500, "kind": "rollout"},
                ],
                "status": "running",
            }
        }
    }
    event = {"step": 20_000, "kind": "confirmatory_midpoint", "new": True}

    update_index(raw, window="midpoint", event=event, step=20_000)

    assert raw["arms"]["true"]["events"][-1] == event
    assert not any(item.get("old") for item in raw["arms"]["true"]["events"])
    assert raw["arms"]["true"]["final_step"] == 20_000
    assert raw["arms"]["shuffled"]["status"] == "skipped_resource_amendment"
    assert raw["final_results_status"] == "complete_resource_amended"
