from blueprint_validation.evaluation.task_success import evaluate_task_success


def test_navigation_task_success_requires_goal_hold_without_collision():
    spec = {
        "success_predicate": {
            "type": "navigation_reach_and_hold",
            "goal_region_id": "region::sink",
            "hold_steps": 2,
            "collision_key": "invalid_collision",
        }
    }
    result = evaluate_task_success(
        task_spec=spec,
        rollout_row={},
        state_trace=[
            {"active_region_id": "region::hallway", "invalid_collision": False},
            {"active_region_id": "region::sink", "invalid_collision": False},
            {"active_region_id": "region::sink", "invalid_collision": False},
        ],
    )
    assert result["task_success"] is True
    assert result["task_success_available"] is True


def test_articulation_task_success_requires_open_then_close_sequence():
    spec = {
        "success_predicate": {
            "type": "articulation_open_close_sequence",
            "target_instance_id": "7",
            "joint_key": "joint_position",
            "open_threshold": 0.8,
            "close_threshold": 0.2,
        }
    }
    result = evaluate_task_success(
        task_spec=spec,
        rollout_row={},
        state_trace=[
            {"joint_positions": {"7": 0.1}, "invalid_collision": False},
            {"joint_positions": {"7": 0.9}, "invalid_collision": False},
            {"joint_positions": {"7": 0.1}, "invalid_collision": False},
        ],
    )
    assert result["task_success"] is True
    assert result["task_success_available"] is True


def test_manipulation_task_success_requires_grasp_lift_place_and_stability():
    spec = {
        "success_predicate": {
            "type": "manipulation_pick_place_stable",
            "target_instance_id": "bowl_1",
            "goal_region_id": "target_zone",
            "require_stable_after_place": True,
        }
    }
    result = evaluate_task_success(
        task_spec=spec,
        rollout_row={},
        state_trace=[
            {"grasp_acquired": True},
            {"lifted_clear": True},
            {"placed_in_target": True},
            {"stable_after_place": True},
        ],
    )
    assert result["task_success"] is True
    assert result["task_success_available"] is True


def test_task_success_falls_back_to_task_score_when_no_state_predicate_available():
    spec = {"success_predicate": {"type": "task_score_threshold", "threshold": 7.0}}
    result = evaluate_task_success(
        task_spec=spec,
        rollout_row={"task_score": 8.0},
        state_trace=[],
    )
    assert result["task_success"] is True
    assert result["task_success_available"] is False
