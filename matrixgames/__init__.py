from gym import register

for k in (0, 25, 50, 75, 100):
    _payoff = [
        [-k, 0, 10],
        [0, 2, 0],
        [10, 0, -k],
    ]
    register(
        f"penalty-{k}-nostate-v0",
        entry_point="matrixgames.games:MatrixGame",
        kwargs={
            "payoff_matrix": _payoff,
            "ep_length": 1,
            "last_action_state": False,
        }
    )
    register(
        f"penalty-{k}-laststate-v0",
        entry_point="matrixgames.games:MatrixGame",
        kwargs={
            "payoff_matrix": _payoff,
            "ep_length": 1,
            "last_action_state": True,
        }
    )


_payoff_climbing = [
    [11, -30, 0],
    [-30, 7, 0],
    [0, 6, 5],
]
register(
    f"climbing-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_climbing,
        "ep_length": 1,
        "last_action_state": False,
    }
)

register(
    f"climbing-laststate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_climbing,
        "ep_length": 1,
        "last_action_state": True,
    }
)
