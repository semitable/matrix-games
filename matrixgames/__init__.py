from gym import register
import numpy as np

for k in (0, 25, 50, 75, 100):
    _payoff = np.array(2*[[
        [-k, 0, 10],
        [0, 2, 0],
        [10, 0, -k]],
    ])
    register(
        f"penalty-{k}-nostate-v0",
        entry_point="matrixgames.games:MatrixGame",
        kwargs={
            "payoff_matrix": _payoff,
            "ep_length": 1000,
            "last_action_state": False,
        }
    )
    register(
        f"penalty-{k}-laststate-v0",
        entry_point="matrixgames.games:MatrixGame",
        kwargs={
            "payoff_matrix": _payoff,
            "ep_length": 1000,
            "last_action_state": True,
        }
    )


_payoff_climbing = np.array(2*[[
    [11, -30, 0],
    [-30, 7, 0],
    [0, 6, 5],
]])
register(
    f"climbing-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_climbing,
        "ep_length": 1000,
        "last_action_state": False,
    }
)

register(
    f"climbing-laststate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_climbing,
        "ep_length": 1000,
        "last_action_state": True,
    }
)

_payoff_pdilemma = np.array([
    [[-1, -3],
     [-3, -2 ],],
    [[-1, -3 ],
     [-3, -2],]
])

register(
    f"pdilemma-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_pdilemma,
        "ep_length": 1000,
        "last_action_state": False,
    }
)

register(
    f"pdilemma-laststate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_pdilemma,
        "ep_length": 1000,
        "last_action_state": True,
    }
)


_payoff_climbing_3_players = np.array(3 * [[[
        [11, -30, 0],
        [-30, 0, 0],
        [0, 0, 0]] ,
        [[-30, 0, 0],
        [0, 7, 0],
        [0, 0, 0],],
        [[0, 0, 0],
        [0, 0, 0],
        [0, 6, 5],]]])
        
register(
    f"climbing-3p-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_climbing_3_players,
        "ep_length": 1000,
        "last_action_state": False,
    }
)