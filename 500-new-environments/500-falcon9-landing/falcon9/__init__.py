"""Gymnasium-compatible Falcon 9 landing environment.

Importing this package registers ``Falcon9-v0`` with Gymnasium:

```python
import falcon9
import gymnasium as gym

env = gym.make("Falcon9-v0", dashboard=True)
```
"""

from falcon9.falcon9 import Falcon9, heuristic, register_falcon9_env

__all__ = ["Falcon9", "heuristic", "register_falcon9_env"]
