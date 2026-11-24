"""Stub for GAm01 (Dosukoi side): import compatibility only.

ga_native240 imports rod from this module at load time, but rod is only used
inside simulate(), which the GA351 blind scoring path never calls (the
generated ensembles are already committed). Calling it raises.
"""


def rod(*args, **kwargs):
    raise NotImplementedError(
        "ga_hybrid300.rod is not part of the GAm01 blind scoring path")
