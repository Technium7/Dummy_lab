"""Epoxy application: demo recording, vision, phased approach, and teleop apply.

Phases (run separately for debug)::

    python -m high_level_jobs.epoxy_app --phase pre_apply --ik-gui
    python -m high_level_jobs.epoxy_app --phase pre_apply --plan-gui
    python -m high_level_jobs.epoxy_app --phase pre_apply --execute --port COM3
    python -m high_level_jobs.epoxy_app --phase xy_optimize --execute
    python -m high_level_jobs.epoxy_app --phase z_creep --execute
"""
